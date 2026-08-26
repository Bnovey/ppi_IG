"""Gradient attribution for predicting mutation effects on binding scores.

``predict_mutants`` is additive by construction: the predicted score change for
a multi-point mutant is the sum of the per-substitution deltas.  Its error on
multi-point mutants therefore measures epistasis (non-additive interactions).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)

CANONICAL_AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")


def free_cuda_memory() -> None:
    """Release cached CUDA allocations between IG steps.

    Tolerates a GPU already in a bad state after an OOM, so that cleanup never
    masks the original error.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


@dataclass
class AttribResult:
    """Container for gradient-attribution outputs."""

    grad: Tensor
    ig: Tensor | None
    n_steps: int
    quadrature: str
    meta: dict = field(default_factory=dict)


def build_mean_aa_baseline(
    embeddings: Tensor,
    peptide_token_indices: np.ndarray,
    per_aa_peptide_embeddings: list[Tensor],
) -> Tensor:
    """Baseline = mean embedder output over 20 canonical homopolymer peptides.

    Non-peptide positions keep the actual ``embeddings`` so that attribution
    is zero there by construction.
    """
    if len(per_aa_peptide_embeddings) != len(CANONICAL_AMINO_ACIDS):
        raise ValueError(
            f"Expected {len(CANONICAL_AMINO_ACIDS)} per-AA tensors, "
            f"got {len(per_aa_peptide_embeddings)}"
        )
    mean_pep = torch.stack(per_aa_peptide_embeddings, dim=0).mean(dim=0)
    baseline = embeddings.detach().clone()
    idx = torch.as_tensor(
        peptide_token_indices, device=embeddings.device, dtype=torch.long
    )
    baseline[..., idx, :] = mean_pep.to(baseline.dtype)
    return baseline


def integrated_gradient(
    forward_fn: Callable[[Tensor], Tensor],
    embeddings: Tensor,
    baseline: Tensor | None = None,
    m_steps: int = 15,
    quadrature: str = "gausslegendre",
    clear_cache_each_step: bool = False,
    log_progress: bool = False,
) -> AttribResult:
    """Path-integral attribution via Gauss-Legendre or uniform quadrature.

    Parameters
    ----------
    forward_fn : callable
        fn(embeddings: Tensor[1, L, D]) -> scalar Tensor with grad enabled.
    embeddings : Tensor
        Input embedding, shape (1, L, D).
    baseline : Tensor or None
        Reference embedding, same shape.  Defaults to zeros.
    m_steps : int
        Number of quadrature points.
    quadrature : str
        ``"gausslegendre"`` or ``"uniform"``.
    clear_cache_each_step : bool
        Release cached CUDA allocations after each step.  Full-trunk Boltz-2
        backprop fills an 80 GiB card, so this is often required there and
        pointless everywhere else.
    log_progress : bool
        Emit a per-step INFO line.  A full-trunk run takes 20-40 minutes and
        is indistinguishable from a hang without this.

    Notes
    -----
    There is deliberately no ``save_on_cpu`` option.  Offloading saved
    activations to host RAM was measured at >18 minutes for a *single* IG step
    on this model because of PCIe paging; per-block gradient checkpointing in
    the ``forward_fn`` is the approach that actually works.

    Returns
    -------
    AttribResult
        ``.grad`` is the path-averaged gradient (the quantity to dot with
        embedding deltas); ``.ig`` is ``(x - baseline) * grad``.
    """
    if baseline is None:
        baseline = torch.zeros_like(embeddings)

    if quadrature == "gausslegendre":
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(m_steps)
        alphas_np = (gl_nodes + 1.0) / 2.0
        weights_np = gl_weights / 2.0
        alphas = torch.as_tensor(
            alphas_np, dtype=embeddings.dtype, device=embeddings.device
        )
        weights = torch.as_tensor(
            weights_np, dtype=embeddings.dtype, device=embeddings.device
        )
    elif quadrature == "uniform":
        alphas = torch.linspace(0.0, 1.0, m_steps + 1, device=embeddings.device)
        weights = torch.ones(
            len(alphas), dtype=embeddings.dtype, device=embeddings.device
        )
        weights = weights / len(alphas)
    else:
        raise ValueError(
            f"Unknown quadrature: {quadrature!r}. Use 'gausslegendre' or 'uniform'."
        )

    accumulated_grads = torch.zeros_like(embeddings)
    n_alphas = len(alphas)

    for step_i, (alpha, weight) in enumerate(zip(alphas, weights)):
        if log_progress:
            log.info("IG step %d/%d", step_i + 1, n_alphas)
        interp = (baseline + alpha * (embeddings - baseline)).detach().requires_grad_(True)
        with torch.enable_grad():
            score = forward_fn(interp)
            score.backward()
        accumulated_grads = accumulated_grads + weight * interp.grad.detach()
        del interp, score
        if clear_cache_each_step:
            free_cuda_memory()

    ig = (embeddings.detach() - baseline.detach()) * accumulated_grads

    return AttribResult(
        grad=accumulated_grads,
        ig=ig,
        n_steps=m_steps,
        quadrature=quadrature,
    )


def plain_gradient(
    forward_fn: Callable[[Tensor], Tensor],
    embeddings: Tensor,
    baseline: Tensor | None = None,
) -> AttribResult:
    """Single backward pass at the actual embedding (alpha=1).

    This is NOT the same as ``integrated_gradient(..., m_steps=1, quadrature="uniform")``,
    because that averages alpha=0 and alpha=1 (linspace(0,1,2) with equal weights).
    Here we evaluate the gradient at the real input only.
    """
    x = embeddings.detach().requires_grad_(True)
    with torch.enable_grad():
        score = forward_fn(x)
        score.backward()
    grad = x.grad.detach()

    ig = None
    if baseline is not None:
        ig = (embeddings.detach() - baseline.detach()) * grad

    return AttribResult(
        grad=grad,
        ig=ig,
        n_steps=1,
        quadrature="plain",
    )


def score_deltas(
    grad: Tensor | np.ndarray,
    delta_emb: dict[tuple[int, str], Tensor | np.ndarray],
) -> dict[tuple[int, str], float]:
    """Per-substitution predicted score changes via gradient dot embedding delta.

    For substitution a->b at position l the predicted effect is
    ``grad[l] @ (E_b[l] - E_a[l])``.

    This uses the *gradient*, not the IG product ``(x - baseline) * grad``.
    The ``(x - baseline)`` factor belongs to the completeness identity and
    must not be included when predicting substitution effects.

    Parameters
    ----------
    grad : array-like, shape (L, D)
        Gradient at positions of interest.
    delta_emb : dict[(position, aa)] -> array-like shape (D,)
        Embedding difference ``E_mutant[l] - E_reference[l]`` per substitution.

    Returns
    -------
    dict[(int, str), float]
    """
    if isinstance(grad, Tensor):
        grad = grad.detach().cpu().float()
    grad = np.asarray(grad, dtype=np.float64)
    L, D = grad.shape

    result: dict[tuple[int, str], float] = {}
    for (pos, aa), delta in delta_emb.items():
        if pos < 0 or pos >= L:
            raise IndexError(
                f"Position {pos} out of range for grad with L={L}"
            )
        d = np.asarray(delta, dtype=np.float64).ravel()
        if d.shape[0] != D:
            raise ValueError(
                f"Dimension mismatch at ({pos}, {aa!r}): "
                f"got {d.shape[0]}, expected {D}"
            )
        result[(pos, aa)] = float(grad[pos] @ d)
    return result


def predict_mutants(
    deltas: dict[tuple[int, str], float],
    substitutions: Sequence[tuple[tuple[int, str], ...]],
) -> np.ndarray:
    """Predict score changes for a list of mutants by additive summation.

    Parameters
    ----------
    deltas : dict[(position, aa), float]
        Per-substitution predicted effects from :func:`score_deltas`.
    substitutions : sequence of tuples of (position, aa)
        Each entry defines a mutant as a tuple of its substitutions.
        An empty tuple represents the reference (wild-type).

    Returns
    -------
    np.ndarray, shape (len(substitutions),)
        Predicted score changes.  The reference gives exactly 0.0.
    """
    out = np.empty(len(substitutions), dtype=np.float64)
    for i, subs in enumerate(substitutions):
        total = 0.0
        for key in subs:
            if key not in deltas:
                raise KeyError(
                    f"No delta for substitution {key!r}; "
                    f"available keys: {sorted(deltas.keys())}"
                )
            total += deltas[key]
        out[i] = total
    return out


def completeness_error(result: AttribResult, f_x: float, f_baseline: float) -> float:
    """Relative error of the completeness identity: sum(ig) vs (f(x) - f(baseline)).

    Parameters
    ----------
    result : AttribResult
        Must have a non-None ``.ig`` field.
    f_x : float
        Scalar output at the input embedding.
    f_baseline : float
        Scalar output at the baseline embedding.

    Returns
    -------
    float
        ``|sum(ig) - (f_x - f_baseline)| / |f_x - f_baseline|``.
        Returns 0.0 if the denominator is zero and the numerator is also zero.
    """
    if result.ig is None:
        raise ValueError("AttribResult.ig is None; cannot compute completeness.")
    ig_sum = float(result.ig.sum())
    diff = f_x - f_baseline
    if abs(diff) == 0.0:
        return 0.0 if abs(ig_sum) == 0.0 else float("inf")
    return abs(ig_sum - diff) / abs(diff)
