"""Gradient attribution for predicting mutation effects on binding scores.

``predict_mutants`` is additive by construction: the predicted score change for
a multi-point mutant is the sum of the per-substitution deltas.  Its error on
multi-point mutants therefore measures epistasis (non-additive interactions).
"""

from __future__ import annotations

import contextlib
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


# Only tensors at least this large are moved to host RAM.  At 730 tokens the
# pair representation z is (1, 730, 730, 128) fp32 = 730*730*128*4 =
# 272_844_800 B = 260.2 MiB and clears the bar; at 500 tokens it is
# 128_000_000 B = 122.1 MiB and stays on the GPU.  The threshold makes the
# offload self-scaling: it engages only on the complexes that need it.
#
# Note this threshold is calibrated on *fp32* z.  Under bf16 the same tensor
# is 130.1 MiB, i.e. BELOW the 200 MiB bar, so the offload would silently stop
# engaging at 730 tokens; a bf16 run wanting it must lower min_bytes.
DEFAULT_OFFLOAD_MIN_BYTES = 200 * 1024**2


@contextlib.contextmanager
def offload_large_saved_tensors(
    min_bytes: int = DEFAULT_OFFLOAD_MIN_BYTES,
    pin_memory: bool = True,
):
    """Keep only *large* saved tensors in host RAM during backward.

    This is NOT ``torch.autograd.graph.save_on_cpu``, and the difference is the
    whole point.  Blanket ``save_on_cpu`` offloads *every* saved activation; the
    predecessor repo measured a single IG step at >18 minutes under it
    ("effectively hung") because of the PCIe traffic, which is why
    :func:`integrated_gradient` still refuses to offer it.

    Here a size threshold is applied, so only the checkpoint *boundary* tensors
    move.  A memory profile of the 730-token 4fqi complex attributed 34.30 GiB
    of a 75.28 GiB peak to 135 such blocks (``pairformer.py:102``, one per
    trunk block per recycling iteration plus the confidence layers), each 254
    MiB.  Moving those and nothing else is ~34 GiB each way per backward --
    seconds over PCIe, not the hundreds of GiB that made blanket offload
    unusable.

    Must wrap ``backward()``, not just the forward call: the trunk is inside a
    reentrant checkpoint, so the tensors are saved during the recompute that
    the autograd engine drives, long after the forward has returned.

    Numerically exact -- tensors are restored bit-for-bit.

    .. warning::
       **Defaults to off, because it does not reach the tensors that matter.**
       Measured on the 730-token complex: peak fell only 77.64 -> 76.50 GiB.
       The 34.3 GiB of checkpoint boundary tensors are held by
       ``torch.utils.checkpoint``'s NON-reentrant path in a Python *closure*,
       not via ``save_for_backward``, so ``saved_tensors_hooks`` never sees
       them. Switching the inner block checkpoints to reentrant does route
       them through the hooks, but breaks the gradient outright --
       ``UserWarning: None of the inputs have requires_grad=True. Gradients
       will be None`` -- because the outer checkpoint runs its forward under
       ``no_grad``, so block inputs carry no ``requires_grad``. That attempt
       also drove host RSS to 165 GB and was killed by the kernel OOM killer.

       Kept because it is correct and cheap where saves *do* go through
       ``save_for_backward``, and so the next person does not re-derive this.
    """
    offloaded = {"count": 0, "bytes": 0}

    def pack(t: Tensor):
        if not t.is_cuda or t.numel() * t.element_size() < min_bytes:
            return t
        host = torch.empty(
            t.size(), dtype=t.dtype, layout=t.layout,
            device="cpu", pin_memory=pin_memory,
        )
        host.copy_(t)
        offloaded["count"] += 1
        offloaded["bytes"] += t.numel() * t.element_size()
        return (t.device, host)

    def unpack(payload):
        if isinstance(payload, tuple):
            device, host = payload
            return host.to(device, non_blocking=True)
        return payload

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield offloaded

    if offloaded["count"]:
        log.info(
            "offloaded %d saved tensors (%.1f GiB) to host RAM",
            offloaded["count"], offloaded["bytes"] / 1024**3,
        )


def integrated_gradient(
    forward_fn: Callable[[Tensor], Tensor],
    embeddings: Tensor,
    baseline: Tensor | None = None,
    m_steps: int = 15,
    quadrature: str = "gausslegendre",
    clear_cache_each_step: bool = False,
    log_progress: bool = False,
    offload_min_bytes: int | None = None,
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

    offload_min_bytes : int or None
        Move saved tensors at least this large to host RAM during backward
        (``None`` disables).  See :func:`offload_large_saved_tensors`.

    Notes
    -----
    There is deliberately no blanket ``save_on_cpu`` option.  Offloading *every*
    saved activation to host RAM was measured at >18 minutes for a *single* IG
    step on this model because of PCIe paging.  Per-block gradient checkpointing
    in the ``forward_fn`` is the approach that actually works, and
    ``offload_min_bytes`` is the narrow, size-thresholded complement to it --
    it moves only the ~254 MiB checkpoint boundary tensors, which are 34.3 GiB
    of a 75.3 GiB peak on a 730-token complex.

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
        # dtype=embeddings.dtype, matching ``weights`` below and the
        # gausslegendre branch above: without it linspace silently takes the
        # *global* default dtype, so a caller running under
        # ``torch.set_default_dtype`` (or a non-fp32 embedding) would
        # interpolate at a different precision than it weights with. No
        # behaviour change at the repo's fp32 default -- the two dtypes
        # already coincide there.
        alphas = torch.linspace(
            0.0, 1.0, m_steps + 1,
            dtype=embeddings.dtype, device=embeddings.device,
        )
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
        # The offload context must span backward(), not just forward(): the
        # trunk sits inside a reentrant checkpoint, so its tensors are saved
        # during the autograd engine's recompute.
        ctx = (
            offload_large_saved_tensors(offload_min_bytes)
            if offload_min_bytes is not None
            else contextlib.nullcontext()
        )
        with torch.enable_grad(), ctx:
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
    offload_min_bytes: int | None = None,
) -> AttribResult:
    """Single backward pass at the actual embedding (alpha=1).

    This is NOT the same as ``integrated_gradient(..., m_steps=1, quadrature="uniform")``,
    because that averages alpha=0 and alpha=1 (linspace(0,1,2) with equal weights).
    Here we evaluate the gradient at the real input only.
    """
    x = embeddings.detach().requires_grad_(True)
    ctx = (
        offload_large_saved_tensors(offload_min_bytes)
        if offload_min_bytes is not None
        else contextlib.nullcontext()
    )
    with torch.enable_grad(), ctx:
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


def make_dead_target(
    forward_fn: Callable[[Tensor], Tensor],
    weight: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Build a gradient-path integrity objective whose value is real evidence.

    The objective is ``f(x.detach()) + weight * x.sum()``, with the real
    ``forward_fn`` evaluated under :func:`torch.no_grad`.  At the default
    ``weight=0.0`` its gradient w.r.t. ``x`` is exactly zero, so attributing it
    must return an all-zero gradient -- but its *value* is the genuine model
    score, so the recorded number moves when the model does.

    Why this replaces a hand-written ``(x * 0.0).sum() + 1.0``: that objective
    is a PyTorch tautology.  Its gradient is analytically zero no matter what
    the model, the trunk, the checkpointing or the score selection do, and its
    value is the constant 1.0, so the only failure it can detect is an autograd
    bug in scalar multiply-by-zero.  Here the same forward the production
    attribution uses is actually executed, so a silently-zero or constant score
    (boltz wraps ``compute_ptms`` in a bare ``except`` that returns zeros) shows
    up in the recorded value.

    A NON-zero ``max|grad|`` from this objective means autograd reached ``x``
    through a path that does not exist in the returned expression: an in-place
    write into a tensor derived from ``x``, or attribution machinery handing
    back a stale or aliased ``.grad`` buffer.  That last one is the live
    regression risk here -- :func:`integrated_gradient` accumulates into one
    buffer across quadrature nodes and drops its leaf each step.

    Cost: **one no-grad forward per quadrature node, and no full-trunk
    backward.**  The ``no_grad`` + ``detach`` is mandatory, not cosmetic:

    * it keeps the backward on the tiny ``weight * x.sum()`` branch, so the
      check cannot OOM the way a real attribution does;
    * ``confidence_forward`` asserts ``scalar.requires_grad`` whenever grad is
      enabled (``src/igv/boltz_score.py:745-749``), which running the real
      forward on a detached input under grad would trip.

    For scale: this is comparable to ``signal_control``'s 30 no-grad forwards
    in ``scripts/07_sanity.py``, which complete fine on runs where every
    gradient check OOMs.

    Parameters
    ----------
    forward_fn : callable
        The real objective, ``fn(embeddings) -> scalar Tensor``.  It receives a
        **detached** tensor, and is called inside ``torch.no_grad()``.
    weight : float
        Coefficient of the graph-connecting ``x.sum()`` term.  ``0.0`` (the
        default) is the dead-target check.  A non-zero value turns the same
        helper into a *positive* control: the gradient is then analytically
        ``weight`` at every element, so a zero reading indicts the attribution
        plumbing rather than the model.

    Returns
    -------
    callable
        ``objective(x) -> 0-dim Tensor``, safe to pass to
        :func:`integrated_gradient` or :func:`plain_gradient`.
    """

    def _dead(x: Tensor) -> Tensor:
        with torch.no_grad():
            value = forward_fn(x.detach())
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)
        if value.numel() != 1:
            raise ValueError(
                f"make_dead_target expects a scalar forward_fn, got shape "
                f"{tuple(value.shape)}"
            )
        # reshape(()) normalises to 0-dim so .backward() needs no grad_output,
        # matching the contract integrated_gradient already relies on.
        value = value.detach().reshape(()).to(device=x.device, dtype=x.dtype)
        return value + weight * x.sum()

    return _dead


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
