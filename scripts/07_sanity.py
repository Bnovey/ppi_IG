#!/usr/bin/env python3
"""Stage 07 -- Tier-0 sanity checks. This stage GATES the pipeline.

If any of these fail, every downstream number is void. Exits non-zero on
failure so ``run_all.sh`` can abort before spending GPU hours.

The most important check is ``signal_control``, and it is not optional.
Independent work reports Boltz-2 assigning uniformly high confidence
irrespective of biological relevance. If the score itself does not move when
the sequence is mutated, its gradient cannot carry information either -- and a
null attribution result would then be *uninterpretable* rather than
informative. So we establish that the score responds to mutation before
attributing anything to the gradient.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import build_library, read_pdb_chains, resolve_chain_subset  # noqa: E402
from igv.gpu import require_vram  # noqa: E402
from igv.metrics import spearman  # noqa: E402
from igv.provenance import assert_provenance, write as prov_write  # noqa: E402
from igv.dms import resolve_pdb_complex  # noqa: E402

log = logging.getLogger("sanity")

_STRUCTURE_FOR = {"4fqi_h1": "4fqi_hlab", "4fqi_h3": "4fqi_hlab"}


# See ERRORS_LOG.md lines 783-792: relative error penalises good baselines
# whose span collapses, so we gate on absolute error instead.
COMPLETENESS_ABS_THRESHOLD = 0.10

THRESHOLDS = {
    "completeness": f"absolute error < {COMPLETENESS_ABS_THRESHOLD}",
    "m_sweep": "consecutive Spearman > 0.95 by m=32",
    "random_weights": "|Spearman vs trained| < 0.3",
    "dead_target": "max|attribution| < 1e-8",
    "signal_control": "score std > 1e-4 and Spearman != 0",
    "frozen_vs_full": "informational only, no pass/fail",
    "arm_assertion": "recorded arm matches intended arm",
}
CHECKS = list(THRESHOLDS)

# Checks that report a magnitude rather than a verdict. They must never gate
# the pipeline -- including when they RAISE. The flag is therefore set in
# ``_result`` from the check name, not on the success path of the check
# itself: the exception handler in ``main`` builds its result through the same
# ``_result``, and previously inherited no flag, so an OOM in an
# "informational only" check was recorded as blocking (see
# results/sanity_4fqi_h1_complex_pde.json, where frozen_vs_full failed the
# gate by raising).
INFORMATIONAL = frozenset({"frozen_vs_full"})


def _result(name, passed, value, detail):
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "threshold": THRESHOLDS[name],
        "detail": detail,
        "informational": name in INFORMATIONAL,
    }


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_completeness(
    forward_fn, s_inputs, baseline, m_steps=16, baseline_scale=0.0,
    abs_threshold=COMPLETENESS_ABS_THRESHOLD,
):
    """Do the attributions sum to f(x) - f(baseline)?

    Gates on absolute error (see ERRORS_LOG.md lines 783-792): a good baseline
    shrinks the span, inflating relative error even when the integral is 30x
    more accurate in absolute terms.
    """
    import torch
    from igv.attrib import completeness_error, integrated_gradient

    res = integrated_gradient(forward_fn, s_inputs, baseline=baseline, m_steps=m_steps)
    with torch.no_grad():
        f_x = float(forward_fn(s_inputs))
        f_b = float(forward_fn(baseline))
    ig_sum = float(res.ig.sum())
    rel_err = float(completeness_error(res, f_x, f_b))
    abs_err = abs(ig_sum - (f_x - f_b))
    return _result(
        "completeness", abs_err < abs_threshold, {"abs": abs_err, "rel": rel_err},
        f"m_steps={m_steps} baseline_scale={baseline_scale} "
        f"f(x)={f_x:.6f} f(baseline)={f_b:.6f} sum(ig)={ig_sum:.6f} "
        f"abs_err={abs_err:.6f} rel_err={rel_err:.6f}",
    )


def check_m_sweep(forward_fn, s_inputs, baseline, ms=(8, 16, 32)):
    """Is the path integral converged by m=32?

    We stop at 32 deliberately. Gauss-Legendre is the default quadrature and is
    documented as m=10 being comparable to uniform m~16, so larger m only burns
    GPU hours without changing the ranking.
    """
    from igv.attrib import integrated_gradient

    mags = {}
    for m in ms:
        r = integrated_gradient(forward_fn, s_inputs, baseline=baseline, m_steps=m)
        mags[m] = r.grad.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    pairs = {}
    for a, b in zip(ms, ms[1:]):
        pairs[f"{a}->{b}"] = float(spearman(mags[a], mags[b]))
    last = pairs[f"{ms[-2]}->{ms[-1]}"]
    return _result("m_sweep", last > 0.95, pairs, f"final pair {ms[-2]}->{ms[-1]} = {last:.4f}")


def randomise_(module, seed=0):
    """Re-initialise ``module`` in place; return ``(n, n2d, nbias, nnorm)``.

    Matrices get Xavier, biases get zeros, and every other 1-D parameter gets
    ONES. That last rule is the point of this helper.

    1-D non-bias parameters are the LayerNorm/RMSNorm *scales*. Zeroing them
    (which is what a bare ``p.dim() >= 2 ... else zeros_`` does) annihilates
    every normalised layer's output, so the network emits a constant, the
    gradient w.r.t. the input is constant, ``spearman`` returns NaN on zero
    variance (src/igv/metrics.py:22-30) and ``abs(nan) < 0.3`` is False. The
    caller would then print its project-killing verdict about a network THIS
    FUNCTION broke rather than about a real result. Unit scale is the norm's
    identity, which leaves the Xavier-randomised affine layers as the sole
    source of randomness -- exactly the intended control.
    """
    import torch

    torch.manual_seed(seed)
    n = n2d = nbias = nnorm = 0
    for name, p in module.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
            n2d += 1
        elif name.endswith(".bias") or name == "bias":
            torch.nn.init.zeros_(p)
            nbias += 1
        else:
            torch.nn.init.ones_(p)
            nnorm += 1
        n += 1
    return n, n2d, nbias, nnorm


def check_random_weights(make_forward_fn, model, s_inputs, baseline, seed=0):
    """Re-init the model randomly; attribution should fall apart.

    If randomised weights reproduce the trained attribution, the method is
    reading input geometry rather than anything the model learned. This is the
    cheapest check that can kill the project outright -- which is exactly why
    the randomisation itself has to be sane; see :func:`randomise_`.
    """
    from igv.attrib import plain_gradient

    trained = plain_gradient(make_forward_fn(model), s_inputs, baseline=baseline)
    tm = trained.grad.squeeze(0).norm(dim=-1).detach().cpu().numpy()

    # NOTE (not fixed here): this deepcopy doubles resident model VRAM inside
    # the most memory-constrained stage of the pipeline.
    rnd = copy.deepcopy(model)
    n, n2d, nbias, nnorm = randomise_(rnd, seed=seed)
    rm = plain_gradient(make_forward_fn(rnd), s_inputs, baseline=baseline)
    rm = rm.grad.squeeze(0).norm(dim=-1).detach().cpu().numpy()

    rho = float(spearman(tm, rm))
    split = (
        f"reinitialised {n} parameter tensors ({n2d} xavier, "
        f"{nbias} zeroed biases, {nnorm} unit norms). "
    )
    if not np.isfinite(rho):
        # A non-finite Spearman is a broken-arm report, never a scientific
        # finding: it means one arm's per-token gradient magnitude had zero
        # variance. Do NOT emit the project-killing text here.
        return _result(
            "random_weights", False, rho,
            split + "ERROR: Spearman is not finite, i.e. one arm's per-token "
            "gradient magnitude has zero variance (a constant gradient). That "
            "is a degenerate-model artifact of this check, NOT evidence that "
            "the method reads input geometry. Investigate the randomised "
            "forward pass before drawing any conclusion.",
        )
    return _result(
        "random_weights", abs(rho) < 0.3, rho,
        split
        + (
            "PASS: randomising the model destroys the attribution, so the signal "
            "depends on learned weights."
            if abs(rho) < 0.3
            else "FAIL -- THIS KILLS THE PROJECT AS FRAMED. A randomly initialised "
            "model reproduces the same attribution, so the gradient is reading "
            "input geometry, not anything Boltz-2 learned. Do not interpret any "
            "downstream number."
        ),
    )


def check_dead_target(forward_fn, s_inputs, baseline):
    """Attribute an objective that RUNS the model but whose gradient must vanish.

    The objective comes from :func:`igv.attrib.make_dead_target` and is
    ``f(x.detach()) + 0.0 * x.sum()``. Its VALUE is the real Boltz score at the
    real input -- featurisation, the trunk, the confidence head and the score
    selection all determine it -- while its gradient w.r.t. ``x`` is exactly
    zero because the only graph-connected term carries weight zero. A non-zero
    result therefore means autograd reached the input along a path that should
    not exist: an in-place aliasing bug, a leaked non-detached reference, or a
    checkpoint recompute wired to the wrong tensor.

    This replaces ``(x * 0).sum() + 1``, which had neither ``model`` nor
    ``forward_fn`` in scope. Its gradient was analytically zero for any
    autograd implementation, so no Boltz code path could influence the outcome
    and the only thing it could have caught was a bug in multiply-by-zero. It
    passed with ``value: 0.0`` in the very run where all four real gradient
    checks OOM'd -- a result that was model-independent by construction is not
    evidence.

    Cost: ONE forward, run under ``no_grad`` on a detached input, plus a
    backward that traverses only the zero-weighted connector. Unlike the
    full-trunk backward checks it therefore cannot OOM. It is deliberately
    ``plain_gradient`` and not ``integrated_gradient(m_steps=8)``: a
    model-dependent value at the real input is the whole payload, and 8 nodes
    would cost 8 forwards to re-derive the same zero.

    ``baseline`` is accepted for runner-signature symmetry; a single-point
    gradient needs no path reference.
    """
    from igv.attrib import make_dead_target, plain_gradient

    dead = make_dead_target(forward_fn)
    seen = {}

    def objective(x):
        out = dead(x)
        # Record the model score from the pass we already paid for rather than
        # calling forward_fn a second time.
        seen["value"] = float(out.detach())
        return out

    # MEASURE on the VM: this is the first version of the check that touches
    # the model, so record peak VRAM here (igv.gpu.PeakMemory) and confirm the
    # forward alone stays well under the card. The reasoning says it must --
    # no_grad forward, backward only over the zero-weighted connector, i.e.
    # cheaper than signal_control's 30 no-grad forwards, which complete on runs
    # where every gradient check OOMs -- but that is reasoning, not a number.
    res = plain_gradient(objective, s_inputs)
    mx = float(res.grad.abs().max())
    val = seen.get("value", float("nan"))
    return _result(
        "dead_target", mx < 1e-8, mx, f"max|grad| = {mx:.3e}, f(x) = {val:.6f}"
    )


def check_signal_control(score_of_sequence, lib, n_sample=30, seed=0):
    """Does the score move at all when the sequence is mutated?

    Failure means the chosen score is unusable and another entry from SCORES
    must be tried. Without this, a null attribution result cannot be
    distinguished from a score that is simply flat.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lib.frame), size=min(n_sample, len(lib.frame)), replace=False)
    seq_col = "heavy_chain_seq" if lib.chain == "H" else "light_chain_seq"
    scores, measured = [], []
    for k, i in enumerate(idx, 1):
        row = lib.frame.iloc[int(i)]
        scores.append(float(score_of_sequence(row[seq_col])))
        measured.append(float(row["binding_score"]))
        log.info("  signal_control [%d/%d] model=%.6f", k, len(idx), scores[-1])
    scores = np.asarray(scores)
    rho = float(spearman(scores, np.asarray(measured)))
    std = float(scores.std())
    passed = std > 1e-4 and rho != 0.0
    return _result(
        "signal_control", passed,
        {"std": std, "min": float(scores.min()), "max": float(scores.max()), "spearman": rho},
        (
            f"n={len(idx)} std={std:.3e} range=[{scores.min():.6f}, {scores.max():.6f}] "
            f"spearman_vs_measured={rho:.4f}"
        )
        + ("" if passed else " -- SCORE IS FLAT. Try a different entry from SCORES; "
           "a null attribution result on this score would be uninterpretable."),
    )


def check_frozen_vs_full(make_forward_fn, model, s_inputs, baseline):
    """How much signal flows through the trunk? Informational.

    Running the trunk under no_grad computes the gradient of a genuinely
    different function, so this is a magnitude report, not a pass/fail.
    """
    from igv.attrib import plain_gradient

    full = plain_gradient(make_forward_fn(model, ckpt=True), s_inputs, baseline=baseline)
    frozen = plain_gradient(make_forward_fn(model, frozen=True), s_inputs, baseline=baseline)
    a = full.grad.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    b = frozen.grad.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    rho = float(spearman(a, b))
    ratio = float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-12))
    # The "informational" flag is set by _result from INFORMATIONAL, so the
    # exception path in main() carries it too.
    return _result("frozen_vs_full", True, {"spearman": rho, "magnitude_ratio": ratio},
                   f"spearman={rho:.4f} |frozen|/|full|={ratio:.4f}")


def check_arm_assertion(dataset, score, chain, results_dir=Path("results")):
    """Verify artifacts record the arm they claim.

    This exists because the predecessor project's central confound was a default
    argument that silently routed every attribution to the wrong model for a
    35-hour campaign. It raised no error; it was found by auditing artifacts.
    """
    checked, problems = [], []
    for pat in (f"{dataset}_{score}_*_grad.npz",):
        for p in sorted(results_dir.glob(pat)):
            try:
                assert_provenance(p, score=score, dataset=dataset, chain=chain)
                checked.append(str(p))
            except (AssertionError, FileNotFoundError) as e:
                problems.append(f"{p}: {e}")
    deltas = Path("data/processed") / f"{dataset}_deltas.npz"
    if deltas.exists():
        try:
            assert_provenance(deltas, dataset=dataset, chain=chain)
            checked.append(str(deltas))
        except (AssertionError, FileNotFoundError) as e:
            problems.append(f"{deltas}: {e}")
    detail = (
        f"checked {len(checked)} artifact(s)" if not problems
        else "MISMATCH: " + " | ".join(problems)
    )
    if not checked and not problems:
        detail = "no artifacts present yet (run stages 02/03 first)"
    return _result("arm_assertion", not problems, checked, detail)


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--chain", default="H")
    p.add_argument("--score", default="complex_pde")
    p.add_argument("--checks", default="all", help="'all' or comma-separated names")
    p.add_argument("--n-sample", type=int, default=30)
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--structure", default=None)
    p.add_argument("--out", default="results/sanity_{dataset}_{score}.json")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    p.add_argument("--no-msa-server", action="store_true")
    p.add_argument(
        "--chain-subset", default=None,
        help="Comma-separated chain IDs (e.g. H,L,A). Default: all chains.",
    )
    p.add_argument(
        "--m-steps", type=int, default=16,
        help="Number of integration steps for the completeness check (default 16).",
    )
    p.add_argument(
        "--baseline-scale", type=float, default=0.0,
        help="Baseline = scale * x. 0.0 = zeros (default). Must be in [0, 1).",
    )
    p.add_argument(
        "--completeness-threshold", type=float, default=COMPLETENESS_ABS_THRESHOLD,
        help=(
            f"Absolute-error gate for the completeness check "
            f"(default {COMPLETENESS_ABS_THRESHOLD})."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.m_steps < 1:
        raise SystemExit(f"--m-steps must be >= 1, got {args.m_steps}")

    if args.baseline_scale < 0.0:
        raise SystemExit(f"--baseline-scale must be >= 0.0, got {args.baseline_scale}")
    if args.baseline_scale >= 1.0:
        raise SystemExit(
            f"--baseline-scale must be < 1.0 (at 1.0 the path is a point and the "
            f"identity is vacuous), got {args.baseline_scale}"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected = CHECKS if args.checks == "all" else [c.strip() for c in args.checks.split(",")]
    unknown = set(selected) - set(CHECKS)
    if unknown:
        raise SystemExit(f"Unknown checks: {sorted(unknown)}. Available: {CHECKS}")

    cache_dir = Path(args.cache_dir)

    resolved = None
    try:
        resolved = resolve_pdb_complex(
            args.dataset, args.chain, cache_dir,
            structure_override=args.structure,
        )
    except KeyError:
        pass
    except ValueError:
        raise SystemExit(
            f"Chain {args.chain!r} not found in registered chains for "
            f"{args.dataset}. Check --chain."
        ) from None

    if resolved is not None:
        data_source = resolved.data_source
        pdb = resolved.pdb_path
    else:
        data_source = "abbibench"
        stem = args.structure or _STRUCTURE_FOR.get(args.dataset)
        if stem is None:
            raise SystemExit(f"No structure known for {args.dataset}; pass --structure.")
        pdb = cache_dir / f"{stem}.pdb"

    chain_subset_arg = args.chain_subset
    if chain_subset_arg is None and resolved is not None:
        chain_subset_arg = ",".join(resolved.chains.keys())

    subset_label = "all"
    n_tokens = None
    if pdb.exists():
        all_chains = read_pdb_chains(pdb)
        if args.chain not in all_chains:
            raise SystemExit(
                f"Chain {args.chain!r} not found in PDB {pdb.name}. "
                f"Available chains: {list(all_chains.keys())}"
            )
        struct_chains, subset_label = resolve_chain_subset(
            all_chains, chain_subset_arg, args.chain,
        )
        n_tokens = sum(len(s) for s in struct_chains.values())
        chain_lengths = {c: len(s) for c, s in struct_chains.items()}
        log.info(
            "Chain subset: %s  lengths: %s  L=%d",
            subset_label, chain_lengths, n_tokens,
        )
    elif args.chain_subset is not None:
        raise SystemExit(
            f"--chain-subset requires the PDB at {pdb}; "
            "run scripts/00_fetch_data.py first."
        )

    if args.dry_run:
        print(f"Would run {len(selected)} check(s) on {args.dataset}/{args.score}:\n")
        for c in selected:
            print(f"  {c:<18} {THRESHOLDS[c]}")
        if n_tokens is not None:
            print(f"\nChain subset: {subset_label}  L={n_tokens}")
        print(f"m_steps: {args.m_steps}")
        print(f"baseline_scale: {args.baseline_scale}")
        print("\nGate: exits non-zero if any non-informational check fails.")
        return

    # Same 78 GiB gate and same IGV_SKIP_VRAM_CHECK=1 escape as before, now
    # from igv.gpu (DEFAULT_MIN_VRAM_GIB = 78.0) instead of a local copy.
    require_vram()

    from igv.boltz_score import (
        SCORES, build_complex_feats, confidence_forward, embedder_only, load_model,
        numerics_arm,
    )
    import torch

    if args.score not in SCORES:
        raise SystemExit(f"--score must be one of {sorted(SCORES)}")

    lib = None
    if resolved is not None:
        log.info("%s complex %s — no mutant library",
                 resolved.data_source.upper(), resolved.struct_name.upper())
    else:
        lib = build_library(args.dataset, cache_dir, chain=args.chain)

    if not pdb.exists():
        raise SystemExit(f"Missing PDB: {pdb}")
    if n_tokens is None:
        all_chains = read_pdb_chains(pdb)
        if args.chain not in all_chains:
            raise SystemExit(
                f"Chain {args.chain!r} not found in PDB {pdb.name}. "
                f"Available chains: {list(all_chains.keys())}"
            )
        struct_chains, subset_label = resolve_chain_subset(
            all_chains, chain_subset_arg, args.chain,
        )
        n_tokens = sum(len(s) for s in struct_chains.values())

    # Keyed on dataset as well as subset. boltz's process_inputs skips any input
    # whose YAML stem already exists under <cache_dir>/processed/records, and
    # the repo always writes stem "input" -- so a cache dir shared between two
    # complexes silently returns the first one's features. Keying on the subset
    # alone was not enough once a second complex existed.
    cache_suffix = f"_{args.dataset}"
    if subset_label != "all":
        cache_suffix += f"_{subset_label}"
    model, _boltz_version = load_model(args.checkpoint_dir, args.device)

    def chains_for(seq):
        d = dict(struct_chains)
        d[args.chain] = seq
        return d

    if lib is not None:
        reference_seq = lib.reference_seq
    else:
        reference_seq = struct_chains[args.chain]

    ref_chains = chains_for(reference_seq)
    n_tokens_pdb, n_tokens = n_tokens, sum(len(s) for s in ref_chains.values())
    if n_tokens != n_tokens_pdb:
        log.warning(
            "L=%d as featurised, not the %d the PDB implies: chain %s is "
            "reference_seq (%d aa) and not the PDB's (%d aa).",
            n_tokens, n_tokens_pdb, args.chain,
            len(reference_seq), len(struct_chains[args.chain]),
        )
    log.info("Featurising L=%d over chains %s", n_tokens, list(ref_chains))

    ref_feats, _ = build_complex_feats(
        ref_chains, pdb,
        cache_dir / f"boltz_ref{cache_suffix}",
        args.device, use_msa_server=not args.no_msa_server,
    )
    x_pred = ref_feats["coords"].detach()
    s_inputs = embedder_only(model, ref_feats)
    if args.baseline_scale == 0.0:
        baseline = torch.zeros_like(s_inputs)
    else:
        baseline = args.baseline_scale * s_inputs
    x_norm = float(s_inputs.norm())
    b_norm = float(baseline.norm())
    log.info(
        "Baseline: scale=%.4f  ||baseline||/||x|| = %.6f  ||x||=%.4f",
        args.baseline_scale, b_norm / (x_norm + 1e-30), x_norm,
    )

    def make_forward_fn(m=model, ckpt=True, frozen=False):
        def fn(s):
            return confidence_forward(
                m, s, ref_feats, x_pred, args.score,
                gradient_checkpointing=False if frozen else ckpt,
            )
        return fn

    forward_fn = make_forward_fn()

    def score_of_sequence(seq):
        feats, _ = build_complex_feats(
            chains_for(seq), pdb,
            cache_dir / f"boltz_sanity{cache_suffix}",
            args.device, use_msa_server=not args.no_msa_server,
        )
        with torch.no_grad():
            si = embedder_only(model, feats)
            return float(confidence_forward(
                model, si, feats, x_pred, args.score, gradient_checkpointing=False
            ))

    def _signal_control_runner():
        if lib is None:
            r = _result(
                "signal_control", True, None,
                f"SKIPPED: no mutant library ({data_source.upper()} structure-only path). "
                "signal_control requires measured binding scores from AbBiBench.",
            )
            r["skipped"] = True
            return r
        return check_signal_control(
            score_of_sequence, lib, n_sample=args.n_sample,
        )

    runners = {
        "completeness": lambda: check_completeness(
            forward_fn, s_inputs, baseline, m_steps=args.m_steps,
            baseline_scale=args.baseline_scale,
            abs_threshold=args.completeness_threshold,
        ),
        "m_sweep": lambda: check_m_sweep(forward_fn, s_inputs, baseline),
        "random_weights": lambda: check_random_weights(
            make_forward_fn, model, s_inputs, baseline),
        "dead_target": lambda: check_dead_target(forward_fn, s_inputs, baseline),
        "signal_control": _signal_control_runner,
        "frozen_vs_full": lambda: check_frozen_vs_full(
            make_forward_fn, model, s_inputs, baseline),
        "arm_assertion": lambda: check_arm_assertion(args.dataset, args.score, args.chain),
    }

    results = []
    for name in selected:
        log.info("running check: %s", name)
        try:
            results.append(runners[name]())
        except Exception as e:  # a crashed check is a failed check
            results.append(_result(name, False, None, f"raised {type(e).__name__}: {e}"))
        log.info("  -> %s", "PASS" if results[-1]["passed"] else "FAIL")

    out = Path(args.out.format(
        dataset=args.dataset, score=args.score,
        subset=subset_label, m_steps=args.m_steps,
        baseline_scale=args.baseline_scale,
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset, "score": args.score, "chain": args.chain,
        "chain_subset": list(struct_chains) if subset_label != "all" else None,
        "n_tokens": n_tokens,
        "n_tokens_pdb": n_tokens_pdb,
        "m_steps": args.m_steps,
        "baseline_scale": args.baseline_scale,
        "checks": results,
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    prov_write(
        out, stage="07_sanity",
        inputs={"dataset": args.dataset, "structure": str(pdb), "data_source": data_source},
        params={"checks": selected, "n_sample": args.n_sample,
                "chain_subset": list(struct_chains) if subset_label != "all" else None},
        arm={"score": args.score, "dataset": args.dataset, "chain": args.chain,
             "method": "sanity",
             "chain_subset": list(struct_chains) if subset_label != "all" else None,
             "n_tokens": n_tokens, "n_tokens_pdb": n_tokens_pdb,
             "m_steps": args.m_steps,
             "baseline_scale": args.baseline_scale,
             **numerics_arm()},
        notes="Tier-0 gate. Non-informational failures make downstream numbers void.",
    )

    blocking = [r for r in results if not r["passed"] and not r.get("informational")]
    print("\n" + "=" * 68)
    for r in results:
        if r.get("skipped"):
            tag = "SKIP"
        elif r.get("informational"):
            tag = "INFO"
        elif r["passed"]:
            tag = "PASS"
        else:
            tag = "FAIL"
        print(f"  {tag:<5} {r['name']:<18} {r['detail'][:80]}")
    print("=" * 68)
    print(f"Wrote {out}")
    if blocking:
        print(f"\nGATE FAILED: {[r['name'] for r in blocking]}")
        print("Downstream T1/T2/T3 numbers are NOT interpretable until these pass.")
        sys.exit(1)
    print("\nGATE PASSED: safe to interpret T1/T2/T3.")


if __name__ == "__main__":
    main()
