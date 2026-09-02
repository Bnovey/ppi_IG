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

from igv.data import build_library, read_pdb_chains  # noqa: E402
from igv.metrics import spearman  # noqa: E402
from igv.provenance import assert_provenance, write as prov_write  # noqa: E402

log = logging.getLogger("sanity")

_STRUCTURE_FOR = {"4fqi_h1": "4fqi_hlab", "4fqi_h3": "4fqi_hlab"}

THRESHOLDS = {
    "completeness": "relative error < 0.05",
    "m_sweep": "consecutive Spearman > 0.95 by m=32",
    "random_weights": "|Spearman vs trained| < 0.3",
    "dead_target": "max|attribution| < 1e-8",
    "signal_control": "score std > 1e-4 and Spearman != 0",
    "frozen_vs_full": "informational only, no pass/fail",
    "arm_assertion": "recorded arm matches intended arm",
}
CHECKS = list(THRESHOLDS)


def _result(name, passed, value, detail):
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "threshold": THRESHOLDS[name],
        "detail": detail,
    }


# 78, not 80: an 80GB-class card reports 81920 MiB to nvidia-smi but
# torch's total_memory returns the usable framebuffer after the ECC/reserve
# carve-out -- 79.2 GiB on A100-SXM4-80GB. A gate of 80 is unreachable on
# the exact hardware this project targets. 78 still rejects a 40GB A100.
def _require_vram(min_gib: int = 78) -> None:
    if os.environ.get("IGV_SKIP_VRAM_CHECK") == "1":
        log.warning("IGV_SKIP_VRAM_CHECK=1 -- skipping the %d GiB check", min_gib)
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. Set IGV_SKIP_VRAM_CHECK=1 to override.")
    gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gib < min_gib:
        raise RuntimeError(f"GPU 0 has {gib:.1f} GiB; need >= {min_gib} GiB.")


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_completeness(forward_fn, s_inputs, baseline, m_steps=16):
    """Do the attributions sum to f(x) - f(baseline)?"""
    import torch
    from igv.attrib import completeness_error, integrated_gradient

    res = integrated_gradient(forward_fn, s_inputs, baseline=baseline, m_steps=m_steps)
    with torch.no_grad():
        f_x = float(forward_fn(s_inputs))
        f_b = float(forward_fn(baseline))
    err = completeness_error(res, f_x, f_b)
    return _result(
        "completeness", err < 0.05, float(err),
        f"f(x)={f_x:.6f} f(baseline)={f_b:.6f} sum(ig)={float(res.ig.sum()):.6f}",
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


def check_random_weights(make_forward_fn, model, s_inputs, baseline, seed=0):
    """Re-init the model randomly; attribution should fall apart.

    If randomised weights reproduce the trained attribution, the method is
    reading input geometry rather than anything the model learned. This is the
    cheapest check that can kill the project outright.
    """
    import torch
    from igv.attrib import plain_gradient

    trained = plain_gradient(make_forward_fn(model), s_inputs, baseline=baseline)
    tm = trained.grad.squeeze(0).norm(dim=-1).detach().cpu().numpy()

    rnd = copy.deepcopy(model)
    torch.manual_seed(seed)
    n = 0
    for p in rnd.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)
        n += 1
    rm = plain_gradient(make_forward_fn(rnd), s_inputs, baseline=baseline)
    rm = rm.grad.squeeze(0).norm(dim=-1).detach().cpu().numpy()

    rho = float(spearman(tm, rm))
    return _result(
        "random_weights", abs(rho) < 0.3, rho,
        f"reinitialised {n} parameter tensors. "
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


def check_dead_target(s_inputs, baseline):
    """Attribute a constant objective; attribution must vanish.

    The objective is built as ``(x * 0).sum() + 1`` rather than a bare constant
    so the autograd graph still connects to the input -- otherwise .backward()
    raises instead of returning the zeros we want to observe.
    """
    from igv.attrib import integrated_gradient

    res = integrated_gradient(
        lambda x: (x * 0.0).sum() + 1.0, s_inputs, baseline=baseline, m_steps=8
    )
    mx = float(res.grad.abs().max())
    return _result("dead_target", mx < 1e-8, mx, f"max|grad| = {mx:.3e}")


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
    r = _result("frozen_vs_full", True, {"spearman": rho, "magnitude_ratio": ratio},
                f"spearman={rho:.4f} |frozen|/|full|={ratio:.4f}")
    r["informational"] = True
    return r


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
    p.add_argument("--chain", default="H", choices=["H", "L"])
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
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected = CHECKS if args.checks == "all" else [c.strip() for c in args.checks.split(",")]
    unknown = set(selected) - set(CHECKS)
    if unknown:
        raise SystemExit(f"Unknown checks: {sorted(unknown)}. Available: {CHECKS}")

    if args.dry_run:
        print(f"Would run {len(selected)} check(s) on {args.dataset}/{args.score}:\n")
        for c in selected:
            print(f"  {c:<18} {THRESHOLDS[c]}")
        print("\nGate: exits non-zero if any non-informational check fails.")
        return

    _require_vram()

    from igv.boltz_score import (
        SCORES, build_complex_feats, confidence_forward, embedder_only, load_model,
    )
    import torch

    if args.score not in SCORES:
        raise SystemExit(f"--score must be one of {sorted(SCORES)}")

    cache_dir = Path(args.cache_dir)
    lib = build_library(args.dataset, cache_dir, chain=args.chain)
    stem = args.structure or _STRUCTURE_FOR.get(args.dataset)
    if stem is None:
        raise SystemExit(f"No structure known for {args.dataset}; pass --structure.")
    pdb = cache_dir / f"{stem}.pdb"
    struct_chains = read_pdb_chains(pdb)
    model = load_model(args.checkpoint_dir, args.device)

    def chains_for(seq):
        d = dict(struct_chains)
        d[args.chain] = seq
        return d

    ref_feats, _ = build_complex_feats(
        chains_for(lib.reference_seq), pdb, cache_dir / "boltz_ref",
        args.device, use_msa_server=not args.no_msa_server,
    )
    x_pred = ref_feats["coords"].detach()
    s_inputs = embedder_only(model, ref_feats)
    baseline = torch.zeros_like(s_inputs)

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
            chains_for(seq), pdb, cache_dir / "boltz_sanity",
            args.device, use_msa_server=not args.no_msa_server,
        )
        with torch.no_grad():
            si = embedder_only(model, feats)
            return float(confidence_forward(
                model, si, feats, x_pred, args.score, gradient_checkpointing=False
            ))

    runners = {
        "completeness": lambda: check_completeness(forward_fn, s_inputs, baseline),
        "m_sweep": lambda: check_m_sweep(forward_fn, s_inputs, baseline),
        "random_weights": lambda: check_random_weights(
            make_forward_fn, model, s_inputs, baseline),
        "dead_target": lambda: check_dead_target(s_inputs, baseline),
        "signal_control": lambda: check_signal_control(
            score_of_sequence, lib, n_sample=args.n_sample),
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

    out = Path(args.out.format(dataset=args.dataset, score=args.score))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset, "score": args.score, "chain": args.chain,
        "checks": results,
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    prov_write(
        out, stage="07_sanity",
        inputs={"dataset": args.dataset, "structure": str(pdb)},
        params={"checks": selected, "n_sample": args.n_sample},
        arm={"score": args.score, "dataset": args.dataset, "chain": args.chain,
             "method": "sanity"},
        notes="Tier-0 gate. Non-informational failures make downstream numbers void.",
    )

    blocking = [r for r in results if not r["passed"] and not r.get("informational")]
    print("\n" + "=" * 68)
    for r in results:
        tag = "INFO" if r.get("informational") else ("PASS" if r["passed"] else "FAIL")
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
