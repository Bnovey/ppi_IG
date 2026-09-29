#!/usr/bin/env python3
"""Stage 09 (DIAGNOSTIC) -- gradient fidelity along the IG path.

Not part of the pipeline. Deliberately absent from ``make all``: it is a
diagnostic for the completeness failure recorded in docs/MEMSCALE_RESULTS.md
section 6a, where the completeness check overshot by 4.64x at L=730 under
IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16.

WHY THIS EXISTS
---------------
The completeness identity says integral_0^1 D_analytic(alpha) dalpha == F(1)-F(0),
where D_analytic is the directional derivative along the IG path, computed by
autograd. A 4.64x systematic overshoot is too large for quadrature error and is
the signature of a wrong gradient -- e.g. a chain-rule defect in the reentrant-
checkpoint / autocast nesting documented in src/igv/boltz_score.py.

This script compares, at selected alphas, the autograd directional derivative
D_analytic(alpha) = sum(grad(F, interp) * d) against a central finite difference
D_fd(alpha) = (F(alpha+h) - F(alpha-h)) / (2h). If the analytic/FD ratio
clusters near 4.6, the gradient is wrong and the completeness failure is
explained. If the ratio is ~1.0, the gradient is right and the failure is
quadrature or the baseline.

COST
----
Each profile point is one forward (under no_grad).
Each gradcheck alpha is one backward plus two forwards.
At L=554, bf16 is ~55 s per forward+backward and fp32 ~98 s (MEMSCALE_RESULTS
section 6). So with defaults (21 profile + 3 gradcheck): ~21 forwards + 3*(1
backward + 2 forwards) = 30 effective forward-equivalents.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import build_library, read_pdb_chains, resolve_chain_subset  # noqa: E402
from igv.gpu import require_vram  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402

log = logging.getLogger("path_profile")

_STRUCTURE_FOR = {"4fqi_h1": "4fqi_hlab", "4fqi_h3": "4fqi_hlab"}


# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable)
# ---------------------------------------------------------------------------


def alpha_grid(n: int) -> list[float]:
    """Return *n* uniformly-spaced alphas from 0 to 1 inclusive. n=0 disables."""
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


def parse_grad_alphas(spec: str) -> list[float]:
    """Parse a comma-separated list of floats in [0, 1]."""
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    for v in values:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"alpha {v} outside [0, 1]")
    return values


def fd_kind_for(alpha: float, h: float) -> str:
    """Determine whether central, forward, or backward FD is needed."""
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")
    if alpha - h >= 0.0 and alpha + h <= 1.0:
        return "central"
    if alpha + h <= 1.0:
        return "forward"
    if alpha - h >= 0.0:
        return "backward"
    raise ValueError(
        f"alpha={alpha} with h={h} cannot fit any finite difference in [0, 1]"
    )


def fd_points(alpha: float, h: float) -> tuple[float, float, str]:
    """Return (lo, hi, kind) for the finite difference."""
    kind = fd_kind_for(alpha, h)
    if kind == "central":
        return alpha - h, alpha + h, kind
    if kind == "forward":
        return alpha, alpha + 2 * h, kind
    return alpha - 2 * h, alpha, kind


def finite_difference(f_lo: float, f_hi: float, lo: float, hi: float) -> float:
    return (f_hi - f_lo) / (hi - lo)


def ratio_and_relerr(analytic: float, fd: float):
    """Return (ratio, relative_error). Guard against fd == 0."""
    if abs(fd) < 1e-30:
        return float("nan"), float("nan")
    ratio = analytic / fd
    relerr = abs(ratio - 1.0)
    return ratio, relerr


def trapezoid_estimate(alphas: list[float], values: list[float]) -> float:
    """Trapezoidal rule over sorted (alpha, value) pairs."""
    if len(alphas) != len(values) or len(alphas) < 2:
        return float("nan")
    pairs = sorted(zip(alphas, values))
    total = 0.0
    for i in range(len(pairs) - 1):
        a0, v0 = pairs[i]
        a1, v1 = pairs[i + 1]
        total += 0.5 * (v0 + v1) * (a1 - a0)
    return total


_PROFILE_COLUMNS = ["kind", "alpha", "F_alpha"]
_GRADCHECK_COLUMNS = [
    "kind", "alpha", "F_alpha", "D_analytic", "D_fd",
    "ratio", "relerr", "fd_kind", "fd_lo", "fd_hi",
    "F_fd_lo", "F_fd_hi", "fd_delta", "fd_snr_warning",
]
_META_COLUMNS = ["baseline", "F_baseline", "F_input"]


def estimate_wall_time(
    n_profile: int,
    n_gradcheck: int,
    seconds_per_forward: float,
    seconds_per_backward: float,
    skip_fd: bool = False,
) -> float:
    """Estimated wall clock in seconds."""
    fwd = n_profile + (0 if skip_fd else n_gradcheck * 2)
    bwd = n_gradcheck
    return fwd * seconds_per_forward + bwd * seconds_per_backward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _chain_id(value: str) -> str:
    """Argparse type: a single alphanumeric PDB chain identifier."""
    if len(value) != 1 or not value.isalnum():
        raise argparse.ArgumentTypeError(
            f"chain ID must be a single alphanumeric character, got {value!r}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--chain", default="H", type=_chain_id)
    p.add_argument("--score", default="complex_pde")
    p.add_argument("--structure", default=None)
    p.add_argument("--cache-dir", default="data/raw")
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
        "--profile-steps", type=int, default=21,
        help="Number of uniformly-spaced F(alpha) evaluations (0 disables). Default 21.",
    )
    p.add_argument(
        "--grad-alphas", default="0.1,0.5,0.9",
        help="Comma-separated alpha values for the analytic-vs-FD comparison.",
    )
    p.add_argument(
        "--fd-step", type=float, default=0.1,
        # docs/MEMSCALE_RESULTS.md section 6a records run-to-run noise on the score
        # of ~1.6% (two identical bf16 runs gave f(x) 3.916786 and 3.855304),
        # i.e. ~0.06 absolute. F traverses roughly 12.4 to 3.9 across the path,
        # so h=0.1 gives an FD window of order 0.8 and a signal-to-noise ratio
        # near 13, while h=0.01 would be swamped by nondeterminism.
        help="Finite-difference step size h (default 0.1).",
    )
    p.add_argument("--skip-fd", action="store_true",
                   help="Skip finite-difference comparison; record D_analytic only.")
    p.add_argument(
        "--out", default="results/path_profile_{dataset}_{score}.csv",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--baseline", default="zeros", choices=["zeros", "mean_aa"],
        help="IG baseline: zeros (all-zeros embedding) or mean_aa (mean over "
             "20 canonical homopolymer embeddings). Default: zeros.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.fd_step <= 0:
        raise SystemExit(f"--fd-step must be > 0, got {args.fd_step}")

    grad_alphas = parse_grad_alphas(args.grad_alphas)

    cache_dir = Path(args.cache_dir)
    stem = args.structure or _STRUCTURE_FOR.get(args.dataset)
    if stem is None:
        raise SystemExit(f"No structure known for {args.dataset}; pass --structure.")
    pdb = cache_dir / f"{stem}.pdb"

    subset_label = "all"
    n_tokens = None
    if pdb.exists():
        all_chains = read_pdb_chains(pdb)
        struct_chains, subset_label = resolve_chain_subset(
            all_chains, args.chain_subset, args.chain,
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

    out_csv = Path(args.out.format(dataset=args.dataset, score=args.score))

    n_profile = len(alpha_grid(args.profile_steps))
    n_gradcheck = len(grad_alphas)

    if args.dry_run:
        print(f"path-profile diagnostic for {args.dataset}/{args.score}\n")
        print(f"  profile steps:  {n_profile} (F(alpha) evaluations, forward-only)")
        print(f"  gradcheck:      {n_gradcheck} alpha(s) at {grad_alphas}")
        print(f"  fd_step (h):    {args.fd_step}")
        print(f"  skip_fd:        {args.skip_fd}")
        print(f"  baseline:       {args.baseline}")
        if n_tokens is not None:
            print(f"  chain subset:   {subset_label}  L={n_tokens}")
        for label, fwd_s, bwd_s in [("bf16", 27.5, 27.5), ("fp32", 49.0, 49.0)]:
            wall = estimate_wall_time(n_profile, n_gradcheck, fwd_s, bwd_s, skip_fd=args.skip_fd)
            print(f"  est. wall ({label}): {wall / 60:.1f} min ({wall:.0f} s)")
        print(f"\n  output CSV:     {out_csv}")
        if args.skip_fd:
            fwd_eq = n_profile + n_gradcheck
            print(f"\nCost: {n_profile} forwards (profile) + "
                  f"{n_gradcheck}*(1 bwd) (gradcheck, no FD) = "
                  f"{fwd_eq} forward-equivalents")
        else:
            print(f"\nCost: {n_profile} forwards (profile) + "
                  f"{n_gradcheck}*(2 fwd + 1 bwd) (gradcheck) = "
                  f"{n_profile + n_gradcheck * 3} forward-equivalents")
        return 0

    require_vram()

    import torch

    from igv.boltz_score import (
        SCORES,
        build_complex_feats,
        compute_homopolymer_embeddings,
        confidence_forward,
        embedder_only,
        load_model,
        numerics_arm,
    )

    if args.score not in SCORES:
        raise SystemExit(f"--score must be one of {sorted(SCORES)}")

    lib = build_library(args.dataset, cache_dir, chain=args.chain)
    if not pdb.exists():
        raise SystemExit(f"Missing PDB: {pdb}")
    if n_tokens is None:
        all_chains = read_pdb_chains(pdb)
        struct_chains, subset_label = resolve_chain_subset(
            all_chains, args.chain_subset, args.chain,
        )
        n_tokens = sum(len(s) for s in struct_chains.values())

    cache_suffix = f"_{subset_label}" if subset_label != "all" else ""
    model, _boltz_version = load_model(args.checkpoint_dir, args.device)

    def chains_for(seq):
        d = dict(struct_chains)
        d[args.chain] = seq
        return d

    ref_chains = chains_for(lib.reference_seq)
    n_tokens_pdb, n_tokens = n_tokens, sum(len(s) for s in ref_chains.values())
    if n_tokens != n_tokens_pdb:
        log.warning(
            "L=%d as featurised, not the %d the PDB implies: chain %s is "
            "reference_seq (%d aa) and not the PDB's (%d aa).",
            n_tokens, n_tokens_pdb, args.chain,
            len(lib.reference_seq), len(struct_chains[args.chain]),
        )
    log.info("Featurising L=%d over chains %s", n_tokens, list(ref_chains))

    ref_feats, token_map = build_complex_feats(
        ref_chains, pdb,
        cache_dir / f"boltz_ref{cache_suffix}",
        args.device, use_msa_server=not args.no_msa_server,
    )
    x_pred = ref_feats["coords"].detach()
    s_inputs = embedder_only(model, ref_feats)

    if args.baseline == "mean_aa":
        import numpy as np

        from igv.attrib import build_mean_aa_baseline

        log.info("Computing mean-AA baseline (20 homopolymer embeddings)")
        token_indices_arr = np.array(
            [token_map[(args.chain, i)] for i in range(len(lib.reference_seq))],
            dtype=np.int64,
        )
        per_aa_embs = compute_homopolymer_embeddings(
            model, ref_chains, args.chain, pdb, cache_dir,
            args.device, use_msa_server=not args.no_msa_server,
        )
        baseline = build_mean_aa_baseline(s_inputs, token_indices_arr, per_aa_embs)
        log.info("mean_aa baseline built, shape %s", list(baseline.shape))
    else:
        baseline = torch.zeros_like(s_inputs)

    x = s_inputs
    b = baseline
    d = x - b

    def forward_scalar(s):
        return confidence_forward(
            model, s, ref_feats, x_pred, args.score,
            gradient_checkpointing=True,
        )

    def F_at(alpha):
        interp = b + alpha * d
        with torch.no_grad():
            return float(forward_scalar(interp))

    def D_analytic_at(alpha):
        # .backward() and interp.grad, NOT torch.autograd.grad: the trunk runs
        # under a REENTRANT checkpoint (ERRORS_LOG entry 9 records that the
        # non-reentrant outer checkpoint OOMs, so reentrant is load-bearing),
        # and torch raises outright on the combination --
        #   "When use_reentrant=True, torch.utils.checkpoint is incompatible
        #    with .grad() or passing an `inputs` parameter to .backward()".
        # attrib.integrated_gradient already uses backward() for this reason,
        # and matching it keeps this diagnostic on the same code path as the
        # thing it is diagnosing.
        interp = (b + alpha * d).detach().requires_grad_(True)
        val = forward_scalar(interp)
        val.backward()
        return float((interp.grad * d).sum())

    # --- F(baseline) and F(input) ---
    f_baseline = F_at(0.0)
    f_input = F_at(1.0)
    log.info(
        "F(baseline) [%s] = %.6f, F(input) = %.6f",
        args.baseline, f_baseline, f_input,
    )
    _row_meta = {
        "baseline": args.baseline,
        "F_baseline": f_baseline,
        "F_input": f_input,
    }

    # --- CSV setup ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(out_csv, "w", newline="")
    all_columns = list(dict.fromkeys(
        _PROFILE_COLUMNS + _GRADCHECK_COLUMNS + _META_COLUMNS
    ))
    writer = csv.DictWriter(csvfile, fieldnames=all_columns)
    writer.writeheader()

    t0 = time.time()

    # --- Profile: F(alpha) on a grid ---
    profile_alphas = alpha_grid(args.profile_steps)
    profile_values = {0.0: f_baseline, 1.0: f_input}
    for i, alpha in enumerate(profile_alphas):
        if alpha in profile_values:
            f_val = profile_values[alpha]
        else:
            f_val = F_at(alpha)
            profile_values[alpha] = f_val
        row = {"kind": "profile", "alpha": alpha, "F_alpha": f_val, **_row_meta}
        writer.writerow(row)
        csvfile.flush()
        log.info("[profile %d/%d] alpha=%.4f F=%.6f", i + 1, len(profile_alphas), alpha, f_val)

    # --- Gradcheck: D_analytic vs D_fd ---
    gc_rows = []
    for i, alpha in enumerate(grad_alphas):
        d_an = D_analytic_at(alpha)
        f_alpha = F_at(alpha) if alpha not in profile_values else profile_values[alpha]

        if args.skip_fd:
            row = {
                "kind": "gradcheck",
                "alpha": alpha,
                "F_alpha": f_alpha,
                "D_analytic": d_an,
                "D_fd": None,
                "ratio": None,
                "relerr": None,
                "fd_kind": "skipped",
                "fd_lo": None,
                "fd_hi": None,
                "F_fd_lo": None,
                "F_fd_hi": None,
                "fd_delta": None,
                "fd_snr_warning": None,
                **_row_meta,
            }
            gc_rows.append(row)
            writer.writerow(row)
            csvfile.flush()
            log.info(
                "[gradcheck %d/%d] alpha=%.4f D_analytic=%.6f (FD skipped)",
                i + 1, len(grad_alphas), alpha, d_an,
            )
        else:
            lo, hi, fk = fd_points(alpha, args.fd_step)
            f_lo = F_at(lo)
            f_hi = F_at(hi)
            d_fd = finite_difference(f_lo, f_hi, lo, hi)

            fd_delta = abs(f_hi - f_lo)
            snr_warning = fd_delta < 0.5

            r, re = ratio_and_relerr(d_an, d_fd)

            if snr_warning:
                log.warning(
                    "alpha=%.4f: |F(hi)-F(lo)| = %.4f < 0.5 -- ratio is noise, "
                    "do not interpret as a finding.",
                    alpha, fd_delta,
                )

            row = {
                "kind": "gradcheck",
                "alpha": alpha,
                "F_alpha": f_alpha,
                "D_analytic": d_an,
                "D_fd": d_fd,
                "ratio": r,
                "relerr": re,
                "fd_kind": fk,
                "fd_lo": lo,
                "fd_hi": hi,
                "F_fd_lo": f_lo,
                "F_fd_hi": f_hi,
                "fd_delta": fd_delta,
                "fd_snr_warning": snr_warning,
                **_row_meta,
            }
            gc_rows.append(row)
            writer.writerow(row)
            csvfile.flush()
            log.info(
                "[gradcheck %d/%d] alpha=%.4f D_analytic=%.6f D_fd=%.6f ratio=%.4f "
                "fd_kind=%s snr_warn=%s",
                i + 1, len(grad_alphas), alpha, d_an, d_fd, r, fk, snr_warning,
            )

    csvfile.close()
    wall = time.time() - t0

    # --- JSON summary ---
    f_0 = profile_values.get(0.0, f_baseline)
    f_1 = profile_values.get(1.0, f_input)

    ratios = [r["ratio"] for r in gc_rows if math.isfinite(r["ratio"])]
    trap = trapezoid_estimate(
        [r["alpha"] for r in gc_rows],
        [r["D_analytic"] for r in gc_rows],
    )

    summary = {
        "baseline": args.baseline,
        "F_baseline": f_baseline,
        "F_input": f_input,
        "F_0": f_0,
        "F_1": f_1,
        "F_1_minus_F_0": f_1 - f_0,
        "trapezoid_D_analytic": trap,
        "trapezoid_note": (
            f"coarse-node estimate from {len(gc_rows)} gradcheck alpha(s); "
            "indicative with defaults, meaningful with a denser grid"
        ),
        "ratio_mean": float(sum(ratios) / len(ratios)) if ratios else float("nan"),
        "ratio_spread": (
            float(max(ratios) - min(ratios)) if len(ratios) >= 2 else 0.0
        ) if ratios else float("nan"),
        "reference_overshoot": 4.64,
        "n_profile_points": len(profile_alphas),
        "n_gradcheck_points": len(gc_rows),
        "wall_s": wall,
    }

    out_json = out_csv.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    prov_write(
        out_json,
        stage="09_path_profile",
        inputs={"dataset": args.dataset, "structure": str(pdb)},
        params={
            "profile_steps": args.profile_steps,
            "grad_alphas": grad_alphas,
            "fd_step": args.fd_step,
            "baseline": args.baseline,
            "chain_subset": list(struct_chains) if subset_label != "all" else None,
        },
        arm={
            "score": args.score,
            "dataset": args.dataset,
            "chain": args.chain,
            "baseline": args.baseline,
            "method": "path_profile",
            "chain_subset": list(struct_chains) if subset_label != "all" else None,
            "n_tokens": n_tokens,
            "fd_step": args.fd_step,
            "grad_alphas": grad_alphas,
            **numerics_arm(),
        },
        notes=(
            "Diagnostic: analytic-vs-FD gradient comparison along the IG path. "
            f"If ratio ~ {summary.get('reference_overshoot', 4.64)}, the gradient "
            "is wrong. If ratio ~ 1.0, the gradient is right."
        ),
    )

    log.info("Wrote %s and %s (%.1f s)", out_csv, out_json, wall)
    log.info("Summary: %s", json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
