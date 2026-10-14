#!/usr/bin/env python3
"""Stage 23 (DIAGNOSTIC) -- integrand profile for affinity and confidence heads.

Profiles the IG path integrand in res_type space for both Boltz-2's affinity
head (``affinity_pred_value``) and its confidence head (``complex_pde``) on
the same protein-ligand complex.  The direct test of section 22's
high-variation-integrand hypothesis: if f(alpha) has high total variation
relative to |f(1)-f(0)|, the path integral carries intrinsic noise that no
step count can fix.

Computes:
  - f(alpha) for both heads on a fine grid (forward-only, through the same
    ``affinity_forward`` / ``confidence_forward`` code path as the gradient)
  - <grad f(alpha), x-baseline> at a subset of alphas (backward)
  - Finite-difference comparison from the profile grid
  - Noise-floor estimate: repeat evaluations at 5 fixed alphas
  - Summary stats: total variation, TV/span ratio, FD/autograd agreement,
    sign changes (raw and noise-adjusted)

Output: CSV + JSON with ``.prov.json`` provenance.

COST
----
Profile: 257 alphas x 2 heads = 514 trunk forwards under no_grad.
Noise floor: 5 alphas x 3 repeats x 2 heads = 30 trunk forwards.
Gradcheck: 65 alphas x 2 heads x 1 backward = 130 backward-equivalent
trunk forwards.  FD is computed from the profile grid.

Total: ~674 forward-equivalents at L<=256.  At ~3-5 s per forward on A100:
~34-56 min.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util as _ilu
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.gpu import require_vram  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402

log = logging.getLogger("affinity_path_profile")

# Reuse tested pure helpers from 09_path_profile.
_09_path = Path(__file__).parent / "09_path_profile.py"
_09_spec = _ilu.spec_from_file_location("_path_profile_09", str(_09_path))
_pp09 = _ilu.module_from_spec(_09_spec)
_09_spec.loader.exec_module(_pp09)

alpha_grid = _pp09.alpha_grid
parse_grad_alphas = _pp09.parse_grad_alphas
fd_points = _pp09.fd_points
finite_difference = _pp09.finite_difference
ratio_and_relerr = _pp09.ratio_and_relerr
trapezoid_estimate = _pp09.trapezoid_estimate


# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable)
# ---------------------------------------------------------------------------


def total_variation(alphas: list[float], values: list[float]) -> float:
    """Sum of |f(a_{i+1}) - f(a_i)| over the sorted profile."""
    if len(alphas) != len(values) or len(alphas) < 2:
        return float("nan")
    pairs = sorted(zip(alphas, values))
    return sum(abs(pairs[i + 1][1] - pairs[i][1]) for i in range(len(pairs) - 1))


def count_sign_changes(values: list[float], threshold: float = 0.0) -> int:
    """Number of sign changes in a sequence, ignoring values at or below *threshold*."""
    changes = 0
    prev = None
    for v in values:
        if abs(v) <= threshold:
            continue
        if prev is not None and prev * v < 0:
            changes += 1
        prev = v
    return changes


def noise_adjusted_threshold(
    noise_sd: float | None,
    span: float,
    multiplier: float = 3.0,
) -> float:
    """Threshold for sign-change counting based on the measured noise floor.

    If *noise_sd* is available, returns ``multiplier * noise_sd * sqrt(2)``
    (the first-difference noise sd is sqrt(2) times the point noise sd for
    iid noise).  Otherwise falls back to ``max(|span| * 1e-4, 1e-8)``.
    """
    if noise_sd is not None and noise_sd > 0:
        return multiplier * noise_sd * math.sqrt(2)
    return max(abs(span) * 1e-4, 1e-8)


def expected_noise_tv(n_grid: int, noise_sd: float) -> float:
    """Expected total variation from iid noise of sd *noise_sd* on *n_grid* points.

    Each first difference of iid N(0, noise_sd) has sd = noise_sd * sqrt(2).
    E[|X|] for X ~ N(0, sigma) is sigma * sqrt(2/pi), so
    expected TV = (n_grid - 1) * noise_sd * sqrt(2) * sqrt(2/pi)
               = (n_grid - 1) * noise_sd * 2 / sqrt(pi).
    """
    if n_grid < 2 or noise_sd <= 0:
        return 0.0
    return (n_grid - 1) * noise_sd * 2.0 / math.sqrt(math.pi)


def nearest_profile_value(
    profile: dict[float, float], target: float,
) -> tuple[float, float]:
    """Return (alpha, f_alpha) from *profile* nearest to *target*."""
    best = min(profile, key=lambda a: abs(a - target))
    return best, profile[best]


BASELINE_CHOICES = ("mean_aa", "unk", "shuffled")

HEADS = ("affinity", "confidence")

_AFFINITY_SCORE = "affinity_pred_value"
_CONFIDENCE_SCORE = "complex_pde"

_NOISE_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
_NOISE_REPEATS = 3

_CSV_COLUMNS = [
    "head", "kind", "alpha", "F_alpha", "F_alpha_grad_pass",
    "D_analytic", "D_fd", "ratio", "relerr",
    "fd_kind", "fd_lo", "fd_hi", "F_fd_lo", "F_fd_hi",
    "fd_delta", "fd_snr_warning",
    "baseline",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--protein-chain-id", default="A")
    p.add_argument(
        "--protein-sequence", default=None,
        help="Protein sequence. Required unless --dry-run.",
    )
    p.add_argument("--ligand-id", default="B")
    p.add_argument("--ligand-smiles", default=None)
    p.add_argument("--ligand-ccd", default=None)
    p.add_argument(
        "--baseline", default="mean_aa", choices=BASELINE_CHOICES,
        help="IG baseline for the res_type interpolation path. Default: mean_aa.",
    )
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--pdb", default=None)
    p.add_argument("--resnum-offset", type=int, default=0)
    p.add_argument(
        "--x-pred", default="pdb", choices=["zeros", "predicted", "pdb"],
    )
    p.add_argument(
        "--profile-steps", type=int, default=257,
        help="Number of uniformly-spaced f(alpha) evaluations (0 disables). Default 257.",
    )
    p.add_argument(
        "--grad-alphas", default=None,
        help="Comma-separated alpha values for gradcheck. "
             "Default: --n-grad-alphas uniformly spaced.",
    )
    p.add_argument(
        "--n-grad-alphas", type=int, default=65,
        help="Number of uniformly-spaced gradcheck alphas (ignored if --grad-alphas given). "
             "Default 65.",
    )
    p.add_argument(
        "--fd-step", type=float, default=0.02,
        help="Finite-difference step h for the profile-grid FD. Default 0.02.",
    )
    p.add_argument("--skip-fd", action="store_true")
    p.add_argument(
        "--deterministic", action="store_true",
        help="Set torch.use_deterministic_algorithms(True) for repeatability. "
             "Falls back to warn_only if a deterministic kernel is unavailable.",
    )
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    p.add_argument("--no-msa-server", action="store_true")
    p.add_argument("--recycling-steps", type=int, default=1)
    p.add_argument(
        "--out", default="results/affinity_path_profile_{baseline}.csv",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.grad_alphas is not None:
        grad_alphas = parse_grad_alphas(args.grad_alphas)
    else:
        grad_alphas = alpha_grid(args.n_grad_alphas)

    out_csv = Path(args.out.format(baseline=args.baseline))
    out_json = out_csv.with_suffix(".json")

    n_profile = len(alpha_grid(args.profile_steps))
    n_gradcheck = len(grad_alphas)
    n_noise = len(_NOISE_ALPHAS) * _NOISE_REPEATS * len(HEADS)

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        print("Affinity path profile (diagnostic)\n")
        print(f"  baseline:         {args.baseline}")
        print(f"  profile steps:    {n_profile} (f(alpha) evaluations, forward-only)")
        print(f"  gradcheck:        {n_gradcheck} alpha(s)")
        print(f"  noise floor:      {len(_NOISE_ALPHAS)} alphas x {_NOISE_REPEATS} repeats"
              f" x {len(HEADS)} heads = {n_noise} forwards")
        print(f"  fd_step (h):      {args.fd_step}")
        print(f"  skip_fd:          {args.skip_fd}")
        print(f"  deterministic:    {args.deterministic}")
        print(f"  heads:            {', '.join(HEADS)}")
        print(f"  output CSV:       {out_csv}")

        fwd_profile = n_profile * len(HEADS)
        fwd_gc = n_gradcheck * len(HEADS) * 2
        fwd_eq = fwd_profile + fwd_gc + n_noise
        print(f"\nCost: {fwd_profile} forwards (profile)"
              f" + {n_noise} forwards (noise floor)"
              f" + {n_gradcheck}*{len(HEADS)}*2 backward-eq (gradcheck)"
              f" = {fwd_eq} forward-equivalents (FD from profile grid)")
        for label, s_per_fwd in [("fp32 L=256", 5.0), ("fp32 L=200", 3.0)]:
            wall = fwd_eq * s_per_fwd
            print(f"  est. wall ({label}): {wall / 60:.1f} min ({wall:.0f} s)")

        print("\nVM command (ABL1/imatinib, 2HYY, mean_aa baseline):")
        print("  python3 scripts/23_affinity_path_profile.py \\")
        print("    --protein-sequence <ABL1_KD_W235_Q498_264aa> \\")
        print("    --ligand-ccd STI --pdb data/raw/2hyy.pdb \\")
        print("    --resnum-offset 235 --baseline mean_aa")
        return 0

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if args.protein_sequence is None:
        raise SystemExit("--protein-sequence is required (not in --dry-run mode)")
    if args.ligand_smiles is None and args.ligand_ccd is None:
        raise SystemExit("One of --ligand-smiles / --ligand-ccd is required")
    if args.ligand_smiles is not None and args.ligand_ccd is not None:
        raise SystemExit("Only one of --ligand-smiles / --ligand-ccd may be given")
    if args.fd_step <= 0:
        raise SystemExit(f"--fd-step must be > 0, got {args.fd_step}")

    # ------------------------------------------------------------------
    # GPU path
    # ------------------------------------------------------------------
    require_vram()

    import torch

    from igv.boltz_score import (
        affinity_forward,
        build_affinity_feats,
        build_mean_aa_res_type_baseline,
        build_shuffled_res_type_baseline,
        build_unk_res_type_baseline,
        confidence_forward,
        enable_confidence_checkpointing,
        load_affinity_model,
        numerics_arm,
        predict_structure_coords,
    )

    # ---- Deterministic mode ----
    deterministic_active = False
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_active = True
            log.info("Deterministic algorithms enabled (strict)")
        except RuntimeError as e:
            log.warning("Strict deterministic failed, trying warn_only: %s", e)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                deterministic_active = True
                log.info("Deterministic algorithms enabled (warn_only)")
            except Exception:
                log.warning("Could not enable deterministic algorithms")

    cache_dir = Path(args.cache_dir)
    protein_chains = {args.protein_chain_id: args.protein_sequence}

    print(f"\n{'=' * 60}")
    print("Affinity path profile (diagnostic)")
    print(f"{'=' * 60}")
    print(f"  baseline:      {args.baseline}")
    print(f"  deterministic: {deterministic_active}")
    print(f"  protein:       {args.protein_chain_id} ({len(args.protein_sequence)} aa)")
    print(f"  ligand:        {args.ligand_id} "
          f"(smiles={args.ligand_smiles}, ccd={args.ligand_ccd})")

    # ---- Load model ----
    model, boltz_version = load_affinity_model(args.checkpoint_dir, args.device)
    enable_confidence_checkpointing(model)

    # ---- Featurise ----
    pdb_path = Path(args.pdb) if args.pdb else None
    cache_suffix = f"_pp_aff_{args.protein_chain_id}_{args.ligand_id}"
    feats, token_map = build_affinity_feats(
        protein_chains,
        args.ligand_id,
        cache_dir / f"boltz_affinity{cache_suffix}",
        args.device,
        ligand_smiles=args.ligand_smiles,
        ligand_ccd=args.ligand_ccd,
        use_msa_server=not args.no_msa_server,
        structure_pdb=pdb_path,
        resnum_offset=args.resnum_offset,
    )

    n_tokens = int(feats["token_pad_mask"][0].sum().item())
    print(f"  L (tokens after crop): {n_tokens}")

    x_pred_mode = args.x_pred
    if x_pred_mode == "pdb":
        if pdb_path is None:
            log.warning("--x-pred pdb but no --pdb; falling back to feats['coords']")
            x_pred_mode = "zeros"
        else:
            x_pred = feats["coords"].detach()
    if x_pred_mode == "predicted":
        x_pred = predict_structure_coords(
            model, feats, cache_dir,
            protein_chains,
            f"affinity_{args.protein_chain_id}_{args.ligand_id}",
            recycling_steps=args.recycling_steps,
        )
    elif x_pred_mode == "zeros":
        x_pred = feats["coords"].detach()

    # ---- Build baseline ----
    res_type_orig = feats["res_type"]
    if args.baseline == "mean_aa":
        baseline_rt = build_mean_aa_res_type_baseline(res_type_orig, feats)
    elif args.baseline == "unk":
        baseline_rt = build_unk_res_type_baseline(res_type_orig, feats)
    elif args.baseline == "shuffled":
        baseline_rt = build_shuffled_res_type_baseline(
            res_type_orig, feats, seed=args.shuffle_seed,
        )
    else:
        raise SystemExit(f"Unknown baseline: {args.baseline}")

    delta_rt = res_type_orig.detach().float() - baseline_rt.float()

    # ---- Forward helpers using the SAME code path as the gradient ----
    def _forward_head(alpha, head, grad_enabled=False):
        """Evaluate f(alpha) through affinity_forward / confidence_forward.

        Uses the same checkpointed trunk as the gradient pass so that
        the profile and the directional-derivative FD are on the same
        function.
        """
        rt_interp = (
            baseline_rt + alpha * delta_rt
        ).float()
        if grad_enabled:
            rt_interp = rt_interp.detach().requires_grad_(True)
        feats["res_type"] = rt_interp

        if grad_enabled:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

        s_inputs = model.input_embedder(feats, affinity=True)
        if head == "affinity":
            score = affinity_forward(
                model, s_inputs, feats, x_pred, _AFFINITY_SCORE,
                gradient_checkpointing=True,
                recycling_steps=args.recycling_steps,
            )
        else:
            score = confidence_forward(
                model, s_inputs, feats, x_pred, _CONFIDENCE_SCORE,
                gradient_checkpointing=True,
                recycling_steps=args.recycling_steps,
            )

        f_val = float(score)
        d_an = None
        if grad_enabled:
            score.backward()
            grad = rt_interp.grad
            if grad is not None:
                d_an = float((grad * delta_rt).sum())
            else:
                d_an = float("nan")
        return f_val, d_an, rt_interp

    # ---- CSV setup ----
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(out_csv, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=_CSV_COLUMNS)
    writer.writeheader()

    t0 = time.time()

    # ---- Noise floor: repeat evaluations at fixed alphas ----
    print(f"\n--- Noise floor: {len(_NOISE_ALPHAS)} alphas x {_NOISE_REPEATS} repeats ---")
    noise_data: dict[str, dict[float, list[float]]] = {
        h: {a: [] for a in _NOISE_ALPHAS} for h in HEADS
    }
    for alpha in _NOISE_ALPHAS:
        for rep in range(_NOISE_REPEATS):
            for head in HEADS:
                with torch.no_grad():
                    f_val, _, _ = _forward_head(alpha, head)
                noise_data[head][alpha].append(f_val)
            log.info(
                "[noise] alpha=%.2f rep=%d aff=%.6f conf=%.6f",
                alpha, rep,
                noise_data["affinity"][alpha][-1],
                noise_data["confidence"][alpha][-1],
            )

    noise_floor: dict[str, dict] = {}
    for head in HEADS:
        all_sds = []
        all_spreads = []
        for alpha in _NOISE_ALPHAS:
            vals = noise_data[head][alpha]
            if len(vals) >= 2:
                mean_v = sum(vals) / len(vals)
                sd_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
                spread = max(vals) - min(vals)
                all_sds.append(sd_v)
                all_spreads.append(spread)
        noise_sd = max(all_sds) if all_sds else 0.0
        noise_max_spread = max(all_spreads) if all_spreads else 0.0
        noise_floor[head] = {
            "noise_sd": noise_sd,
            "noise_max_spread": noise_max_spread,
            "per_alpha_sd": {a: all_sds[i] for i, a in enumerate(_NOISE_ALPHAS) if i < len(all_sds)},
        }
        print(f"  {head}: noise_sd={noise_sd:.2e}, max_spread={noise_max_spread:.2e}")

    # ---- Profile: f(alpha) on a grid ----
    profile_alphas = alpha_grid(args.profile_steps)
    profile_aff: dict[float, float] = {}
    profile_conf: dict[float, float] = {}

    print(f"\n--- Profile: {len(profile_alphas)} alphas ---")
    for i, alpha in enumerate(profile_alphas):
        for head in HEADS:
            with torch.no_grad():
                f_val, _, _ = _forward_head(alpha, head)
            if head == "affinity":
                profile_aff[alpha] = f_val
            else:
                profile_conf[alpha] = f_val
            writer.writerow({
                "head": head, "kind": "profile",
                "alpha": alpha, "F_alpha": f_val, "baseline": args.baseline,
            })

        csvfile.flush()
        if i % 32 == 0 or i == len(profile_alphas) - 1:
            log.info(
                "[profile %d/%d] alpha=%.4f aff=%.6f conf=%.6f",
                i + 1, len(profile_alphas), alpha,
                profile_aff[alpha], profile_conf[alpha],
            )

    # ---- Gradcheck: D_analytic vs D_fd for each head ----
    gc_results: dict[str, list[dict]] = {"affinity": [], "confidence": []}
    max_profile_grad_diff: dict[str, float] = {"affinity": 0.0, "confidence": 0.0}

    # Endpoint values for relative SNR threshold
    f0_aff = profile_aff.get(0.0, 0.0)
    f1_aff = profile_aff.get(1.0, 0.0)
    f0_conf = profile_conf.get(0.0, 0.0)
    f1_conf = profile_conf.get(1.0, 0.0)

    print(f"\n--- Gradcheck: {n_gradcheck} alphas, both heads ---")
    for i, alpha in enumerate(grad_alphas):
        for head in HEADS:
            profile = profile_aff if head == "affinity" else profile_conf

            f_grad, d_an, _ = _forward_head(alpha, head, grad_enabled=True)

            f_profile = profile.get(alpha)
            if f_profile is None:
                with torch.no_grad():
                    f_profile, _, _ = _forward_head(alpha, head)
                if head == "affinity":
                    profile_aff[alpha] = f_profile
                else:
                    profile_conf[alpha] = f_profile

            diff = abs(f_grad - f_profile)
            if diff > max_profile_grad_diff[head]:
                max_profile_grad_diff[head] = diff

            if head == "affinity":
                f0_ep, f1_ep = f0_aff, f1_aff
            else:
                f0_ep, f1_ep = f0_conf, f1_conf
            nsd = noise_floor[head]["noise_sd"]
            snr_scale = max(abs(f0_ep), abs(f1_ep), 1e-6)

            if args.skip_fd or not profile:
                row = {
                    "head": head, "kind": "gradcheck",
                    "alpha": alpha, "F_alpha": f_profile,
                    "F_alpha_grad_pass": f_grad,
                    "D_analytic": d_an,
                    "D_fd": None, "ratio": None, "relerr": None,
                    "fd_kind": "skipped",
                    "fd_lo": None, "fd_hi": None,
                    "F_fd_lo": None, "F_fd_hi": None,
                    "fd_delta": None, "fd_snr_warning": None,
                    "baseline": args.baseline,
                }
            else:
                lo_target, hi_target, fk = fd_points(alpha, args.fd_step)
                lo_alpha, f_lo = nearest_profile_value(profile, lo_target)
                hi_alpha, f_hi = nearest_profile_value(profile, hi_target)

                if abs(hi_alpha - lo_alpha) < 1e-12:
                    d_fd = float("nan")
                else:
                    d_fd = finite_difference(f_lo, f_hi, lo_alpha, hi_alpha)

                fd_delta = abs(f_hi - f_lo)
                snr_threshold = max(1e-3 * snr_scale, 3.0 * nsd) if nsd > 0 else 1e-3 * snr_scale
                snr_warning = fd_delta < snr_threshold
                r, re = ratio_and_relerr(d_an, d_fd)

                row = {
                    "head": head, "kind": "gradcheck",
                    "alpha": alpha, "F_alpha": f_profile,
                    "F_alpha_grad_pass": f_grad,
                    "D_analytic": d_an, "D_fd": d_fd,
                    "ratio": r, "relerr": re,
                    "fd_kind": fk,
                    "fd_lo": lo_alpha, "fd_hi": hi_alpha,
                    "F_fd_lo": f_lo, "F_fd_hi": f_hi,
                    "fd_delta": fd_delta, "fd_snr_warning": snr_warning,
                    "baseline": args.baseline,
                }

            gc_results[head].append(row)
            writer.writerow(row)
            csvfile.flush()

        log.info(
            "[gradcheck %d/%d] alpha=%.4f D_an(aff)=%.6f D_an(conf)=%.6f",
            i + 1, n_gradcheck, alpha,
            gc_results["affinity"][-1]["D_analytic"],
            gc_results["confidence"][-1]["D_analytic"],
        )

    csvfile.close()

    feats["res_type"] = res_type_orig
    wall = time.time() - t0

    # ---- Summary stats per head ----
    head_summaries = {}
    for head in HEADS:
        profile = profile_aff if head == "affinity" else profile_conf
        p_alphas = sorted(profile.keys())
        p_values = [profile[a] for a in p_alphas]

        f_0 = profile.get(0.0, float("nan"))
        f_1 = profile.get(1.0, float("nan"))
        span = f_1 - f_0

        tv_raw = total_variation(p_alphas, p_values)
        tv_span = tv_raw / abs(span) if abs(span) > 1e-12 else float("nan")

        nsd = noise_floor[head]["noise_sd"]
        expected_ntv = expected_noise_tv(len(p_alphas), nsd)
        tv_above_noise = tv_raw - expected_ntv

        first_diffs = [p_values[j + 1] - p_values[j] for j in range(len(p_values) - 1)]
        sc_threshold = noise_adjusted_threshold(nsd if nsd > 0 else None, span)
        n_sign_changes_f_raw = count_sign_changes(first_diffs, threshold=0.0)
        n_sign_changes_f_adj = count_sign_changes(first_diffs, threshold=sc_threshold)

        gc_rows = gc_results[head]
        d_an_vals = [r["D_analytic"] for r in gc_rows if math.isfinite(r["D_analytic"])]
        n_sign_changes_d_raw = count_sign_changes(d_an_vals, threshold=0.0)
        n_sign_changes_d_adj = count_sign_changes(d_an_vals, threshold=sc_threshold)

        ratios = [
            r["ratio"] for r in gc_rows
            if r["ratio"] is not None and math.isfinite(r["ratio"])
        ]
        relerrs = [
            r["relerr"] for r in gc_rows
            if r["relerr"] is not None and math.isfinite(r["relerr"])
        ]

        trap = trapezoid_estimate(
            [r["alpha"] for r in gc_rows],
            [r["D_analytic"] for r in gc_rows],
        )

        head_summaries[head] = {
            "F_baseline": f_0,
            "F_input": f_1,
            "span": span,
            "total_variation_raw": tv_raw,
            "total_variation_above_noise": tv_above_noise,
            "expected_noise_tv": expected_ntv,
            "tv_span_ratio": tv_span,
            "sign_change_threshold": sc_threshold,
            "n_sign_changes_f_first_diff_raw": n_sign_changes_f_raw,
            "n_sign_changes_f_first_diff_adjusted": n_sign_changes_f_adj,
            "n_sign_changes_D_analytic_raw": n_sign_changes_d_raw,
            "n_sign_changes_D_analytic_adjusted": n_sign_changes_d_adj,
            "trapezoid_D_analytic": trap,
            "completeness_check": (
                f"trap={trap:.6f} vs span={span:.6f}"
                if math.isfinite(trap) else "N/A"
            ),
            "n_gradcheck_points": len(gc_rows),
            "ratio_mean": (
                sum(ratios) / len(ratios) if ratios else float("nan")
            ),
            "ratio_median": (
                sorted(ratios)[len(ratios) // 2] if ratios else float("nan")
            ),
            "relerr_mean": (
                sum(relerrs) / len(relerrs) if relerrs else float("nan")
            ),
            "max_profile_vs_grad_diff": max_profile_grad_diff[head],
            "n_profile_points": len(p_alphas),
            "noise_floor": noise_floor[head],
        }

    # ---- Print summary ----
    for head in HEADS:
        hs = head_summaries[head]
        print(f"\n--- {head} head ---")
        print(f"  f(baseline) = {hs['F_baseline']:.6f}")
        print(f"  f(input)    = {hs['F_input']:.6f}")
        print(f"  span        = {hs['span']:.6f}")
        print(f"  TV (raw)            = {hs['total_variation_raw']:.6f}")
        print(f"  TV (above noise)    = {hs['total_variation_above_noise']:.6f}")
        print(f"  expected noise TV   = {hs['expected_noise_tv']:.6f}")
        print(f"  TV / |span|         = {hs['tv_span_ratio']:.2f}")
        print(f"  sign changes f' (raw/adj)  = {hs['n_sign_changes_f_first_diff_raw']}"
              f" / {hs['n_sign_changes_f_first_diff_adjusted']}")
        print(f"  sign changes D_an (raw/adj) = {hs['n_sign_changes_D_analytic_raw']}"
              f" / {hs['n_sign_changes_D_analytic_adjusted']}")
        print(f"  trap(D_an)          = {hs['trapezoid_D_analytic']:.6f}")
        print(f"  max |profile - grad_pass| = {hs['max_profile_vs_grad_diff']:.2e}")
        nf = hs["noise_floor"]
        print(f"  noise_sd={nf['noise_sd']:.2e}  max_spread={nf['noise_max_spread']:.2e}")
        if hs["ratio_mean"] is not None and math.isfinite(hs["ratio_mean"]):
            print(f"  mean D_an/D_fd      = {hs['ratio_mean']:.4f}")
            print(f"  median D_an/D_fd    = {hs['ratio_median']:.4f}")
            print(f"  mean |ratio - 1|    = {hs['relerr_mean']:.4f}")

    print(f"\n  wall time: {wall:.1f} s ({wall / 60:.1f} min)")

    # ---- Write JSON ----
    summary = {
        "baseline": args.baseline,
        "shuffle_seed": args.shuffle_seed if args.baseline == "shuffled" else None,
        "protein_chain_id": args.protein_chain_id,
        "ligand_id": args.ligand_id,
        "ligand_smiles": args.ligand_smiles,
        "ligand_ccd": args.ligand_ccd,
        "n_tokens": n_tokens,
        "x_pred_mode": x_pred_mode,
        "deterministic": deterministic_active,
        "profile_steps": args.profile_steps,
        "n_grad_alphas": n_gradcheck,
        "fd_step": args.fd_step,
        "skip_fd": args.skip_fd,
        "recycling_steps": args.recycling_steps,
        "heads": {head: head_summaries[head] for head in HEADS},
        "wall_s": wall,
        "boltz_version": boltz_version,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    prov_write(
        out_json,
        stage="23_affinity_path_profile",
        inputs={
            "protein_chain_id": args.protein_chain_id,
            "ligand_id": args.ligand_id,
            "ligand_smiles": args.ligand_smiles,
            "ligand_ccd": args.ligand_ccd,
            "structure_pdb": str(pdb_path) if pdb_path else None,
        },
        params={
            "baseline": args.baseline,
            "shuffle_seed": args.shuffle_seed,
            "profile_steps": args.profile_steps,
            "n_grad_alphas": n_gradcheck,
            "fd_step": args.fd_step,
            "skip_fd": args.skip_fd,
            "deterministic": deterministic_active,
            "recycling_steps": args.recycling_steps,
        },
        arm={
            "method": "affinity_path_profile",
            "baseline": args.baseline,
            "x_pred_mode": x_pred_mode,
            "deterministic": deterministic_active,
            "n_tokens": n_tokens,
            **numerics_arm(),
        },
    )

    prov_write(
        out_csv,
        stage="23_affinity_path_profile",
        inputs={
            "protein_chain_id": args.protein_chain_id,
            "ligand_id": args.ligand_id,
        },
        params={
            "baseline": args.baseline,
            "profile_steps": args.profile_steps,
            "n_grad_alphas": n_gradcheck,
            "deterministic": deterministic_active,
        },
        arm={
            "method": "affinity_path_profile",
            "baseline": args.baseline,
            "deterministic": deterministic_active,
            "n_tokens": n_tokens,
            **numerics_arm(),
        },
    )

    log.info("Wrote %s and %s (%.1f s)", out_csv, out_json, wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
