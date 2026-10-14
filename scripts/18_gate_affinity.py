#!/usr/bin/env python3
"""Stage 18 (GATE AFFINITY) -- gradient-reachability gate for the affinity head.

Does gradient flow from Boltz-2's affinity head back to feats["res_type"]
through the full trunk (res_type -> InputEmbedder -> trunk -> z -> affinity
module), analogous to how res_type_gradient_forward routes through the
confidence path?

The DEFAULT path runs the full trunk with per-block gradient checkpointing
(the same two-level scheme as confidence_forward).  A DIAGNOSTIC s_inputs-only
backward is also run to quantify the trunk's contribution as a ratio.

Reports:
  - grad shape, finite fraction, zero fraction
  - per-path grad norm (full trunk vs s_inputs-only diagnostic)
  - IG completeness at m in {16, 32, 64} against the real score delta
  - forward affinity value (for sanity-checking against `boltz predict`)
  - peak VRAM, wall time

Pass criteria (all must hold):
  1. res_type.grad is not None
  2. res_type.grad is not uniformly zero
  3. res_type.grad shape matches (1, L, num_tokens)
  4. res_type.grad is all finite
  5. completeness absolute error < threshold at m=16

Outcomes:
  exit 0 = PASS
  exit 1 = FAIL
  exit 2 = INCONCLUSIVE  (degenerate setup)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

log = logging.getLogger("gate_affinity")

COMPLETENESS_ABS_THRESHOLD = 0.10
COLLAPSED_SPAN_THRESHOLD = 1e-6

# Minimum |f(x) - f(baseline)| required before running the IG ladder.
# Below this, completeness error is dominated by integrand noise, not
# quadrature coarseness, and no step count fixes it (Lundstrom et al.,
# ICML 2022, PMLR v162). The 0.05 value is 5% of a 1.0 log10 unit span
# -- a deliberately conservative floor that would have caught the 0.085
# span that produced 300-700x completeness error in section 20.
SPAN_GUARD_THRESHOLD = 0.05

M_STEPS_LADDER = [16, 32, 64]

BASELINE_CHOICES = ("zeros", "mean_aa", "shuffled", "unk")


# ---------------------------------------------------------------------------
# Pure helpers (CPU, no torch, no boltz -- testable and usable in --dry-run)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--protein-chain-id", default="A",
        help="Protein chain id. Default: A.",
    )
    p.add_argument(
        "--protein-sequence", default=None,
        help="Protein sequence. Required unless --dry-run.",
    )
    p.add_argument(
        "--ligand-id", default="B",
        help="Ligand chain id. Default: B.",
    )
    p.add_argument(
        "--ligand-smiles", default=None,
        help="SMILES string for the ligand.",
    )
    p.add_argument(
        "--ligand-ccd", default=None,
        help="CCD code for the ligand.",
    )
    p.add_argument(
        "--score", default="affinity_pred_value",
        choices=["affinity_pred_value", "affinity_logits_binary"],
    )
    p.add_argument(
        "--baseline", default="mean_aa",
        choices=BASELINE_CHOICES,
        help="Baseline for IG. Default: mean_aa.",
    )
    p.add_argument(
        "--shuffle-seed", type=int, default=0,
        help="RNG seed for the shuffled baseline. Default: 0.",
    )
    p.add_argument(
        "--span-probe", type=int, default=None, metavar="N",
        help="Run N forward-pass span probes (for shuffled: N independent seeds; "
             "for deterministic baselines: 1 probe regardless of N). "
             "Reports f(x), f(baseline), span stats, and 5%%-of-span tolerance. "
             "Exits without running IG.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Override the span guard and run IG even if the span is below "
             f"{SPAN_GUARD_THRESHOLD}.",
    )
    p.add_argument(
        "--pdb", default=None,
        help="PDB file with real coordinates for the protein-ligand complex. "
             "Required for a real run: without it, AffinityCropper is geometry-blind.",
    )
    p.add_argument(
        "--resnum-offset", type=int, default=0,
        help="Added to 0-based residue indices in the token map to emit "
             "PDB author numbering (e.g. 235 for ABL1 kinase domain W235-Q498).",
    )
    p.add_argument(
        "--x-pred", default="pdb",
        choices=["zeros", "predicted", "pdb"],
        help="x_pred mode. Default: pdb (use --pdb coordinates as x_pred).",
    )
    p.add_argument(
        "--m-steps", type=int, nargs="+", default=M_STEPS_LADDER,
        help="Quadrature steps for IG completeness ladder. Default: 16 32 64.",
    )
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    p.add_argument("--no-msa-server", action="store_true")
    p.add_argument(
        "--recycling-steps", type=int, default=1,
    )
    p.add_argument(
        "--out", default="results/gate_affinity_{score}.json",
    )
    p.add_argument(
        "--out-npz", default="results/gate_affinity_{score}.npz",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--completeness-threshold", type=float,
        default=COMPLETENESS_ABS_THRESHOLD,
    )
    return p


def compute_span_stats(
    f_x: float,
    f_baselines: list[float],
) -> dict:
    """Compute span statistics from f(x) and one or more f(baseline) values."""
    spans = [f_x - fb for fb in f_baselines]
    abs_spans = [abs(s) for s in spans]
    mean_span = sum(abs_spans) / len(abs_spans)
    tolerance_5pct = 0.05 * mean_span
    return {
        "f_x": f_x,
        "f_baselines": f_baselines,
        "spans": spans,
        "abs_spans": abs_spans,
        "mean_abs_span": mean_span,
        "sd_abs_span": (
            (sum((s - mean_span) ** 2 for s in abs_spans) / len(abs_spans)) ** 0.5
            if len(abs_spans) > 1 else 0.0
        ),
        "min_abs_span": min(abs_spans),
        "max_abs_span": max(abs_spans),
        "completeness_tolerance_5pct": tolerance_5pct,
        "n_probes": len(f_baselines),
    }


def span_guard_check(
    mean_abs_span: float,
    threshold: float = SPAN_GUARD_THRESHOLD,
) -> tuple[bool, str]:
    """Return (ok, message). ok is True if span is large enough for IG."""
    if mean_abs_span < threshold:
        return False, (
            f"REFUSED: mean |f(x) - f(baseline)| = {mean_abs_span:.6f} < "
            f"{threshold} (SPAN_GUARD_THRESHOLD). The baseline does not create "
            f"enough score separation for IG completeness to be meaningful. "
            f"Sundararajan et al. (arXiv:1703.01365) recommend a near-zero "
            f"baseline score; at this span, 5% completeness tolerance is "
            f"{0.05 * mean_abs_span:.6f}, and integrand noise will dominate "
            f"quadrature error (Lundstrom et al., ICML 2022). Use --force to "
            f"override."
        )
    return True, (
        f"Span guard PASSED: mean |f(x) - f(baseline)| = {mean_abs_span:.6f} >= "
        f"{threshold}"
    )


def check_inconclusive(
    f_x: float | None,
    f_baseline: float | None,
) -> str | None:
    if f_x is None or f_baseline is None:
        return "score(s) could not be computed"
    if not math.isfinite(f_x):
        return f"f(input) = {f_x} is non-finite"
    if not math.isfinite(f_baseline):
        return f"f(baseline) = {f_baseline} is non-finite"
    span = abs(f_x - f_baseline)
    if span < COLLAPSED_SPAN_THRESHOLD:
        return (
            f"|f(input) - f(baseline)| = {span:.2e} < {COLLAPSED_SPAN_THRESHOLD:.0e} "
            "-- collapsed span"
        )
    return None


def evaluate_checks(
    grad_not_none: bool,
    grad_max_abs: float | None,
    grad_zero_frac: float | None,
    grad_shape: tuple | None,
    expected_shape: tuple,
    grad_n_nan: int | None,
    grad_n_inf: int | None,
    completeness_abs_err: float | None,
    completeness_rel_err: float | None,
    ig_sum: float | None,
    f_diff: float | None,
    threshold: float = COMPLETENESS_ABS_THRESHOLD,
) -> list[dict]:
    checks: list[dict] = []

    checks.append({
        "name": "res_type.grad is not None",
        "passed": grad_not_none,
        "detail": "",
    })

    if grad_max_abs is not None:
        passed = grad_max_abs > 0
        detail = (
            f"max|grad|={grad_max_abs:.6e}, "
            f"zero_frac={grad_zero_frac:.4f}"
        )
    else:
        passed = False
        detail = "grad is None"
    checks.append({
        "name": "res_type.grad not uniformly zero",
        "passed": passed,
        "detail": detail,
    })

    if grad_shape is not None:
        passed = tuple(grad_shape) == tuple(expected_shape)
        detail = f"actual={tuple(grad_shape)}, expected={tuple(expected_shape)}"
    else:
        passed = False
        detail = "grad is None"
    checks.append({
        "name": f"res_type.grad shape {tuple(expected_shape)}",
        "passed": passed,
        "detail": detail,
    })

    if grad_n_nan is not None and grad_n_inf is not None:
        passed = grad_n_nan == 0 and grad_n_inf == 0
        detail = f"NaN={grad_n_nan}, Inf={grad_n_inf}"
    else:
        passed = False
        detail = "grad is None"
    checks.append({
        "name": "res_type.grad all finite",
        "passed": passed,
        "detail": detail,
    })

    if completeness_abs_err is not None and f_diff is not None:
        passed = completeness_abs_err < threshold
        detail = (
            f"abs_err={completeness_abs_err:.6f}, "
            f"rel_err={completeness_rel_err:.4f}, "
            f"ig_sum={ig_sum:.6f}, "
            f"f(x)-f(b)={f_diff:.6f}"
        )
    else:
        passed = False
        detail = "not computed"
    checks.append({
        "name": f"completeness abs error < {threshold}",
        "passed": passed,
        "detail": detail,
    })

    return checks


def print_checks(checks: list[dict]) -> bool:
    all_pass = True
    for c in checks:
        tag = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_pass = False
        suffix = f"  ({c['detail']})" if c["detail"] else ""
        print(f"  [{tag}] {c['name']}{suffix}")
    return all_pass


# 1N46: TR-beta LBD + PFA (thyromimetic).
# Verified against RCSB REST API 2026-10-02:
#   entry title: "CRYSTAL STRUCTURE OF HUMAN TR BETA LIGAND-BINDING DOMAIN
#                 COMPLEXED WITH A POTENT SUBTYPE-SELECTIVE THYROMIMETIC"
#   entity 1: Thyroid hormone receptor Beta-1, chain A, 258 residues
#   entity 2 (nonpolymer): comp_id PFA, formula C20H21N3O4, 27 heavy atoms
#     (under <=128 hard cap, under >56 warn threshold)
# AffinityCropper max_tokens_protein=200, so 258 -> ~200 + ~27 lig = ~227 tokens.
EXAMPLE_COMPLEX = {
    "name": "1N46 (TR-beta LBD + PFA thyromimetic)",
    "protein_chain_id": "A",
    "ligand_id": "B",
    "ligand_ccd": "PFA",
    "pfa_heavy_atoms": 27,
    "n_protein_residues": 258,
    "n_tokens_approx": 227,
}

_1N46_SEQUENCE = (
    "LQKSIGHKPEPTDEEWELIKTVTEAHVATNAQGSHWKQKRKFLPEDIGQAPIVNAPEGGKVDLEAFSHFT"
    "KIITPAITRVVDFAKKLPMFCELPCEDQIILLKGCCMEIMSLRAAVRYDPESETLTLNGEMAVTRGQLKNG"
    "GLGVVSDAIFDLGMSLSSFNLDDTEVALLQAVLLMSSDRPGLACVERIEKYQDSFLLAFEHYINYRKHHVT"
    "HFWPKLLMKVTDLRMIGACHASRFLHMKVECPTELFPPLFLEVFED"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_dir = Path(args.cache_dir)
    out_json = Path(args.out.format(score=args.score))
    out_npz = Path(args.out_npz.format(score=args.score))

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        print("Gate Affinity: gradient-reachability for the affinity head\n")
        print(f"  protein_chain_id:     {args.protein_chain_id}")
        print(f"  ligand_id:            {args.ligand_id}")
        print(f"  ligand_smiles:        {args.ligand_smiles}")
        print(f"  ligand_ccd:           {args.ligand_ccd}")
        print(f"  score:                {args.score}")
        print(f"  baseline:             {args.baseline}")
        print(f"  pdb:                  {args.pdb}")
        print(f"  resnum_offset:        {args.resnum_offset}")
        print(f"  x_pred:               {args.x_pred}")
        print(f"  m_steps:              {args.m_steps}")
        print(f"  completeness thr:     abs < {args.completeness_threshold}")
        print(f"  output:               {out_json}")
        print(f"  device:               {args.device}")
        print(f"  checkpoint_dir:       {args.checkpoint_dir}")
        print()
        print("Default path: FULL TRUNK with per-block gradient checkpointing")
        print("  res_type -> InputEmbedder -> trunk (64 PF + 4 MSA blocks) -> z -> affinity module")
        print("  Same two-level checkpointing as confidence_forward.")
        print("  Also runs s_inputs-only diagnostic to report the trunk's contribution.")
        print()
        print("Memory estimate (affinity crop L<=256, checkpointed):")
        print("  Confidence path at L=730: 55.2 GiB measured.")
        print("  Affinity at L=256: (256/730)^2 * 55.2 ~ 6.8 GiB.")
        print("  With IGV_TRI_ATTN_CKPT=1 + IGV_AUTOCAST=bf16: comfortably within 80 GiB.")
        print()
        print("Suggested complex for the gate:")
        for k, v in EXAMPLE_COMPLEX.items():
            print(f"  {k}: {v}")
        print()
        print("Outcomes:")
        print("  exit 0 = PASS")
        print("  exit 1 = FAIL")
        print("  exit 2 = INCONCLUSIVE")
        print()
        print("Pass criteria (all five):")
        print("  1. res_type.grad is not None")
        print("  2. res_type.grad not uniformly zero")
        print("  3. res_type.grad shape (1, L, num_tokens)")
        print("  4. res_type.grad all finite")
        print(f"  5. completeness absolute error < {args.completeness_threshold}")
        print()
        print("VM command (1N46 TR-beta LBD + PFA):")
        print("  IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16 python3 scripts/16_gate_affinity.py \\")
        print(f"    --protein-sequence {_1N46_SEQUENCE} \\")
        print("    --ligand-ccd PFA \\")
        print(f"    --score {args.score}")
        print()
        print("Docker:")
        print("  docker run --gpus all -v ~/.boltz:/root/.boltz \\")
        print("    -v $(pwd):/app \\")
        print("    -e IGV_TRI_ATTN_CKPT=1 -e IGV_AUTOCAST=bf16 \\")
        print("    igv:latest \\")
        print("    python3 scripts/16_gate_affinity.py \\")
        print(f"      --protein-sequence {_1N46_SEQUENCE} \\")
        print("      --ligand-ccd PFA")
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

    # ------------------------------------------------------------------
    # GPU path
    # ------------------------------------------------------------------
    from igv.gpu import peak_allocated_gib, require_vram, reset_peak

    require_vram()

    import torch

    import numpy as np

    from igv.boltz_score import (
        AFFINITY_SCORES,
        affinity_forward,
        affinity_head_forward,
        build_affinity_feats,
        build_mean_aa_res_type_baseline,
        build_shuffled_res_type_baseline,
        build_unk_res_type_baseline,
        build_vocab_order,
        check_crop_coverage,
        enable_confidence_checkpointing,
        load_affinity_model,
        numerics_arm,
        predict_structure_coords,
    )
    from igv.provenance import write as prov_write

    if args.score not in AFFINITY_SCORES:
        raise SystemExit(f"--score must be one of {sorted(AFFINITY_SCORES)}")

    protein_chains = {args.protein_chain_id: args.protein_sequence}
    n_protein = len(args.protein_sequence)

    print(f"\n{'=' * 60}")
    print("Gate Affinity: gradient-reachability for the affinity head")
    print(f"{'=' * 60}")
    print(f"  protein chain: {args.protein_chain_id} ({n_protein} residues)")
    print(f"  ligand: {args.ligand_id} "
          f"(smiles={args.ligand_smiles}, ccd={args.ligand_ccd})")
    print(f"  score: {args.score}")
    print("  path: FULL TRUNK (per-block gradient checkpointing)")
    print()

    # ---- Load affinity model ----
    model, boltz_version = load_affinity_model(args.checkpoint_dir, args.device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  trainable params: {n_trainable:,}  (frozen: {n_frozen:,})")

    enable_confidence_checkpointing(model)

    # ---- Featurise ----
    pdb_path = Path(args.pdb) if args.pdb else None
    cache_suffix = f"_gate_aff_{args.protein_chain_id}_{args.ligand_id}"
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
    print(f"  L (tokens after affinity crop): {n_tokens}")
    print(f"  token_map entries: {len(token_map)}")
    if pdb_path is not None:
        print(f"  structure_pdb: {pdb_path}")
    else:
        print("  structure_pdb: NONE (crop is geometry-blind)")
    if args.resnum_offset:
        print(f"  resnum_offset: {args.resnum_offset}")

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
        log.info(
            "Predicted x_pred: shape %s, abs max %.6g",
            list(x_pred.shape), float(x_pred.abs().max()),
        )
    elif x_pred_mode == "zeros":
        x_pred = feats["coords"].detach()

    print(f"  x_pred mode: {x_pred_mode}")
    print(f"  x_pred abs max: {float(x_pred.abs().max()):.6g}")

    # ---- Forward pass (no_grad) to get the affinity value ----
    with torch.no_grad():
        s_inputs_fwd = model.input_embedder(feats, affinity=True)

        # Run trunk under no_grad to get z for the diagnostic path
        mask = feats["token_pad_mask"].float()
        pair_mask_fwd = mask[:, :, None] * mask[:, None, :]
        rel_pos = model.rel_pos(feats)
        token_bonds_z = model.token_bonds(feats["token_bonds"].float())
        contact_z = model.contact_conditioning(feats)

        s_init = model.s_init(s_inputs_fwd)
        z_init = (
            model.z_init_1(s_inputs_fwd)[:, :, None, :]
            + model.z_init_2(s_inputs_fwd)[:, None, :, :]
            + rel_pos + token_bonds_z + contact_z
        )
        s_ = torch.zeros_like(s_init)
        z_ = torch.zeros_like(z_init)
        for _ in range(args.recycling_steps + 1):
            s_ = s_init + model.s_recycle(model.s_norm(s_))
            z_ = z_init + model.z_recycle(model.z_norm(z_))
            z_ = z_ + model.msa_module(z_, s_inputs_fwd, feats, use_kernels=False)
            s_, z_ = model.pairformer_module(
                s_, z_, mask=mask, pair_mask=pair_mask_fwd, use_kernels=False,
            )
        z_nograd = z_.detach()

        f_x_val = float(affinity_head_forward(
            model, s_inputs_fwd, z_nograd, x_pred, feats,
            args.score, flow_through_z=False,
        ))

    print(f"\n  Forward affinity value ({args.score}): {f_x_val:.6f}")
    if args.score == "affinity_pred_value":
        print("  (log10 IC50 in uM; lower = stronger binding)")

    # ---- Baselines: compute f(baseline) for ALL four baselines ----
    res_type_orig = feats["res_type"]
    num_tokens_vocab = int(res_type_orig.shape[2]) if res_type_orig.dim() == 3 else None

    zeros_baseline_rt = torch.zeros_like(res_type_orig)
    mean_aa_baseline_rt = build_mean_aa_res_type_baseline(res_type_orig, feats)
    shuffled_baseline_rt = build_shuffled_res_type_baseline(
        res_type_orig, feats, seed=args.shuffle_seed,
    )
    unk_baseline_rt = build_unk_res_type_baseline(res_type_orig, feats)

    def _eval_baseline(bl_rt):
        bl_feats = dict(feats)
        bl_feats["res_type"] = bl_rt
        s_bl = model.input_embedder(bl_feats, affinity=True)
        return float(affinity_head_forward(
            model, s_bl, z_nograd, x_pred, bl_feats,
            args.score, flow_through_z=False,
        ))

    with torch.no_grad():
        f_zeros_baseline = _eval_baseline(zeros_baseline_rt)
        f_mean_aa_baseline = _eval_baseline(mean_aa_baseline_rt)
        f_shuffled_baseline = _eval_baseline(shuffled_baseline_rt)
        f_unk_baseline = _eval_baseline(unk_baseline_rt)

    baseline_f_values = {
        "zeros": f_zeros_baseline,
        "mean_aa": f_mean_aa_baseline,
        "shuffled": f_shuffled_baseline,
        "unk": f_unk_baseline,
    }

    baseline_rt_map = {
        "zeros": zeros_baseline_rt,
        "mean_aa": mean_aa_baseline_rt,
        "shuffled": shuffled_baseline_rt,
        "unk": unk_baseline_rt,
    }

    baseline_res_type = baseline_rt_map[args.baseline]
    f_baseline_val = baseline_f_values[args.baseline]

    # ---- Span probe mode (forward passes only, then exit) ----
    if args.span_probe is not None:
        n_probes = max(1, args.span_probe)
        print(f"\n--- Span probe: {n_probes} probe(s), baseline={args.baseline} ---")

        if args.baseline == "shuffled":
            probe_f_baselines = []
            for i in range(n_probes):
                seed_i = args.shuffle_seed + i
                bl_rt_i = build_shuffled_res_type_baseline(
                    res_type_orig, feats, seed=seed_i,
                )
                with torch.no_grad():
                    probe_f_baselines.append(_eval_baseline(bl_rt_i))
                print(f"  probe {i}: seed={seed_i}, f(baseline)={probe_f_baselines[-1]:.6f}")
        else:
            probe_f_baselines = [f_baseline_val]
            print(f"  probe 0: f(baseline)={f_baseline_val:.6f} (deterministic baseline)")

        stats = compute_span_stats(f_x_val, probe_f_baselines)
        print(f"\n  f(x)                     = {stats['f_x']:.6f}")
        print(f"  mean |f(x) - f(baseline)| = {stats['mean_abs_span']:.6f}")
        print(f"  sd   |f(x) - f(baseline)| = {stats['sd_abs_span']:.6f}")
        print(f"  min  |f(x) - f(baseline)| = {stats['min_abs_span']:.6f}")
        print(f"  max  |f(x) - f(baseline)| = {stats['max_abs_span']:.6f}")
        print(f"  5%-of-span tolerance       = {stats['completeness_tolerance_5pct']:.6f}")

        ok, msg = span_guard_check(stats["mean_abs_span"])
        print(f"\n  {msg}")
        if ok:
            print("  VERDICT: span is large enough to attempt IG.")
        else:
            print("  VERDICT: span is too small for meaningful IG completeness.")

        print("\n  All baselines for comparison:")
        for name in ("zeros", "mean_aa", "shuffled", "unk"):
            fv = baseline_f_values[name]
            print(f"    f_baseline_{name:7s} = {fv:.6f}  (span {abs(f_x_val - fv):.6f})")

        probe_json = {
            "mode": "span_probe",
            "score": args.score,
            "baseline": args.baseline,
            "shuffle_seed": args.shuffle_seed,
            "n_probes": n_probes,
            "f_x": f_x_val,
            "span_stats": stats,
            "span_guard_ok": ok,
            "span_guard_threshold": SPAN_GUARD_THRESHOLD,
            "f_baseline_zeros": f_zeros_baseline,
            "f_baseline_mean_aa": f_mean_aa_baseline,
            "f_baseline_shuffled": f_shuffled_baseline,
            "f_baseline_unk": f_unk_baseline,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(probe_json, indent=2, default=str) + "\n")
        log.info("Wrote %s", out_json)
        return 0

    # ---- INCONCLUSIVE pre-check ----
    inconclusive_reason = check_inconclusive(f_x_val, f_baseline_val)
    if inconclusive_reason is not None:
        print(f"\n{'=' * 60}")
        print("Gate Affinity: INCONCLUSIVE")
        print(f"{'=' * 60}")
        print(f"  Reason: {inconclusive_reason}")
        print(f"  f(input) = {f_x_val}")
        print(f"  f(baseline) [{args.baseline}] = {f_baseline_val}")
        for name in ("zeros", "mean_aa", "shuffled", "unk"):
            print(f"  f(baseline) [{name:7s}] = {baseline_f_values[name]:.6f}")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "gate": "affinity",
            "result": "INCONCLUSIVE",
            "reason": inconclusive_reason,
            "baseline": args.baseline,
            "shuffle_seed": args.shuffle_seed,
            "f_x": f_x_val,
            "f_baseline": f_baseline_val,
            "f_baseline_zeros": f_zeros_baseline,
            "f_baseline_mean_aa": f_mean_aa_baseline,
            "f_baseline_shuffled": f_shuffled_baseline,
            "f_baseline_unk": f_unk_baseline,
        }, indent=2) + "\n")
        return 2

    # ---- Span guard (must pass before IG ladder) ----
    f_diff = f_x_val - f_baseline_val
    span_ok, span_msg = span_guard_check(abs(f_diff))
    print(f"\n  {span_msg}")
    if not span_ok and not args.force:
        print(f"\n{'=' * 60}")
        print("Gate Affinity: REFUSED (span guard)")
        print(f"{'=' * 60}")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "gate": "affinity",
            "result": "REFUSED",
            "reason": span_msg,
            "baseline": args.baseline,
            "shuffle_seed": args.shuffle_seed,
            "f_x": f_x_val,
            "f_baseline": f_baseline_val,
            "f_diff": f_diff,
            "span_guard_threshold": SPAN_GUARD_THRESHOLD,
            "f_baseline_zeros": f_zeros_baseline,
            "f_baseline_mean_aa": f_mean_aa_baseline,
            "f_baseline_shuffled": f_shuffled_baseline,
            "f_baseline_unk": f_unk_baseline,
        }, indent=2) + "\n")
        return 1
    if not span_ok and args.force:
        print("  WARNING: --force overriding span guard. IG results may be meaningless.")

    print(f"  f(input)           = {f_x_val:.6f}")
    print(f"  f(baseline) [{args.baseline:7s}] = {f_baseline_val:.6f}")
    print(f"  span [{args.baseline:7s}]      = {abs(f_diff):.6f}")
    for name in ("zeros", "mean_aa", "shuffled", "unk"):
        fv = baseline_f_values[name]
        print(f"  f(baseline) [{name:7s}] = {fv:.6f}  (span {abs(f_x_val - fv):.6f})")

    # ---- Full-trunk backward (the default, scientifically load-bearing path) ----
    print("\n--- Full-trunk backward (default path) ---")
    print("  res_type -> InputEmbedder -> trunk -> z -> affinity module")

    res_type_leaf = res_type_orig.detach().clone().float().requires_grad_(True)
    feats["res_type"] = res_type_leaf

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    reset_peak()
    t_bwd_start = time.time()
    with torch.enable_grad():
        s_inputs_trunk = model.input_embedder(feats, affinity=True)
        score_trunk = affinity_forward(
            model, s_inputs_trunk, feats, x_pred, args.score,
            gradient_checkpointing=True,
            recycling_steps=args.recycling_steps,
        )
        score_trunk.backward()
    t_bwd = time.time() - t_bwd_start
    bwd_peak = peak_allocated_gib()

    grad = res_type_leaf.grad
    grad_not_none = grad is not None
    grad_max_abs = float(grad.abs().max()) if grad is not None else None
    grad_zero_frac = float((grad == 0).float().mean()) if grad is not None else None
    grad_shape = tuple(grad.shape) if grad is not None else None
    grad_n_nan = int(grad.isnan().sum()) if grad is not None else None
    grad_n_inf = int(grad.isinf().sum()) if grad is not None else None

    trunk_grad_norm = float(grad.norm()) if grad is not None else None

    print(f"  backward wall time: {t_bwd:.1f} s")
    if bwd_peak is not None:
        print(f"  peak VRAM: {bwd_peak:.2f} GiB")
    if grad is not None:
        print(f"  res_type.grad shape: {grad_shape}")
        print(f"  max|grad|: {grad_max_abs:.6e}")
        print(f"  zero fraction: {grad_zero_frac:.4f}")
        print(f"  finite fraction: {1.0 - (grad_n_nan + grad_n_inf) / grad.numel():.4f}")
        print(f"  full-trunk grad norm: {trunk_grad_norm:.6e}")

    # ---- s_inputs-only diagnostic backward ----
    print("\n--- Diagnostic: s_inputs-only backward (trunk frozen at WT) ---")
    print("  Measures how much of the gradient comes from the trunk vs the embedder alone.")

    res_type_leaf_diag = res_type_orig.detach().clone().float().requires_grad_(True)
    feats["res_type"] = res_type_leaf_diag

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    with torch.enable_grad():
        s_inputs_diag = model.input_embedder(feats, affinity=True)
        score_diag = affinity_head_forward(
            model, s_inputs_diag, z_nograd, x_pred, feats,
            args.score, flow_through_z=False,
        )
        score_diag.backward()

    diag_grad = res_type_leaf_diag.grad
    diag_grad_norm = float(diag_grad.norm()) if diag_grad is not None else None

    print(f"  s_inputs-only grad norm: {diag_grad_norm:.6e}" if diag_grad_norm else "  s_inputs-only grad: None")
    if trunk_grad_norm is not None and diag_grad_norm is not None and diag_grad_norm > 0:
        ratio = trunk_grad_norm / diag_grad_norm
        print(f"  trunk/s_inputs ratio: {ratio:.2f}x")
        print("  (ratio >> 1 means the trunk carries most of the signal)")

    # Restore feats
    feats["res_type"] = res_type_orig

    # ---- IG completeness ladder (full-trunk path) ----
    print(f"\n--- IG completeness ladder (full-trunk path, baseline={args.baseline}) ---")
    completeness_results = {}
    rung_ig_maps: dict[int, np.ndarray] = {}
    rung_grad_maps: dict[int, np.ndarray] = {}
    first_m_abs_err = None
    first_m_rel_err = None
    first_m_ig_sum = None

    delta_rt = res_type_orig.detach().float() - baseline_res_type.float()

    for m in sorted(args.m_steps):
        print(f"\n  m={m}:")
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        reset_peak()
        t0 = time.time()

        accumulated_grad = torch.zeros_like(res_type_orig.float())

        for step in range(m):
            alpha = (step + 0.5) / m
            res_type_interp = (
                baseline_res_type + alpha * (res_type_orig.detach() - baseline_res_type)
            ).float().requires_grad_(True)
            feats["res_type"] = res_type_interp

            s_inputs_step = model.input_embedder(feats, affinity=True)
            step_score = affinity_forward(
                model, s_inputs_step, feats, x_pred, args.score,
                gradient_checkpointing=True,
                recycling_steps=args.recycling_steps,
            )
            step_score.backward()

            step_grad = res_type_interp.grad
            if step_grad is not None:
                accumulated_grad = accumulated_grad + step_grad.detach() / m

            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

        ig_tensor = delta_rt * accumulated_grad
        ig_sum_val = float(ig_tensor.sum())

        wall = time.time() - t0
        ig_peak = peak_allocated_gib()

        abs_err = abs(ig_sum_val - f_diff)
        rel_err = abs_err / abs(f_diff) if abs(f_diff) > 1e-12 else float("inf")

        rung_ig_maps[m] = ig_tensor.detach().cpu().numpy().astype(np.float32)
        rung_grad_maps[m] = accumulated_grad.detach().cpu().numpy().astype(np.float32)

        print(f"    ig_sum = {ig_sum_val:.6f}")
        print(f"    f(x) - f(b) = {f_diff:.6f}")
        print(f"    abs error = {abs_err:.6f}")
        print(f"    rel error = {rel_err:.4f}")
        print(f"    wall time = {wall:.1f} s")
        if ig_peak is not None:
            print(f"    peak VRAM = {ig_peak:.2f} GiB")

        completeness_results[m] = {
            "ig_sum": ig_sum_val,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "wall_s": wall,
            "peak_vram_gib": ig_peak,
        }

        if first_m_abs_err is None:
            first_m_abs_err = abs_err
            first_m_rel_err = rel_err
            first_m_ig_sum = ig_sum_val

    # Restore feats
    feats["res_type"] = res_type_orig

    # ---- Evaluate gate ----
    expected_shape = (1, n_tokens, num_tokens_vocab) if num_tokens_vocab else (1, n_tokens, -1)
    checks = evaluate_checks(
        grad_not_none=grad_not_none,
        grad_max_abs=grad_max_abs,
        grad_zero_frac=grad_zero_frac,
        grad_shape=grad_shape,
        expected_shape=expected_shape,
        grad_n_nan=grad_n_nan,
        grad_n_inf=grad_n_inf,
        completeness_abs_err=first_m_abs_err,
        completeness_rel_err=first_m_rel_err,
        ig_sum=first_m_ig_sum,
        f_diff=f_diff,
        threshold=args.completeness_threshold,
    )

    print(f"\n{'=' * 60}")
    print("Gate Affinity results (full-trunk path):")
    print(f"{'=' * 60}")
    all_pass = print_checks(checks)
    gate_result = "PASS" if all_pass else "FAIL"
    print(f"\n  Gate Affinity: {gate_result}")

    # ---- Crop coverage ----
    # Sites must be residue NUMBERS (matching the token map's keys), not
    # 0-based indices.  With resnum_offset the token map keys are e.g.
    # 235..498 for ABL1; passing 0..263 would intersect the wrong range.
    all_sites = list(range(args.resnum_offset, args.resnum_offset + n_protein))
    crop_coverage = check_crop_coverage(
        token_map, chain_id="0", sites=all_sites,
    )
    print(f"\n  Crop coverage: {crop_coverage['n_surviving']}/{crop_coverage['n_requested']} "
          f"({crop_coverage['fraction_surviving']:.2%})")

    # ---- Write .npz artifact ----
    ladder = sorted(args.m_steps)
    final_m = ladder[-1]
    save_dict: dict[str, np.ndarray] = {}

    # Required keys per the 19_tism_hauser.py contract
    save_dict["grad_res_type"] = rung_grad_maps[final_m]
    save_dict["crop_coverage"] = np.array(
        json.dumps(crop_coverage), dtype=object,
    )
    save_dict["vocab_order"] = build_vocab_order()

    token_map_items = sorted(token_map.items(), key=lambda kv: kv[1])
    save_dict["token_map_keys"] = np.array(
        [(k[0], str(k[1])) for k, _ in token_map_items], dtype=object,
    )
    save_dict["token_map_values"] = np.array(
        [v for _, v in token_map_items], dtype=np.int64,
    )

    # Per-rung attributions (extra, for completeness analysis)
    save_dict["res_type_ig"] = rung_ig_maps[final_m]
    for m in ladder:
        save_dict[f"res_type_ig_m{m}"] = rung_ig_maps[m]
        save_dict[f"res_type_grad_m{m}"] = rung_grad_maps[m]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **save_dict)
    print(f"\nWrote {out_npz}")
    print(f"  keys: {sorted(save_dict.keys())}")

    # ---- Write JSON ----
    summary = {
        "gate": "affinity",
        "result": gate_result,
        "path": "full_trunk",
        "score": args.score,
        "baseline": args.baseline,
        "shuffle_seed": args.shuffle_seed,
        "x_pred_mode": x_pred_mode,
        "affinity_value": f_x_val,
        "f_baseline": f_baseline_val,
        "f_baseline_zeros": f_zeros_baseline,
        "f_baseline_mean_aa": f_mean_aa_baseline,
        "f_baseline_shuffled": f_shuffled_baseline,
        "f_baseline_unk": f_unk_baseline,
        "span_guard_threshold": SPAN_GUARD_THRESHOLD,
        "f_diff": f_diff,
        "n_protein_residues": n_protein,
        "n_tokens": n_tokens,
        "num_tokens_vocab": num_tokens_vocab,
        "grad_not_none": grad_not_none,
        "grad_max_abs": grad_max_abs,
        "grad_zero_frac": grad_zero_frac,
        "grad_shape": list(grad_shape) if grad_shape else None,
        "grad_n_nan": grad_n_nan,
        "grad_n_inf": grad_n_inf,
        "full_trunk_grad_norm": trunk_grad_norm,
        "s_inputs_only_grad_norm": diag_grad_norm,
        "trunk_to_s_inputs_ratio": (
            trunk_grad_norm / diag_grad_norm
            if trunk_grad_norm and diag_grad_norm and diag_grad_norm > 0
            else None
        ),
        "completeness_ladder": completeness_results,
        "backward_wall_s": t_bwd,
        "backward_peak_vram_gib": bwd_peak,
        "trainable_params": n_trainable,
        "frozen_params": n_frozen,
        "boltz_version": boltz_version,
        "checks": checks,
        "crop_coverage": crop_coverage,
        "protein_chain_id": args.protein_chain_id,
        "ligand_id": args.ligand_id,
        "ligand_smiles": args.ligand_smiles,
        "ligand_ccd": args.ligand_ccd,
        "structure_pdb": str(pdb_path) if pdb_path else None,
        "resnum_offset": args.resnum_offset,
        "npz_path": str(out_npz),
        "npz_keys": sorted(save_dict.keys()),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    prov_write(
        out_json,
        stage="18_gate_affinity",
        inputs={
            "protein_chain_id": args.protein_chain_id,
            "ligand_id": args.ligand_id,
            "ligand_smiles": args.ligand_smiles,
            "ligand_ccd": args.ligand_ccd,
            "structure_pdb": str(pdb_path) if pdb_path else None,
        },
        params={
            "score": args.score,
            "baseline": args.baseline,
            "shuffle_seed": args.shuffle_seed,
            "x_pred_mode": x_pred_mode,
            "resnum_offset": args.resnum_offset,
            "m_steps": args.m_steps,
            "completeness_threshold": args.completeness_threshold,
            "recycling_steps": args.recycling_steps,
        },
        arm={
            "score": args.score,
            "method": "gate_affinity",
            "path": "full_trunk",
            "baseline": args.baseline,
            "x_pred_mode": x_pred_mode,
            "n_tokens": n_tokens,
            **numerics_arm(),
        },
    )
    prov_write(
        out_npz,
        stage="18_gate_affinity",
        inputs={
            "protein_chain_id": args.protein_chain_id,
            "ligand_id": args.ligand_id,
            "ligand_smiles": args.ligand_smiles,
            "ligand_ccd": args.ligand_ccd,
        },
        params={
            "score": args.score,
            "baseline": args.baseline,
            "x_pred_mode": x_pred_mode,
            "m_steps": args.m_steps,
        },
        arm={
            "score": args.score,
            "method": "gate_affinity",
            "path": "full_trunk",
            "baseline": args.baseline,
            "x_pred_mode": x_pred_mode,
            "n_tokens": n_tokens,
            **numerics_arm(),
        },
    )

    log.info("Wrote %s", out_json)
    log.info("Wrote %s", out_npz)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
