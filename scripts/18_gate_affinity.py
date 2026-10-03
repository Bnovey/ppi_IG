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

M_STEPS_LADDER = [16, 32, 64]


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
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--completeness-threshold", type=float,
        default=COMPLETENESS_ABS_THRESHOLD,
    )
    return p


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

    from igv.boltz_score import (
        AFFINITY_SCORES,
        affinity_forward,
        affinity_head_forward,
        build_affinity_feats,
        enable_confidence_checkpointing,
        load_affinity_model,
        numerics_arm,
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
    cache_suffix = f"_gate_aff_{args.protein_chain_id}_{args.ligand_id}"
    feats, token_map = build_affinity_feats(
        protein_chains,
        args.ligand_id,
        cache_dir / f"boltz_affinity{cache_suffix}",
        args.device,
        ligand_smiles=args.ligand_smiles,
        ligand_ccd=args.ligand_ccd,
        use_msa_server=not args.no_msa_server,
    )

    n_tokens = int(feats["token_pad_mask"][0].sum().item())
    print(f"  L (tokens after affinity crop): {n_tokens}")
    print(f"  token_map entries: {len(token_map)}")

    x_pred = feats["coords"].detach()

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

    # ---- Baseline: zeros res_type ----
    res_type_orig = feats["res_type"]
    num_tokens_vocab = int(res_type_orig.shape[2]) if res_type_orig.dim() == 3 else None

    with torch.no_grad():
        baseline_feats = dict(feats)
        baseline_feats["res_type"] = torch.zeros_like(res_type_orig)
        s_bl = model.input_embedder(baseline_feats, affinity=True)
        f_baseline_val = float(affinity_head_forward(
            model, s_bl, z_nograd, x_pred, baseline_feats,
            args.score, flow_through_z=False,
        ))

    # ---- INCONCLUSIVE pre-check ----
    inconclusive_reason = check_inconclusive(f_x_val, f_baseline_val)
    if inconclusive_reason is not None:
        print(f"\n{'=' * 60}")
        print("Gate Affinity: INCONCLUSIVE")
        print(f"{'=' * 60}")
        print(f"  Reason: {inconclusive_reason}")
        print(f"  f(input) = {f_x_val}")
        print(f"  f(baseline) = {f_baseline_val}")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "gate": "affinity",
            "result": "INCONCLUSIVE",
            "reason": inconclusive_reason,
            "f_x": f_x_val,
            "f_baseline": f_baseline_val,
        }, indent=2) + "\n")
        return 2

    f_diff = f_x_val - f_baseline_val
    print(f"  f(input)    = {f_x_val:.6f}")
    print(f"  f(baseline) = {f_baseline_val:.6f}")
    print(f"  span        = {abs(f_diff):.6f}")

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
    print("\n--- IG completeness ladder (full-trunk path) ---")
    completeness_results = {}
    first_m_abs_err = None
    first_m_rel_err = None
    first_m_ig_sum = None

    for m in sorted(args.m_steps):
        print(f"\n  m={m}:")
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        reset_peak()
        t0 = time.time()

        ig_sum_val = 0.0
        baseline_res_type = torch.zeros_like(res_type_orig)

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
                ig_sum_val += float(
                    ((res_type_orig.detach().float() - baseline_res_type) * step_grad).sum()
                ) / m

            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

        wall = time.time() - t0
        ig_peak = peak_allocated_gib()

        abs_err = abs(ig_sum_val - f_diff)
        rel_err = abs_err / abs(f_diff) if abs(f_diff) > 1e-12 else float("inf")

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

    # ---- Write JSON ----
    summary = {
        "gate": "affinity",
        "result": gate_result,
        "path": "full_trunk",
        "score": args.score,
        "affinity_value": f_x_val,
        "f_baseline": f_baseline_val,
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
        "protein_chain_id": args.protein_chain_id,
        "ligand_id": args.ligand_id,
        "ligand_smiles": args.ligand_smiles,
        "ligand_ccd": args.ligand_ccd,
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
        },
        params={
            "score": args.score,
            "m_steps": args.m_steps,
            "completeness_threshold": args.completeness_threshold,
            "recycling_steps": args.recycling_steps,
        },
        arm={
            "score": args.score,
            "method": "gate_affinity",
            "path": "full_trunk",
            "n_tokens": n_tokens,
            **numerics_arm(),
        },
    )

    log.info("Wrote %s", out_json)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
