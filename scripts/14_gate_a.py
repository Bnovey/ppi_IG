#!/usr/bin/env python3
"""Stage 14 (GATE A) -- z-gradient de-risking for Phase 5.

Does gradient flow through the real checkpointed confidence head to an
externally supplied pair tensor z?  If not, Phase 5 is dead and nothing
downstream is worth buying.

Pass criteria (all five must hold):
  1. z.grad is not None
  2. z.grad is not uniformly zero
  3. z.grad has shape (1, L, L, 128)
  4. z.grad is all finite (no NaN/Inf)
  5. completeness absolute error < COMPLETENESS_ABS_THRESHOLD

Outcomes:
  exit 0 = PASS   -- all five criteria satisfied
  exit 1 = FAIL   -- at least one criterion failed
  exit 2 = INCONCLUSIVE -- setup was degenerate, says nothing about the seam

Target cost: ~$2 on A100 at L ~ 352, m_steps = 5.
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

from igv.data import read_pdb_chains, resolve_chain_subset  # noqa: E402

log = logging.getLogger("gate_a")

COMPLETENESS_ABS_THRESHOLD = 0.10
COLLAPSED_SPAN_THRESHOLD = 1e-6


# ---------------------------------------------------------------------------
# Pure helpers (CPU, no torch, no boltz -- testable and usable in --dry-run)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="1VFB",
        help="SKEMPI complex key. Default: 1VFB.",
    )
    p.add_argument(
        "--chain", default="A",
        help="Chain to include (must be in --chain-subset). Default: A.",
    )
    p.add_argument(
        "--chain-subset", default="A,B,C",
        help="Comma-separated chain IDs to featurise. Default: A,B,C "
             "(full 1VFB complex, L=352).",
    )
    p.add_argument("--score", default="complex_pde")
    p.add_argument(
        "--m-steps", type=int, default=5,
        help="Quadrature points for pair-layer IG. Default: 5.",
    )
    p.add_argument(
        "--structure", default=None,
        help="PDB stem under --cache-dir (without .pdb). "
             "Default: lowercase of --dataset.",
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
        help="Trunk recycling iterations. Default: 1.",
    )
    p.add_argument(
        "--out", default="results/gate_a_{dataset}_{score}.json",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--completeness-threshold", type=float,
        default=COMPLETENESS_ABS_THRESHOLD,
        help=f"Absolute completeness error threshold. "
             f"Default: {COMPLETENESS_ABS_THRESHOLD}.",
    )
    p.add_argument(
        "--z-baseline", default="mean_aa",
        choices=["mean_aa", "zeros"],
        help="Baseline for z interpolation. Default: mean_aa (pair tensor "
             "from running the trunk on the mean-AA baseline sequence). "
             "'zeros' available as a comparison mode.",
    )
    p.add_argument(
        "--msa", default=None,
        choices=["empty"],
        help="MSA mode. Default: use MSA server (production path). "
             "Pass --msa empty for a fast fallback without the server.",
    )
    return p


def check_inconclusive(
    f_x: float | None,
    f_baseline: float | None,
    z_x_is_constant: bool,
    path_is_constant: bool,
    msa_mode: str,
) -> str | None:
    """Return a reason string if the setup is degenerate, else None.

    Pure function -- no torch, testable in --dry-run.
    """
    if f_x is None or f_baseline is None:
        return "f(z_x) or f(z_baseline) could not be computed"

    if not math.isfinite(f_x):
        return f"f(z_x) = {f_x} is non-finite"
    if not math.isfinite(f_baseline):
        return f"f(z_baseline) = {f_baseline} is non-finite"

    if f_x == 0.0:
        return "f(z_x) is exactly zero"
    if f_baseline == 0.0:
        return "f(z_baseline) is exactly zero"

    span = abs(f_x - f_baseline)
    if span < COLLAPSED_SPAN_THRESHOLD:
        return (
            f"|f(z_x) - f(z_baseline)| = {span:.2e} < {COLLAPSED_SPAN_THRESHOLD:.0e} "
            f"-- collapsed span makes completeness meaningless"
        )

    if z_x_is_constant:
        return "z_x is all zeros or constant (no spatial structure)"

    if path_is_constant:
        return (
            "score is constant across the interpolation path "
            "(f at alpha=0, 0.5, 1 are identical)"
        )

    return None


def evaluate_checks(
    z_grad_not_none: bool,
    z_grad_max_abs: float | None,
    z_grad_zero_frac: float | None,
    z_grad_shape: tuple | None,
    expected_shape: tuple,
    z_grad_n_nan: int | None,
    z_grad_n_inf: int | None,
    completeness_abs_err: float | None,
    completeness_rel_err: float | None,
    ig_sum: float | None,
    f_diff: float | None,
    threshold: float = COMPLETENESS_ABS_THRESHOLD,
) -> list[dict]:
    """Evaluate all five gate criteria. Pure function, no torch.

    Returns a list of dicts, each with keys ``name``, ``passed``, ``detail``.
    """
    checks: list[dict] = []

    checks.append({
        "name": "z.grad is not None",
        "passed": z_grad_not_none,
        "detail": "",
    })

    if z_grad_max_abs is not None:
        passed = z_grad_max_abs > 0
        detail = (
            f"max|z.grad|={z_grad_max_abs:.6e}, "
            f"zero_frac={z_grad_zero_frac:.4f}"
        )
    else:
        passed = False
        detail = "z.grad is None"
    checks.append({
        "name": "z.grad not uniformly zero",
        "passed": passed,
        "detail": detail,
    })

    if z_grad_shape is not None:
        passed = tuple(z_grad_shape) == tuple(expected_shape)
        detail = f"actual={tuple(z_grad_shape)}, expected={tuple(expected_shape)}"
    else:
        passed = False
        detail = "z.grad is None"
    checks.append({
        "name": f"z.grad shape {tuple(expected_shape)}",
        "passed": passed,
        "detail": detail,
    })

    if z_grad_n_nan is not None and z_grad_n_inf is not None:
        passed = z_grad_n_nan == 0 and z_grad_n_inf == 0
        detail = f"NaN={z_grad_n_nan}, Inf={z_grad_n_inf}"
    else:
        passed = False
        detail = "z.grad is None"
    checks.append({
        "name": "z.grad all finite",
        "passed": passed,
        "detail": detail,
    })

    if completeness_abs_err is not None and f_diff is not None:
        passed = completeness_abs_err < threshold
        detail = (
            f"abs_err={completeness_abs_err:.6f}, "
            f"rel_err={completeness_rel_err:.4f}, "
            f"ig_sum={ig_sum:.6f}, "
            f"f(x)-f(b)={f_diff:.6f}, "
            f"span={abs(f_diff):.6f}"
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
    """Print labelled PASS/FAIL lines. Returns True if all pass."""
    all_pass = True
    for c in checks:
        tag = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_pass = False
        suffix = f"  ({c['detail']})" if c["detail"] else ""
        print(f"  [{tag}] {c['name']}{suffix}")
    return all_pass


def resolve_l(args) -> tuple[dict[str, str] | None, int | None, str]:
    """Resolve chain subset and L from args. Returns (chains, L, label).

    Works without boltz or torch -- used in --dry-run.
    """
    cache_dir = Path(args.cache_dir)
    stem = args.structure or args.dataset.lower()
    pdb = cache_dir / f"{stem}.pdb"

    if not pdb.exists():
        return None, None, "?"

    all_chains = read_pdb_chains(pdb)
    struct_chains, label = resolve_chain_subset(
        all_chains, args.chain_subset, args.chain,
    )
    n_tokens = sum(len(s) for s in struct_chains.values())
    return struct_chains, n_tokens, label


def _is_single_chain(chain_subset: str) -> bool:
    return "," not in chain_subset


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _run_trunk(model, s_inputs, feats, recycling_steps=1):
    """Run the full trunk under no_grad. Returns ``(s, z)`` after recycling.

    Mirrors the non-checkpointed branch of ``confidence_forward`` (lines
    1766-1775 in ``boltz_score.py``).  No gradient checkpointing is needed
    under ``no_grad``.
    """
    import torch

    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    with torch.no_grad():
        rel_pos = model.rel_pos(feats)
        token_bonds_z = model.token_bonds(feats["token_bonds"].float())
        contact_z = model.contact_conditioning(feats)

        s_init = model.s_init(s_inputs)
        z_init = (
            model.z_init_1(s_inputs)[:, :, None, :]
            + model.z_init_2(s_inputs)[:, None, :, :]
            + rel_pos + token_bonds_z + contact_z
        )

        s_ = torch.zeros_like(s_init)
        z_ = torch.zeros_like(z_init)

        for _ in range(recycling_steps + 1):
            s_ = s_init + model.s_recycle(model.s_norm(s_))
            z_ = z_init + model.z_recycle(model.z_norm(z_))
            z_ = z_ + model.msa_module(
                z_, s_inputs, feats, use_kernels=False,
            )
            s_, z_ = model.pairformer_module(
                s_, z_, mask=mask, pair_mask=pair_mask,
                use_kernels=False,
            )

    return s_.detach(), z_.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_dir = Path(args.cache_dir)
    stem = args.structure or args.dataset.lower()
    pdb = cache_dir / f"{stem}.pdb"

    struct_chains, n_tokens, subset_label = resolve_l(args)
    out_json = Path(args.out.format(dataset=args.dataset, score=args.score))

    msa_mode = args.msa or "server"
    single_chain = _is_single_chain(args.chain_subset)

    # ------------------------------------------------------------------
    # Dry run: print the full plan, no boltz, no GPU
    # ------------------------------------------------------------------
    if args.dry_run:
        print("Gate A de-risking test for Phase 5\n")
        print(f"  dataset:              {args.dataset}")
        print(f"  structure:            {pdb}")
        print(f"  score:                {args.score}")
        print(f"  m_steps:              {args.m_steps}")
        print(f"  recycling_steps:      {args.recycling_steps}")
        print(f"  chain:                {args.chain}")
        print(f"  chain_subset:         {args.chain_subset}")
        if struct_chains is not None:
            chain_info = {c: len(s) for c, s in struct_chains.items()}
            print(f"  resolved chains:      {chain_info}")
            print(f"  L (tokens):           {n_tokens}")
            z_bytes = n_tokens * n_tokens * 128 * 4
            print(f"  z tensor (fp32):      {z_bytes / 1024**2:.1f} MiB")
        else:
            print(f"  PDB not found:        {pdb} (will resolve on VM)")
        print(f"  z_baseline:           {args.z_baseline}")
        print(f"  msa:                  {msa_mode}")
        print(f"  completeness thr:     abs < {args.completeness_threshold}")
        print(f"  output:               {out_json}")
        print(f"  device:               {args.device}")
        print(f"  checkpoint_dir:       {args.checkpoint_dir}")
        print(f"  no_msa_server:        {args.no_msa_server}")
        if single_chain:
            print()
            print("  *** WARNING: single-chain subset selected. complex_pde "
                  "may be degenerate on a single chain -- the score may not "
                  "move and gradients would be legitimately zero. Consider "
                  "using the full complex (--chain-subset A,B,C for 1VFB). ***")
        print()
        print("Outcomes:")
        print("  exit 0 = PASS          all five criteria satisfied")
        print("  exit 1 = FAIL          at least one criterion failed")
        print("  exit 2 = INCONCLUSIVE  setup degenerate, not a seam verdict")
        print()
        print("Pass criteria (all five):")
        print("  1. z.grad is not None")
        print("  2. z.grad not uniformly zero  (reports max abs, zero fraction)")
        print("  3. z.grad shape (1, L, L, 128)")
        print("  4. z.grad all finite  (reports NaN/Inf counts)")
        print(f"  5. completeness absolute error < {args.completeness_threshold}")
        print("     (also reports relative error as information)")
        print()
        print("Also measured:")
        print("  (a) trainable param count  -- are parameters frozen?")
        print("  (b) s dimension")
        print("  (c) z.grad.dtype  -- stays fp32 under autocast?")
        print("  peak VRAM (torch.cuda.max_memory_allocated)")
        print("  param .grad bytes after one backward")
        print()
        print("VM command:")
        cmd = "python3 scripts/14_gate_a.py"
        if args.dataset != "1VFB":
            cmd += f" --dataset {args.dataset}"
        if args.chain != "A":
            cmd += f" --chain {args.chain}"
        if args.chain_subset != "A,B,C":
            cmd += f" --chain-subset {args.chain_subset}"
        if args.score != "complex_pde":
            cmd += f" --score {args.score}"
        if args.m_steps != 5:
            cmd += f" --m-steps {args.m_steps}"
        if args.z_baseline != "mean_aa":
            cmd += f" --z-baseline {args.z_baseline}"
        if args.msa is not None:
            cmd += f" --msa {args.msa}"
        print(f"  {cmd}")
        return 0

    # ------------------------------------------------------------------
    # GPU path
    # ------------------------------------------------------------------
    from igv.gpu import peak_allocated_gib, require_vram, reset_peak  # noqa: E402

    require_vram()

    import torch  # noqa: E402

    from igv.attrib import pair_completeness_error, pair_layer_ig  # noqa: E402
    from igv.boltz_score import (  # noqa: E402
        SCORES,
        build_complex_feats,
        confidence_head_forward,
        embedder_only,
        enable_confidence_checkpointing,
        load_model,
        numerics_arm,
    )
    from igv.provenance import write as prov_write  # noqa: E402

    if args.score not in SCORES:
        raise SystemExit(f"--score must be one of {sorted(SCORES)}")

    if struct_chains is None:
        if not pdb.exists():
            raise SystemExit(f"PDB not found: {pdb}")
        all_chains = read_pdb_chains(pdb)
        struct_chains, subset_label = resolve_chain_subset(
            all_chains, args.chain_subset, args.chain,
        )
        n_tokens = sum(len(s) for s in struct_chains.values())

    print(f"\n{'=' * 60}")
    print("Gate A: z-gradient de-risking")
    print(f"{'=' * 60}")
    chain_info = {c: len(s) for c, s in struct_chains.items()}
    print(f"  L = {n_tokens} tokens, chains = {chain_info}")
    print(f"  score = {args.score}, m_steps = {args.m_steps}")
    print(f"  z_baseline = {args.z_baseline}")
    print(f"  msa = {msa_mode}")
    if single_chain:
        print()
        print("  *** WARNING: single-chain subset. complex_pde may be "
              "degenerate on one chain -- if the score does not move, "
              "gradients are legitimately zero and the result is "
              "INCONCLUSIVE, not a seam failure. ***")
    print()

    # ---- Load model ----
    model, boltz_version = load_model(args.checkpoint_dir, args.device)

    n_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    n_frozen = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print(
        f"Runtime unknown (a): trainable params = {n_trainable:,}"
        f"  (frozen = {n_frozen:,})"
    )

    enable_confidence_checkpointing(model)

    # ---- Featurise ----
    cache_suffix = f"_{args.dataset}"
    if subset_label != "all":
        cache_suffix += f"_{subset_label}"
    msa_kwarg = {"msa": args.msa} if args.msa is not None else {}
    feats, _token_map = build_complex_feats(
        struct_chains,
        pdb,
        cache_dir / f"boltz_gate_a{cache_suffix}",
        args.device,
        use_msa_server=not args.no_msa_server,
        **msa_kwarg,
    )

    x_pred = feats["coords"].detach()
    s_inputs = embedder_only(model, feats)

    # ---- Trunk forward (no_grad) ----
    log.info("Running trunk forward under no_grad...")
    s, z_trunk = _run_trunk(model, s_inputs, feats, args.recycling_steps)

    print(f"Runtime unknown (b): s.shape = {tuple(s.shape)}")

    z_x = z_trunk.clone()
    L = z_x.shape[1]
    print(f"  z.shape = {tuple(z_x.shape)}, L = {L}")

    # ---- Build z baseline ----
    if args.z_baseline == "zeros":
        z_baseline = torch.zeros_like(z_x)
        print("  z_baseline: zeros")
    else:
        from igv.boltz_score import build_mean_aa_pair_baseline  # noqa: E402
        log.info("Building mean_aa z baseline (trunk forward on mean-AA "
                 "baseline sequence)...")
        z_baseline = build_mean_aa_pair_baseline(
            model, struct_chains, pdb, cache_dir, feats, args.device,
            recycling_steps=args.recycling_steps,
            use_msa_server=not args.no_msa_server,
        )
        print("  z_baseline: mean_aa (pair tensor from trunk on mean-AA "
              "baseline sequence)")

    def score_fn(z):
        return confidence_head_forward(
            model, s_inputs.detach(), s.detach(), z, x_pred, feats,
            args.score,
        )

    # ---- INCONCLUSIVE pre-checks ----
    print("\n--- Pre-checks for INCONCLUSIVE conditions ---")
    z_x_is_constant = bool(
        (z_x == 0).all() or (z_x == z_x.flatten()[0]).all()
    )

    with torch.no_grad():
        f_x = float(score_fn(z_x))
        f_baseline = float(score_fn(z_baseline))
        z_mid = z_baseline + 0.5 * (z_x - z_baseline)
        f_mid = float(score_fn(z_mid))

    path_is_constant = (f_x == f_mid == f_baseline)

    inconclusive_reason = check_inconclusive(
        f_x=f_x,
        f_baseline=f_baseline,
        z_x_is_constant=z_x_is_constant,
        path_is_constant=path_is_constant,
        msa_mode=msa_mode,
    )

    if inconclusive_reason is not None:
        print(f"\n{'=' * 60}")
        print("Gate A: INCONCLUSIVE")
        print(f"{'=' * 60}")
        print(f"  Reason: {inconclusive_reason}")
        print(f"  f(z_x) = {f_x}")
        print(f"  f(z_baseline) = {f_baseline}")
        print(f"  f(z_mid) = {f_mid}")
        if msa_mode == "empty":
            print()
            print("  NOTE: msa=empty was used. An empty MSA is a plausible "
                  "cause of a degenerate score -- Boltz-2's z is MSA-driven.")
        print()
        print("  This says nothing about whether the seam works -- "
              "do not conclude Gate A failed.")
        print()

        summary = {
            "gate": "A",
            "result": "INCONCLUSIVE",
            "reason": inconclusive_reason,
            "dataset": args.dataset,
            "score": args.score,
            "L": L,
            "chains": chain_info,
            "f_x": f_x,
            "f_baseline": f_baseline,
            "f_mid": f_mid,
            "z_baseline_mode": args.z_baseline,
            "msa_mode": msa_mode,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )
        return 2

    f_diff = f_x - f_baseline
    print(f"  f(z_x)       = {f_x:.6f}")
    print(f"  f(z_baseline) = {f_baseline:.6f}")
    print(f"  f(z_mid)     = {f_mid:.6f}")
    print(f"  span |f(x)-f(b)| = {abs(f_diff):.6f}")
    print("  Pre-checks: OK (no INCONCLUSIVE condition detected)")

    # ---- Single backward to inspect z.grad ----
    print("\n--- Single backward ---")
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    reset_peak()
    z_test = z_x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        score_test = score_fn(z_test)
        log.info("score_test = %.6f (requires_grad=%s)", score_test.item(), score_test.requires_grad)
        score_test.backward()

    z_grad = z_test.grad
    single_bwd_peak = peak_allocated_gib()

    z_grad_dtype_str = str(z_grad.dtype) if z_grad is not None else "None"
    print(f"Runtime unknown (c): z.grad.dtype = {z_grad_dtype_str}")

    param_grad_bytes = 0
    param_grad_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_grad_bytes += p.grad.numel() * p.grad.element_size()
            param_grad_count += 1
    print(
        f"Param .grad after 1 backward: {param_grad_count} tensors, "
        f"{param_grad_bytes / 1024**2:.1f} MiB ({param_grad_bytes:,} bytes)"
    )

    print(
        f"Peak VRAM (single backward): "
        f"{single_bwd_peak:.2f} GiB"
        if single_bwd_peak is not None
        else "Peak VRAM (single backward): n/a"
    )

    z_grad_not_none = z_grad is not None
    z_grad_max_abs = float(z_grad.abs().max()) if z_grad is not None else None
    z_grad_zero_frac = float((z_grad == 0).float().mean()) if z_grad is not None else None
    z_grad_shape = tuple(z_grad.shape) if z_grad is not None else None
    z_grad_n_nan = int(z_grad.isnan().sum()) if z_grad is not None else None
    z_grad_n_inf = int(z_grad.isinf().sum()) if z_grad is not None else None

    del z_test, score_test

    # ---- Reference scores under no_grad ----
    print("\n--- Reference scores (no_grad) ---")
    print(f"  f(z_x)      = {f_x:.6f}")
    print(f"  f(z_b)      = {f_baseline:.6f}")
    print(f"  f(x) - f(b) = {f_diff:.6f}")
    print(f"  span |f(x) - f(b)| = {abs(f_diff):.6f}")

    # ---- pair_layer_ig ----
    print(f"\n--- pair_layer_ig (m_steps={args.m_steps}) ---")
    reset_peak()
    t0 = time.time()
    result = pair_layer_ig(
        score_fn, z_baseline, z_x,
        m_steps=args.m_steps,
        log_progress=True,
    )
    wall = time.time() - t0
    ig_peak = peak_allocated_gib()
    print(f"  wall time: {wall:.1f} s")
    if ig_peak is not None:
        print(f"  peak VRAM (pair_layer_ig): {ig_peak:.2f} GiB")

    rel_err = pair_completeness_error(result, f_x, f_baseline)
    ig_sum = float(result.interaction_map.sum())
    abs_err = abs(ig_sum - f_diff)

    # ---- Evaluate gate criteria ----
    print(f"\n{'=' * 60}")
    print("Gate A results:")
    print(f"{'=' * 60}")

    expected_shape = (1, L, L, 128)
    checks = evaluate_checks(
        z_grad_not_none=z_grad_not_none,
        z_grad_max_abs=z_grad_max_abs,
        z_grad_zero_frac=z_grad_zero_frac,
        z_grad_shape=z_grad_shape,
        expected_shape=expected_shape,
        z_grad_n_nan=z_grad_n_nan,
        z_grad_n_inf=z_grad_n_inf,
        completeness_abs_err=abs_err,
        completeness_rel_err=rel_err,
        ig_sum=ig_sum,
        f_diff=f_diff,
        threshold=args.completeness_threshold,
    )

    all_pass = print_checks(checks)

    gate_result = "PASS" if all_pass else "FAIL"
    print(f"\n  z_baseline: {args.z_baseline}")
    print(f"  msa: {msa_mode}")
    print(f"  Gate A: {gate_result}")

    # ---- Write JSON ----
    summary = {
        "gate": "A",
        "result": gate_result,
        "dataset": args.dataset,
        "score": args.score,
        "L": L,
        "m_steps": args.m_steps,
        "chains": chain_info,
        "f_x": f_x,
        "f_baseline": f_baseline,
        "f_diff": f_diff,
        "f_diff_abs": abs(f_diff),
        "ig_sum": ig_sum,
        "completeness_abs_err": abs_err,
        "completeness_rel_err": rel_err,
        "z_grad_not_none": z_grad_not_none,
        "z_grad_max_abs": z_grad_max_abs,
        "z_grad_zero_frac": z_grad_zero_frac,
        "z_grad_shape": list(z_grad_shape) if z_grad_shape else None,
        "z_grad_dtype": z_grad_dtype_str,
        "z_grad_n_nan": z_grad_n_nan,
        "z_grad_n_inf": z_grad_n_inf,
        "trainable_params": n_trainable,
        "frozen_params": n_frozen,
        "s_shape": [int(d) for d in s.shape],
        "param_grad_bytes": param_grad_bytes,
        "param_grad_count": param_grad_count,
        "peak_vram_single_bwd_gib": single_bwd_peak,
        "peak_vram_pair_ig_gib": ig_peak,
        "wall_s": wall,
        "boltz_version": boltz_version,
        "z_baseline_mode": args.z_baseline,
        "msa_mode": msa_mode,
        "checks": checks,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    prov_write(
        out_json,
        stage="14_gate_a",
        inputs={"dataset": args.dataset, "structure": str(pdb)},
        params={
            "m_steps": args.m_steps,
            "chain_subset": list(struct_chains),
            "recycling_steps": args.recycling_steps,
            "completeness_threshold": args.completeness_threshold,
            "z_baseline": args.z_baseline,
            "msa": msa_mode,
        },
        arm={
            "score": args.score,
            "dataset": args.dataset,
            "chain": args.chain,
            "chain_subset": list(struct_chains),
            "n_tokens": n_tokens,
            "method": "gate_a",
            **numerics_arm(),
        },
    )

    log.info("Wrote %s", out_json)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
