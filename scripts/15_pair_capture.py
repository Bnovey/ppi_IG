#!/usr/bin/env python3
"""Stage 15 -- capture the (L, L) pair-attribution map for 1JTG.

Runs pair-layer Integrated Gradients on the z tensor through the real
confidence head, with a mean_aa z baseline, real predicted x_pred, and a
convergence ladder at m_steps = 8, 16, 32, 64.

Output .npz key layout
----------------------
The primary output is an .npz consumed by ``scripts/13_coupling.py``::

    pair_ig          : (L, L) float32  -- symmetrised pair attribution at m=64
    pair_ig_m8       : (L, L) float32  -- symmetrised pair attribution at m=8
    pair_ig_m16      : (L, L) float32  -- symmetrised pair attribution at m=16
    pair_ig_m32      : (L, L) float32  -- symmetrised pair attribution at m=32
    pair_ig_m64      : (L, L) float32  -- same as pair_ig
    grad_m64         : (1, L, L, 128) float32 -- path-averaged gradient at m=64
    token_map_keys   : (N, 2) object   -- (chain_id, residue_index) per token
    token_map_values : (N,) int64      -- global token index per token
    chain_order      : (n_chains,) str  -- chains in token order
    chain_offsets    : (n_chains,) int64 -- start index of each chain
    chain_lengths    : (n_chains,) int64 -- length of each chain

``13_coupling.py`` tries keys ``pair_ig``, ``pair_map``, ``attribution``
in that order.  The token map lets it join SKEMPI residue numbers to
matrix indices.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import read_pdb_chains  # noqa: E402
from igv.metrics import step_convergence_spearman  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402
from igv.skempi import get_complex  # noqa: E402

log = logging.getLogger("pair_capture")

COMPLETENESS_ABS_THRESHOLD = 0.10

M_STEPS_LADDER = [8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Gate B concentration statistics (CPU, no boltz)
# ---------------------------------------------------------------------------


def gate_b_statistics(pair_map: np.ndarray) -> dict:
    """Compute Gate B concentration statistics on a symmetric (L, L) map.

    These are sanity-only, not a kill gate: a flat map means the contraction
    is wrong, not that biology is absent.
    """
    L = pair_map.shape[0]
    assert pair_map.shape == (L, L)

    iu = np.triu_indices(L, k=1)
    upper = np.abs(pair_map[iu])
    n_upper = len(upper)

    abs_map = np.abs(pair_map)

    diag_vals = np.diag(abs_map)
    off_diag_mask = ~np.eye(L, dtype=bool)
    off_diag_vals = abs_map[off_diag_mask]

    diag_mass = float(diag_vals.sum())
    off_diag_mass = float(off_diag_vals.sum())
    total_mass = diag_mass + off_diag_mass

    diag_share = diag_mass / total_mass if total_mass > 0 else 0.0
    off_diag_share = off_diag_mass / total_mass if total_mass > 0 else 0.0

    row_masses = abs_map.sum(axis=1)
    max_row_mass = float(row_masses.max())
    max_row_share = max_row_mass / total_mass if total_mass > 0 else 0.0

    n_zero = int((pair_map == 0.0).sum())
    total_entries = L * L
    zero_fraction = n_zero / total_entries

    mean_abs = float(upper.mean()) if n_upper > 0 else 0.0
    std_abs = float(upper.std()) if n_upper > 0 else 0.0
    cv = std_abs / mean_abs if mean_abs > 0 else 0.0

    stats: dict = {
        "L": L,
        "upper_triangle_n": n_upper,
        "min_abs": float(upper.min()) if n_upper > 0 else None,
        "max_abs": float(upper.max()) if n_upper > 0 else None,
        "median_abs": float(np.median(upper)) if n_upper > 0 else None,
        "mean_abs": mean_abs,
        "std_abs": std_abs,
        "cv_std_over_mean": cv,
        "near_constant": cv < 0.1,
        "zero_entries": n_zero,
        "zero_fraction": zero_fraction,
        "diag_mass_share": diag_share,
        "off_diag_mass_share": off_diag_share,
        "max_single_row_mass_share": max_row_share,
        "max_single_row_index": int(row_masses.argmax()),
    }

    sorted_upper = np.sort(upper)[::-1]
    total_upper_mass = float(upper.sum()) if n_upper > 0 else 0.0
    for k in [10, 100, 1000]:
        if k > n_upper:
            stats[f"top_{k}_mass_share"] = None
            stats[f"top_{k}_uniform_expectation"] = None
            stats[f"top_{k}_enrichment"] = None
            continue
        top_k_mass = float(sorted_upper[:k].sum())
        top_k_share = top_k_mass / total_upper_mass if total_upper_mass > 0 else 0.0
        uniform_expectation = k / n_upper
        enrichment = top_k_share / uniform_expectation if uniform_expectation > 0 else 0.0
        stats[f"top_{k}_mass_share"] = top_k_share
        stats[f"top_{k}_uniform_expectation"] = uniform_expectation
        stats[f"top_{k}_enrichment"] = enrichment

    return stats


def print_gate_b(stats: dict) -> None:
    """Print Gate B statistics in a human-readable block."""
    print(f"\n{'=' * 60}")
    print("Gate B: concentration statistics (sanity, not a kill gate)")
    print(f"{'=' * 60}")
    print(f"  L = {stats['L']}, upper triangle entries = {stats['upper_triangle_n']}")
    print("  abs(A) over upper triangle:")
    print(f"    min:    {stats['min_abs']:.6e}")
    print(f"    max:    {stats['max_abs']:.6e}")
    print(f"    median: {stats['median_abs']:.6e}")
    print(f"    mean:   {stats['mean_abs']:.6e}")
    print(f"    std:    {stats['std_abs']:.6e}")
    print(f"    CV (std/mean): {stats['cv_std_over_mean']:.4f}")
    if stats["near_constant"]:
        print("    *** MAP IS NEAR-CONSTANT (CV < 0.1) ***")
    print(f"  zero entries: {stats['zero_entries']} / {stats['L']**2} "
          f"({stats['zero_fraction']:.4f})")
    print(f"  diagonal mass share:     {stats['diag_mass_share']:.4f}")
    print(f"  off-diagonal mass share: {stats['off_diag_mass_share']:.4f}")
    print(f"  max single-row mass share: {stats['max_single_row_mass_share']:.4f} "
          f"(row {stats['max_single_row_index']})")
    for k in [10, 100, 1000]:
        key = f"top_{k}_mass_share"
        if stats.get(key) is not None:
            print(f"  top-{k:>4d} mass share: {stats[key]:.4f} "
                  f"vs uniform {stats[f'top_{k}_uniform_expectation']:.4f} "
                  f"= {stats[f'top_{k}_enrichment']:.2f}x enrichment")


# ---------------------------------------------------------------------------
# Token map serialisation
# ---------------------------------------------------------------------------


def serialise_token_map(
    token_map: dict,
    chains: dict[str, str],
) -> dict[str, np.ndarray]:
    """Serialise a token map and chain layout into .npz-compatible arrays."""
    sorted_items = sorted(token_map.items(), key=lambda kv: kv[1])
    keys = np.array([(k[0], str(k[1])) for k, _v in sorted_items], dtype=object)
    values = np.array([v for _k, v in sorted_items], dtype=np.int64)

    chain_order_list: list[str] = []
    offsets: list[int] = []
    lengths: list[int] = []
    seen: set[str] = set()
    for (chain_id, _resi), tok_idx in sorted_items:
        if chain_id not in seen:
            seen.add(chain_id)
            chain_order_list.append(chain_id)
            offsets.append(tok_idx)
            lengths.append(len(chains[chain_id]))

    return {
        "token_map_keys": keys,
        "token_map_values": values,
        "chain_order": np.array(chain_order_list, dtype=object),
        "chain_offsets": np.array(offsets, dtype=np.int64),
        "chain_lengths": np.array(lengths, dtype=np.int64),
    }


def deserialise_token_map(npz) -> dict[tuple[str, int], int]:
    """Reconstruct a token map from its .npz representation."""
    keys = npz["token_map_keys"]
    values = npz["token_map_values"]
    return {(str(k[0]), int(k[1])): int(v) for k, v in zip(keys, values)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture the (L, L) pair-attribution map for a SKEMPI complex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="1JTG",
        help="SKEMPI complex key. Default: 1JTG.",
    )
    p.add_argument(
        "--score", default="complex_pde",
        help="Score to attribute. Default: complex_pde.",
    )
    p.add_argument(
        "--structure", default=None,
        help="PDB stem under --cache-dir (without .pdb). Default: lowercase of --dataset.",
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
        "--m-steps", type=int, nargs="+", default=M_STEPS_LADDER,
        help=f"Step ladder for convergence. Default: {M_STEPS_LADDER}.",
    )
    p.add_argument(
        "--x-pred", default="predicted",
        choices=["zeros", "predicted"],
        help="x_pred mode. Default: predicted (run structure prediction).",
    )
    p.add_argument(
        "--out-npz",
        default="results/{dataset}_{score}_pair_ig.npz",
    )
    p.add_argument(
        "--out-json",
        default="results/{dataset}_{score}_pair_capture.json",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--completeness-threshold", type=float,
        default=COMPLETENESS_ABS_THRESHOLD,
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset = args.dataset.upper()
    cache_dir = Path(args.cache_dir)
    stem = args.structure or dataset.lower()
    pdb = cache_dir / f"{stem}.pdb"

    try:
        cx = get_complex(dataset)
    except KeyError:
        sys.exit(f"ERROR: complex {dataset} not registered in SKEMPI_COMPLEXES")

    chain_ids = list(cx.all_chains)

    struct_chains = None
    n_tokens = None
    if pdb.exists():
        all_chains = read_pdb_chains(pdb)
        struct_chains = {c: all_chains[c] for c in chain_ids if c in all_chains}
        n_tokens = sum(len(s) for s in struct_chains.values())

    out_npz = Path(args.out_npz.format(dataset=dataset, score=args.score))
    out_json = Path(args.out_json.format(dataset=dataset, score=args.score))

    ladder = sorted(args.m_steps)

    est_time_per_step_s = 1.5
    total_steps = sum(ladder)
    est_total_s = total_steps * est_time_per_step_s
    est_vram_gib = 9.0

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        print("Stage 15: pair-attribution capture\n")
        print(f"  dataset:             {dataset} ({cx.note})")
        print(f"  partner 1:          {cx.partner1}")
        print(f"  partner 2:          {cx.partner2}")
        print(f"  chain subset:       {chain_ids}")
        print(f"  PDB path:           {pdb}  "
              f"{'(exists)' if pdb.exists() else '(MISSING -- will fetch on VM)'}")
        if n_tokens is not None:
            print(f"  L (tokens):         {n_tokens}")
            z_bytes = n_tokens * n_tokens * 128 * 4
            print(f"  z tensor (fp32):    {z_bytes / 1024**2:.1f} MiB")
        else:
            print("  L (tokens):         (resolve on VM)")
        print(f"  score:              {args.score}")
        print(f"  x_pred mode:        {args.x_pred}")
        print("  z baseline:         mean_aa")
        print(f"  recycling_steps:    {args.recycling_steps}")
        print(f"  checkpoint_dir:     {args.checkpoint_dir}")
        print(f"  msa_server:         {'off' if args.no_msa_server else 'on'}")
        print()
        print(f"  Convergence ladder: {ladder}")
        print(f"  Total steps:        {total_steps}")
        print(f"  Estimated time:     ~{est_total_s / 60:.1f} min "
              f"(at ~{est_time_per_step_s:.1f} s/step)")
        print(f"  Estimated VRAM:     ~{est_vram_gib:.0f} GiB (m-independent)")
        print("  Note: Gauss-Legendre nodes do not nest; each rung is independent.")
        print()
        print(f"  Output .npz:        {out_npz}")
        print(f"  Output JSON:        {out_json}")
        print()
        print("  .npz key layout:")
        print("    pair_ig          : (L, L) float32  -- primary, m=64")
        print("    pair_ig_m{8,16,32,64} : (L, L) float32  -- per-rung")
        print("    grad_m64         : (1, L, L, 128) float32  -- path-avg grad")
        print("    token_map_keys   : (N, 2) object")
        print("    token_map_values : (N,) int64")
        print("    chain_order      : (n_chains,) str")
        print("    chain_offsets    : (n_chains,) int64")
        print("    chain_lengths    : (n_chains,) int64")
        print()
        print("  Gate B statistics will be computed on the m=64 map:")
        print("    min / max / median / mean of abs(A) over upper triangle")
        print("    top-k mass share vs uniform (k=10, 100, 1000)")
        print("    diagonal vs off-diagonal mass share")
        print("    max single-row mass share")
        print("    zero fraction, near-constant check (std/mean)")
        print()
        print("  Stage 2b: F(baseline) and F(input) also captured for alpha-profile.")
        print()
        print("  VM command:")
        cmd = "python3 scripts/15_pair_capture.py"
        if dataset != "1JTG":
            cmd += f" --dataset {dataset}"
        if args.score != "complex_pde":
            cmd += f" --score {args.score}"
        if args.x_pred != "predicted":
            cmd += f" --x-pred {args.x_pred}"
        if args.no_msa_server:
            cmd += " --no-msa-server"
        print(f"    {cmd}")
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
        build_mean_aa_pair_baseline,
        confidence_head_forward,
        embedder_only,
        enable_confidence_checkpointing,
        load_model,
        numerics_arm,
        predict_structure_coords,
        resolve_x_pred_mode,
        _chain_cache_key,
    )

    if args.score not in SCORES:
        sys.exit(f"--score must be one of {sorted(SCORES)}")

    if struct_chains is None:
        if not pdb.exists():
            sys.exit(f"PDB not found: {pdb}")
        all_chains = read_pdb_chains(pdb)
        struct_chains = {c: all_chains[c] for c in chain_ids if c in all_chains}
        n_tokens = sum(len(s) for s in struct_chains.values())

    x_pred_mode = resolve_x_pred_mode(args.x_pred)
    log.info("x_pred mode: %s", x_pred_mode)

    print(f"\n{'=' * 60}")
    print("Stage 15: pair-attribution capture")
    print(f"{'=' * 60}")
    chain_info = {c: len(s) for c, s in struct_chains.items()}
    print(f"  dataset = {dataset}, L = {n_tokens}, chains = {chain_info}")
    print(f"  score = {args.score}")
    print(f"  x_pred mode = {x_pred_mode}")
    print("  z baseline = mean_aa")
    print(f"  ladder = {ladder}")
    print()

    # ---- Load model ----
    model, boltz_version = load_model(args.checkpoint_dir, args.device)
    enable_confidence_checkpointing(model)

    # ---- Featurise ----
    cache_suffix = f"_{dataset}_{_chain_cache_key(struct_chains)}"
    feats, token_map = build_complex_feats(
        struct_chains,
        pdb,
        cache_dir / f"boltz_pair_capture{cache_suffix}",
        args.device,
        use_msa_server=not args.no_msa_server,
    )

    # ---- x_pred ----
    if x_pred_mode == "predicted":
        x_pred = predict_structure_coords(
            model, feats, cache_dir, struct_chains, dataset,
            recycling_steps=args.recycling_steps,
        )
        log.info(
            "Predicted x_pred: shape %s, abs max %.6g",
            list(x_pred.shape), float(x_pred.abs().max()),
        )
    else:
        x_pred = feats["coords"].detach()

    print(f"  x_pred mode: {x_pred_mode}")
    if x_pred_mode == "predicted":
        print(f"  x_pred abs max: {float(x_pred.abs().max()):.6g}")

    # ---- Trunk forward ----
    s_inputs = embedder_only(model, feats)

    log.info("Running trunk forward under no_grad...")
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

        for _ in range(args.recycling_steps + 1):
            s_ = s_init + model.s_recycle(model.s_norm(s_))
            z_ = z_init + model.z_recycle(model.z_norm(z_))
            z_ = z_ + model.msa_module(
                z_, s_inputs, feats, use_kernels=False,
            )
            s_, z_ = model.pairformer_module(
                s_, z_, mask=mask, pair_mask=pair_mask,
                use_kernels=False,
            )

    s = s_.detach()
    z_x = z_.detach()
    L = z_x.shape[1]
    print(f"  z.shape = {tuple(z_x.shape)}, L = {L}")

    # ---- Build mean_aa z baseline ----
    log.info("Building mean_aa z baseline...")
    z_baseline = build_mean_aa_pair_baseline(
        model, struct_chains, pdb, cache_dir, feats, args.device,
        recycling_steps=args.recycling_steps,
        use_msa_server=not args.no_msa_server,
    )
    print("  z_baseline: mean_aa")

    def score_fn(z):
        return confidence_head_forward(
            model, s_inputs.detach(), s.detach(), z, x_pred, feats,
            args.score,
        )

    # ---- F(baseline) and F(input) for alpha-profile (Stage 2b) ----
    with torch.no_grad():
        f_x = float(score_fn(z_x))
        f_baseline = float(score_fn(z_baseline))

    f_diff = f_x - f_baseline
    print(f"\n  F(input)    = {f_x:.6f}")
    print(f"  F(baseline) = {f_baseline:.6f}")
    print(f"  F(x) - F(b) = {f_diff:.6f}")

    # ---- Convergence ladder ----
    rung_results: dict[int, dict] = {}
    rung_maps: dict[int, np.ndarray] = {}

    for m in ladder:
        print(f"\n--- m_steps = {m} ---")
        reset_peak()
        t0 = time.time()
        result = pair_layer_ig(
            score_fn, z_baseline, z_x,
            m_steps=m,
            clear_cache_each_step=True,
            log_progress=True,
        )
        wall = time.time() - t0
        peak = peak_allocated_gib()

        imap = result.interaction_map.detach().cpu().numpy().astype(np.float32)
        ig_sum = float(result.interaction_map.sum())
        abs_err = abs(ig_sum - f_diff)
        rel_err = pair_completeness_error(result, f_x, f_baseline)
        completeness_pass = abs_err < args.completeness_threshold

        rung_maps[m] = imap

        rung_info = {
            "m_steps": m,
            "wall_s": wall,
            "ig_sum": ig_sum,
            "completeness_abs_err": abs_err,
            "completeness_rel_err": rel_err,
            "completeness_pass": completeness_pass,
            "peak_vram_gib": peak,
        }

        print(f"  wall time:          {wall:.1f} s")
        print(f"  ig_sum:             {ig_sum:.6f}")
        print(f"  completeness abs:   {abs_err:.6f}  "
              f"{'PASS' if completeness_pass else 'FAIL'} "
              f"(threshold {args.completeness_threshold})")
        print(f"  completeness rel:   {rel_err:.4f} (information only)")
        if peak is not None:
            print(f"  peak VRAM:          {peak:.2f} GiB")

        rung_results[m] = rung_info

        if m != ladder[-1]:
            del result
            torch.cuda.empty_cache()
        else:
            final_result = result

    # ---- Step convergence Spearman between consecutive rungs ----
    print(f"\n{'=' * 60}")
    print("Step convergence (Spearman between consecutive rungs)")
    print(f"{'=' * 60}")

    convergence_pairs: list[dict] = []
    for i in range(len(ladder) - 1):
        m_lo, m_hi = ladder[i], ladder[i + 1]
        iu = np.triu_indices(L, k=1)
        lo_flat = rung_maps[m_lo][iu]
        hi_flat = rung_maps[m_hi][iu]
        rho = step_convergence_spearman(lo_flat, hi_flat)
        pair_info = {
            "m_lo": m_lo,
            "m_hi": m_hi,
            "spearman": rho,
        }
        convergence_pairs.append(pair_info)
        print(f"  m={m_lo} vs m={m_hi}: rho = {rho:+.4f}")

    # ---- Gate B on the final (highest m) map ----
    final_m = ladder[-1]
    final_map = rung_maps[final_m]
    gate_b = gate_b_statistics(final_map)
    print_gate_b(gate_b)

    # ---- Save .npz ----
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, np.ndarray] = {
        "pair_ig": final_map,
    }
    for m in ladder:
        save_dict[f"pair_ig_m{m}"] = rung_maps[m]

    save_dict["grad_m64"] = final_result.grad.detach().cpu().numpy().astype(np.float32)

    token_map_arrays = serialise_token_map(token_map, struct_chains)
    save_dict.update(token_map_arrays)

    np.savez(out_npz, **save_dict)
    print(f"\nWrote {out_npz}")
    print(f"  keys: {sorted(save_dict.keys())}")

    # ---- Save JSON ----
    summary: dict = {
        "stage": "15_pair_capture",
        "dataset": dataset,
        "score": args.score,
        "L": L,
        "chains": chain_info,
        "x_pred_mode": x_pred_mode,
        "z_baseline_mode": "mean_aa",
        "f_x": f_x,
        "f_baseline": f_baseline,
        "f_diff": f_diff,
        "ladder": ladder,
        "rung_results": {str(k): v for k, v in rung_results.items()},
        "convergence": convergence_pairs,
        "gate_b": gate_b,
        "npz_path": str(out_npz),
        "npz_keys": sorted(save_dict.keys()),
        "boltz_version": boltz_version,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    print(f"Wrote {out_json}")

    prov_write(
        out_json,
        stage="15_pair_capture",
        inputs={"dataset": dataset, "structure": str(pdb)},
        params={
            "score": args.score,
            "x_pred_mode": x_pred_mode,
            "z_baseline_mode": "mean_aa",
            "ladder": ladder,
            "recycling_steps": args.recycling_steps,
            "completeness_threshold": args.completeness_threshold,
        },
        arm={
            "score": args.score,
            "dataset": dataset,
            "chain_subset": list(struct_chains),
            "n_tokens": n_tokens,
            "x_pred_mode": x_pred_mode,
            "method": "pair_ig",
            **numerics_arm(),
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
