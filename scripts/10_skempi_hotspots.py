#!/usr/bin/env python3
"""Correlate per-residue gradient norms with SKEMPI 2.0 ΔΔG hot spots."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.metrics import (
    auroc,
    auprc,
    bootstrap_ci,
    hotspot_precision_at_k,
    hotspot_precision_chance,
    partial_spearman,
    precision_at_k,
    spearman,
    topk_overlap_chance,
)
from igv.provenance import write as prov_write
from igv.skempi import (
    SKEMPI_MUTATION_COL,
    add_ddg,
    compute_confounds,
    filter_complex,
    get_complex,
    load_skempi,
    map_mutations_to_indices,
    parse_mutation,
    read_pdb_residue_ids,
    single_point,
)

log = logging.getLogger(__name__)

HOTSPOT_THRESHOLD = 2.0  # kcal/mol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate gradient norms with SKEMPI ΔΔG hot spots",
    )
    parser.add_argument("--complex", required=True, help="PDB ID, e.g. 3HFM")
    parser.add_argument("--grad", required=True, help="Path to gradient .npz")
    parser.add_argument("--chain", required=True, help="Chain the gradient is for")
    parser.add_argument("--pdb", default=None, help="Path to PDB file (default: cache-dir/<pdb>.pdb)")
    parser.add_argument("--cache-dir", default="data/raw", help="Cache directory")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--allow-mismatch", action="store_true",
        help="Log and exclude wild-type mismatches instead of failing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    grad_path = Path(args.grad)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)
    pdb_id = args.complex.upper()
    chain = args.chain

    if not grad_path.exists():
        sys.exit(f"ERROR: gradient file not found: {grad_path}")

    # --- Load gradient and compute per-residue L2 norms ---
    npz = np.load(grad_path, allow_pickle=True)
    grad_chain = npz["grad_chain"]  # (n_residues, 384)
    norms = np.linalg.norm(grad_chain, axis=1)  # (n_residues,)
    log.info("Loaded gradient: %d residues, norm range [%.4f, %.4f]",
             len(norms), norms.min(), norms.max())

    # --- Load PDB residue mapping ---
    pdb_path = Path(args.pdb) if args.pdb else cache_dir / f"{pdb_id.lower()}.pdb"
    if not pdb_path.exists():
        sys.exit(f"ERROR: PDB file not found: {pdb_path}")

    residue_ids, sequences = read_pdb_residue_ids(pdb_path)
    if chain not in residue_ids:
        sys.exit(f"ERROR: chain {chain} not found in PDB. Available: {list(residue_ids)}")

    chain_ids = residue_ids[chain]
    chain_seq = sequences[chain]
    log.info("Chain %s: %d residues, IDs %s..%s",
             chain, len(chain_ids), chain_ids[0], chain_ids[-1])

    if len(norms) != len(chain_ids):
        sys.exit(
            f"ERROR: gradient has {len(norms)} residues but PDB chain {chain} "
            f"has {len(chain_ids)} residues"
        )

    # --- Load and filter SKEMPI ---
    df = load_skempi(cache_dir)
    df = filter_complex(df, pdb_id)
    df = single_point(df)
    df = add_ddg(df)
    log.info("SKEMPI after filtering: %d single-point mutations with ΔΔG", len(df))

    mutations = [parse_mutation(m.strip()) for m in df[SKEMPI_MUTATION_COL]]
    chain_mutations = [(mut, i) for i, mut in enumerate(mutations) if mut.chain == chain]

    if not chain_mutations:
        sys.exit(f"ERROR: no mutations on chain {chain} for {pdb_id}")

    chain_muts = [m for m, _ in chain_mutations]
    chain_ddgs = [float(df.iloc[i]["ddg_kcal_mol"]) for _, i in chain_mutations]

    mapped, mismatches = map_mutations_to_indices(
        chain_muts, chain_ids, chain_seq,
        allow_mismatch=args.allow_mismatch,
    )

    n_excluded = len(mismatches)
    log.info("Mapped %d mutations, %d mismatches excluded", len(mapped), n_excluded)

    mapped_with_ddg: list[tuple[int, float]] = []
    for mut, seq_idx in mapped:
        orig_pos = chain_muts.index(mut)
        mapped_with_ddg.append((seq_idx, chain_ddgs[orig_pos]))

    # --- Aggregate per position (max and mean) ---
    pos_ddgs: dict[int, list[float]] = {}
    for seq_idx, ddg_val in mapped_with_ddg:
        pos_ddgs.setdefault(seq_idx, []).append(ddg_val)

    positions = sorted(pos_ddgs)
    grad_scores = np.array([norms[p] for p in positions])
    ddg_max = np.array([max(pos_ddgs[p]) for p in positions])
    ddg_mean = np.array([np.mean(pos_ddgs[p]) for p in positions])
    ddg_abs_max = np.abs(ddg_max)
    ddg_abs_mean = np.abs(ddg_mean)
    n_mutations_per_pos = [len(pos_ddgs[p]) for p in positions]

    n_residues = len(positions)
    n_mutations_used = len(mapped)

    log.info("Unique residue positions: %d (from %d mutations)", n_residues, n_mutations_used)

    # --- Metrics ---
    rho_abs_max = spearman(grad_scores, ddg_abs_max)
    rho_abs_mean = spearman(grad_scores, ddg_abs_mean)
    rho_signed_max = spearman(grad_scores, ddg_max)
    rho_signed_mean = spearman(grad_scores, ddg_mean)

    print(f"Spearman (norm vs |ΔΔG|, max agg):  {rho_abs_max:.4f}")
    print(f"Spearman (norm vs |ΔΔG|, mean agg): {rho_abs_mean:.4f}")
    print(f"Spearman (norm vs  ΔΔG,  max agg):  {rho_signed_max:.4f}")
    print(f"Spearman (norm vs  ΔΔG,  mean agg): {rho_signed_mean:.4f}")

    # Binary hot-spot labels (ΔΔG >= 2.0 kcal/mol)
    hot_max = (ddg_max >= HOTSPOT_THRESHOLD).astype(float)
    hot_mean = (ddg_mean >= HOTSPOT_THRESHOLD).astype(float)

    prec_results = {}
    for k in (5, 10, 20):
        k_eff = min(k, n_residues)
        if k_eff == 0:
            continue

        # Rank overlap: fraction of top-k by pred also in top-k by true (continuous)
        overlap_max = precision_at_k(grad_scores, ddg_max, k_eff)
        overlap_mean = precision_at_k(grad_scores, ddg_mean, k_eff)
        overlap_chance = topk_overlap_chance(k_eff, n_residues)
        prec_results[f"topk_overlap_at_{k}_max"] = overlap_max
        prec_results[f"topk_overlap_at_{k}_mean"] = overlap_mean
        prec_results[f"topk_overlap_at_{k}_chance"] = overlap_chance

        # Hot-spot precision: fraction of top-k by pred that are true hot spots
        hp_max = hotspot_precision_at_k(grad_scores, hot_max, k_eff)
        hp_mean = hotspot_precision_at_k(grad_scores, hot_mean, k_eff)
        hp_chance_max = hotspot_precision_chance(int(hot_max.sum()), n_residues)
        hp_chance_mean = hotspot_precision_chance(int(hot_mean.sum()), n_residues)
        prec_results[f"hotspot_precision_at_{k}_max"] = hp_max
        prec_results[f"hotspot_precision_at_{k}_mean"] = hp_mean
        prec_results[f"hotspot_precision_at_{k}_chance_max"] = hp_chance_max
        prec_results[f"hotspot_precision_at_{k}_chance_mean"] = hp_chance_mean

        print(f"Top-k overlap @{k} (max agg):  {overlap_max:.4f}  chance={overlap_chance:.4f}")
        print(f"Top-k overlap @{k} (mean agg): {overlap_mean:.4f}  chance={overlap_chance:.4f}")
        print(f"Hotspot prec  @{k} (max agg):  {hp_max:.4f}  chance={hp_chance_max:.4f}")
        print(f"Hotspot prec  @{k} (mean agg): {hp_mean:.4f}  chance={hp_chance_mean:.4f}")

    n_hotspots_max = int(hot_max.sum())
    n_hotspots_mean = int(hot_mean.sum())
    print(f"\nResidues compared: {n_residues}")
    print(f"Mutations used: {n_mutations_used}")
    print(f"Excluded (mismatch): {n_excluded}")
    print(f"Hot spots (ΔΔG >= {HOTSPOT_THRESHOLD}, max agg): {n_hotspots_max}")
    print(f"Hot spots (ΔΔG >= {HOTSPOT_THRESHOLD}, mean agg): {n_hotspots_mean}")

    # --- Confound panel ---
    try:
        complex_info = get_complex(pdb_id)
        if chain in complex_info.partner1:
            partner_chains = complex_info.partner2
        elif chain in complex_info.partner2:
            partner_chains = complex_info.partner1
        else:
            partner_chains = ()
    except KeyError:
        partner_chains = ()
        log.warning("Complex %s not registered; distance_to_partner unavailable", pdb_id)

    confounds = compute_confounds(
        pdb_path, chain, chain_ids, chain_seq, partner_chains, positions,
    )

    confound_corrs: dict[str, dict] = {}
    print("\n--- Confound panel ---")
    for name in sorted(confounds):
        vals = confounds[name]
        rho_ddg = spearman(vals, ddg_abs_max)
        rho_grad = spearman(vals, grad_scores)
        _, ci_ddg_lo, ci_ddg_hi = bootstrap_ci(spearman, vals, ddg_abs_max)
        _, ci_grad_lo, ci_grad_hi = bootstrap_ci(spearman, vals, grad_scores)
        confound_corrs[name] = {
            "rho_vs_abs_ddg": rho_ddg,
            "rho_vs_abs_ddg_ci": [ci_ddg_lo, ci_ddg_hi],
            "rho_vs_grad": rho_grad,
            "rho_vs_grad_ci": [ci_grad_lo, ci_grad_hi],
        }
        print(
            f"  {name:25s}  vs |ddG|: {rho_ddg:+.4f} [{ci_ddg_lo:+.4f},{ci_ddg_hi:+.4f}]"
            f"   vs grad: {rho_grad:+.4f} [{ci_grad_lo:+.4f},{ci_grad_hi:+.4f}]"
        )

    # --- Partial correlation ---
    confound_names_sorted = sorted(confounds)
    confound_matrix = np.column_stack(
        [confounds[k] for k in confound_names_sorted]
    )
    partial_rho_abs_max = partial_spearman(grad_scores, ddg_abs_max, confound_matrix)
    partial_rho_abs_mean = partial_spearman(grad_scores, ddg_abs_mean, confound_matrix)
    print("\nPartial Spearman (grad vs |ddG|, controlling all confounds):")
    print(f"  max agg:  simple {rho_abs_max:+.4f}  partial {partial_rho_abs_max:+.4f}")
    print(f"  mean agg: simple {rho_abs_mean:+.4f}  partial {partial_rho_abs_mean:+.4f}")

    # --- AUROC / AUPRC for hot-spot classification ---
    auroc_max = auroc(grad_scores, hot_max)
    auroc_mean_val = auroc(grad_scores, hot_mean)
    auprc_max = auprc(grad_scores, hot_max)
    auprc_mean_val = auprc(grad_scores, hot_mean)
    print(f"\nHot-spot classification (ddG >= {HOTSPOT_THRESHOLD}):")
    print(f"  AUROC  max agg: {auroc_max:.4f}   mean agg: {auroc_mean_val:.4f}")
    print(f"  AUPRC  max agg: {auprc_max:.4f}   mean agg: {auprc_mean_val:.4f}")

    # --- Shuffled-ranking null ---
    rng = np.random.default_rng(42)
    n_shuffle = 1000
    null_rhos = np.array([
        spearman(rng.permutation(grad_scores), ddg_abs_max)
        for _ in range(n_shuffle)
    ])
    null_aurocs = np.array([
        auroc(rng.permutation(grad_scores), hot_max)
        for _ in range(n_shuffle)
    ])
    null_auprcs = np.array([
        auprc(rng.permutation(grad_scores), hot_max)
        for _ in range(n_shuffle)
    ])
    print(f"\nShuffled-ranking null (n={n_shuffle}):")
    print(f"  Spearman  mean {np.mean(null_rhos):+.4f}  p95 {np.percentile(null_rhos, 95):+.4f}")
    print(f"  AUROC     mean {np.nanmean(null_aurocs):.4f}  p95 {np.nanpercentile(null_aurocs, 95):.4f}")
    print(f"  AUPRC     mean {np.nanmean(null_auprcs):.4f}  p95 {np.nanpercentile(null_auprcs, 95):.4f}")

    # --- Bootstrap confidence intervals ---
    _, rho_ci_lo, rho_ci_hi = bootstrap_ci(spearman, grad_scores, ddg_abs_max)
    _, partial_ci_lo, partial_ci_hi = bootstrap_ci(
        partial_spearman, grad_scores, ddg_abs_max, confound_matrix,
    )
    _, auroc_ci_lo, auroc_ci_hi = bootstrap_ci(auroc, grad_scores, hot_max)
    _, auprc_ci_lo, auprc_ci_hi = bootstrap_ci(auprc, grad_scores, hot_max)
    print("\n95% Bootstrap CIs:")
    print(f"  Spearman |ddG|: [{rho_ci_lo:+.4f}, {rho_ci_hi:+.4f}]")
    print(f"  Partial:        [{partial_ci_lo:+.4f}, {partial_ci_hi:+.4f}]")
    print(f"  AUROC:          [{auroc_ci_lo:.4f}, {auroc_ci_hi:.4f}]")
    print(f"  AUPRC:          [{auprc_ci_lo:.4f}, {auprc_ci_hi:.4f}]")

    # --- Write JSON artifact ---
    result = {
        "complex": pdb_id,
        "chain": chain,
        "n_residues_compared": n_residues,
        "n_mutations_used": n_mutations_used,
        "n_excluded_mismatch": n_excluded,
        "hotspot_threshold_kcal_mol": HOTSPOT_THRESHOLD,
        "n_hotspots_max_agg": n_hotspots_max,
        "n_hotspots_mean_agg": n_hotspots_mean,
        "spearman_abs_ddg_max_agg": rho_abs_max,
        "spearman_abs_ddg_mean_agg": rho_abs_mean,
        "spearman_signed_ddg_max_agg": rho_signed_max,
        "spearman_signed_ddg_mean_agg": rho_signed_mean,
        **prec_results,
        "confound_correlations": confound_corrs,
        "partial_spearman_abs_ddg_max_agg": partial_rho_abs_max,
        "partial_spearman_abs_ddg_mean_agg": partial_rho_abs_mean,
        "auroc_max_agg": auroc_max,
        "auroc_mean_agg": auroc_mean_val,
        "auprc_max_agg": auprc_max,
        "auprc_mean_agg": auprc_mean_val,
        "shuffled_null": {
            "n_permutations": n_shuffle,
            "spearman_mean": float(np.mean(null_rhos)),
            "spearman_p95": float(np.percentile(null_rhos, 95)),
            "auroc_mean": float(np.nanmean(null_aurocs)),
            "auroc_p95": float(np.nanpercentile(null_aurocs, 95)),
            "auprc_mean": float(np.nanmean(null_auprcs)),
            "auprc_p95": float(np.nanpercentile(null_auprcs, 95)),
        },
        "bootstrap_ci_95": {
            "spearman_abs_ddg_max": [rho_ci_lo, rho_ci_hi],
            "partial_spearman_abs_ddg_max": [partial_ci_lo, partial_ci_hi],
            "auroc_max": [auroc_ci_lo, auroc_ci_hi],
            "auprc_max": [auprc_ci_lo, auprc_ci_hi],
        },
        "per_position": [
            {
                "residue_id": chain_ids[p],
                "seq_index": p,
                "grad_norm": float(norms[p]),
                "ddg_max": float(ddg_max[i]),
                "ddg_mean": float(ddg_mean[i]),
                "n_mutations": n_mutations_per_pos[i],
                **{k: float(confounds[k][i]) for k in confounds},
            }
            for i, p in enumerate(positions)
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path}")

    prov_write(
        out_path,
        stage="10_skempi_hotspots",
        inputs={"grad": str(grad_path), "pdb": str(pdb_path), "skempi": "skempi_v2.csv"},
        params={
            "complex": pdb_id,
            "chain": chain,
            "hotspot_threshold": HOTSPOT_THRESHOLD,
            "allow_mismatch": args.allow_mismatch,
        },
        arm={"stage": "skempi_hotspots", "complex": pdb_id, "chain": chain},
    )


if __name__ == "__main__":
    main()
