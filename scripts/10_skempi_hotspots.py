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

from igv.metrics import precision_at_k, spearman
from igv.provenance import write as prov_write
from igv.skempi import (
    add_ddg,
    filter_complex,
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

    mutations = [parse_mutation(m.strip()) for m in df["Mutation(s)_cleaned"]]
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

    # precision@k for hot-spot recovery (ΔΔG >= 2.0)
    hot_max = (ddg_max >= HOTSPOT_THRESHOLD).astype(float)
    hot_mean = (ddg_mean >= HOTSPOT_THRESHOLD).astype(float)

    prec_results = {}
    for k in (5, 10, 20):
        k_eff = min(k, n_residues)
        if k_eff == 0:
            continue
        p_max = precision_at_k(grad_scores, ddg_max, k_eff)
        p_mean = precision_at_k(grad_scores, ddg_mean, k_eff)
        prec_results[f"precision_at_{k}_max"] = p_max
        prec_results[f"precision_at_{k}_mean"] = p_mean
        print(f"Precision@{k} (max agg):  {p_max:.4f}")
        print(f"Precision@{k} (mean agg): {p_mean:.4f}")

    n_hotspots_max = int(hot_max.sum())
    n_hotspots_mean = int(hot_mean.sum())
    print(f"\nResidues compared: {n_residues}")
    print(f"Mutations used: {n_mutations_used}")
    print(f"Excluded (mismatch): {n_excluded}")
    print(f"Hot spots (ΔΔG >= {HOTSPOT_THRESHOLD}, max agg): {n_hotspots_max}")
    print(f"Hot spots (ΔΔG >= {HOTSPOT_THRESHOLD}, mean agg): {n_hotspots_mean}")

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
        "per_position": [
            {
                "residue_id": chain_ids[p],
                "seq_index": p,
                "grad_norm": float(norms[p]),
                "ddg_max": float(ddg_max[i]),
                "ddg_mean": float(ddg_mean[i]),
                "n_mutations": n_mutations_per_pos[i],
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
