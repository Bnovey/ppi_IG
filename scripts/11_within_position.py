#!/usr/bin/env python3
"""Within-position Spearman analysis for DMS saturation data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.dms import AA_ORDER, get_complex, load_starr2020, map_sites_to_indices, singles, substitution_matrix
from igv.metrics import aggregate_within_position, spearman, within_position_spearman
from igv.provenance import write as prov_write
from igv.skempi import (
    HYDROPHOBICITY_KD,
    RESIDUE_VOLUME,
    compute_distance_to_partner,
    parse_pdb_heavy_atoms,
    read_pdb_residue_ids,
)

log = logging.getLogger(__name__)

EXPECTED_INTERFACE_POSITIONS = [
    417, 446, 447, 449, 453, 455, 456, 473, 475, 476,
    484, 486, 487, 489, 493, 496, 498, 500, 501, 502, 505,
]


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("position", "mut_aa", "score_delta"):
        if col not in df.columns:
            sys.exit(
                f"ERROR: prediction CSV missing column {col!r}. "
                f"Expected schema from 05_predict (SKEMPI path): "
                f"position, mut_aa, score_delta. Got: {list(df.columns)}"
            )
    return df


def _pivot_to_matrix(
    df: pd.DataFrame,
    positions: list[int],
    value_col: str,
) -> np.ndarray:
    """Pivot a long-form DataFrame to a ``(n_positions, 20)`` matrix."""
    aa_to_col = {aa: i for i, aa in enumerate(AA_ORDER)}
    pos_to_row = {p: i for i, p in enumerate(positions)}

    mat = np.full((len(positions), 20), np.nan)
    for _, row in df.iterrows():
        r = pos_to_row.get(int(row["position"]))
        c = aa_to_col.get(row["mut_aa"])
        if r is not None and c is not None and pd.notna(row[value_col]):
            mat[r, c] = row[value_col]

    return mat


def _property_matrix(
    positions: list[int],
    observed_positions: list[int],
    obs_df: pd.DataFrame,
    prop_dict: dict[str, float],
) -> np.ndarray:
    """Build a ``(n_positions, 20)`` matrix of an amino-acid property.

    Each cell gets the property value of the mutant amino acid, but only
    where the observed data has a measurement (to match the NaN pattern).
    """
    aa_to_col = {aa: i for i, aa in enumerate(AA_ORDER)}
    pos_to_row = {p: i for i, p in enumerate(positions)}
    obs_pos_set = set(observed_positions)

    mat = np.full((len(positions), 20), np.nan)
    for _, row in obs_df.iterrows():
        site = int(row["site_SARS2"])
        if site not in obs_pos_set:
            continue
        r = pos_to_row.get(site)
        c = aa_to_col.get(row["mutant"])
        if r is not None and c is not None and pd.notna(row["bind_avg"]):
            mat[r, c] = prop_dict.get(row["mutant"], np.nan)

    return mat


def _pooled_property_correlation(
    obs_df: pd.DataFrame,
    prop_dict: dict[str, float],
    value_col: str,
    sites: set[int] | None = None,
) -> float:
    """Pooled Spearman between a per-mutant property and an observed readout."""
    df = obs_df.copy()
    if sites is not None:
        df = df[df["site_SARS2"].isin(sites)]
    vals = np.array([prop_dict.get(aa, np.nan) for aa in df["mutant"]])
    return spearman(vals, df[value_col].values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-position Spearman analysis for DMS saturation data",
    )
    parser.add_argument("--pred", required=True, help="Prediction CSV from 05_predict (SKEMPI path)")
    parser.add_argument("--dataset", default="spike_rbd", help="DMS complex key")
    parser.add_argument("--cache-dir", default="data/raw", help="Cache directory")
    parser.add_argument("--interface-cutoff", type=float, default=5.0, help="Distance cutoff for interface (A)")
    parser.add_argument("--out", default="results/within_position.csv", help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pred_path = Path(args.pred)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)

    if not pred_path.exists():
        sys.exit(f"ERROR: prediction file not found: {pred_path}")

    cx = get_complex(args.dataset)
    pdb_path = cache_dir / f"{cx.pdb_id.lower()}.pdb"
    if not pdb_path.exists():
        sys.exit(f"ERROR: PDB file not found: {pdb_path}")

    # --- Load PDB structure ---
    residue_ids, sequences = read_pdb_residue_ids(pdb_path)
    chain = cx.mutated_chain
    chain_ids = residue_ids[chain]
    chain_seq = sequences[chain]

    # --- Compute interface positions ---
    coords, atom_chains, atom_res_keys = parse_pdb_heavy_atoms(pdb_path)
    dists = compute_distance_to_partner(
        coords, atom_chains, atom_res_keys,
        chain, cx.partner_chains, chain_ids,
    )
    interface_rids = sorted(
        int(rid) for rid, d in zip(chain_ids, dists) if d <= args.interface_cutoff
    )
    if args.interface_cutoff == 5.0:
        assert interface_rids == EXPECTED_INTERFACE_POSITIONS, (
            f"Expected {len(EXPECTED_INTERFACE_POSITIONS)} interface positions at "
            f"5.0 A, got {len(interface_rids)}: {interface_rids}"
        )
    interface_set = set(interface_rids)
    log.info("Interface positions (%d at %.1f A): %s", len(interface_rids), args.interface_cutoff, interface_rids)

    # --- Load DMS data ---
    raw_df = load_starr2020(cache_dir)
    dms_df = singles(raw_df)
    site_map, mismatches = map_sites_to_indices(
        dms_df, chain_ids, chain_seq, allow_mismatch=True,
    )
    log.info("DMS sites mapped: %d, mismatches: %d", len(site_map), len(mismatches))

    bind_mat, bind_positions, _ = substitution_matrix(dms_df, value_col="bind_avg")
    expr_mat, _, _ = substitution_matrix(dms_df, value_col="expr_avg")

    # --- Load predictions ---
    pred_df = _load_pred(pred_path)
    pred_positions_raw = sorted(pred_df["position"].unique())

    # Predictions use 0-based indices; DMS uses author residue numbers.
    # Map 0-based prediction positions to author residue numbers.
    pred_site_map: dict[int, int] = {}
    idx_to_rid = {idx: int(rid) for rid, idx in site_map.items()}
    for pos in pred_positions_raw:
        pos_int = int(pos)
        if pos_int in idx_to_rid:
            pred_site_map[pos_int] = idx_to_rid[pos_int]

    # Build prediction matrix aligned with bind_positions
    pred_mat = np.full((len(bind_positions), 20), np.nan)
    aa_to_col = {aa: i for i, aa in enumerate(AA_ORDER)}
    pos_to_row = {p: i for i, p in enumerate(bind_positions)}
    for _, row in pred_df.iterrows():
        pos0 = int(row["position"])
        site = pred_site_map.get(pos0)
        if site is None:
            continue
        r = pos_to_row.get(site)
        c = aa_to_col.get(row["mut_aa"])
        if r is not None and c is not None and pd.notna(row["score_delta"]):
            pred_mat[r, c] = row["score_delta"]

    # --- Build confound property matrices (matching NaN pattern of bind_mat) ---
    volume_mat = _property_matrix(bind_positions, bind_positions, dms_df, RESIDUE_VOLUME)
    hydro_mat = _property_matrix(bind_positions, bind_positions, dms_df, HYDROPHOBICITY_KD)

    # --- Identify interface / non-interface row masks ---
    iface_mask = np.array([p in interface_set for p in bind_positions])
    non_iface_mask = ~iface_mask

    # --- Within-position Spearman ---
    results_rows: list[dict] = []

    comparisons = [
        ("pred_vs_bind", pred_mat, bind_mat),
        ("pred_vs_expr", pred_mat, expr_mat),
        ("bind_vs_volume", bind_mat, volume_mat),
        ("bind_vs_hydrophobicity", bind_mat, hydro_mat),
    ]

    for scope_name, scope_mask in [("interface", iface_mask), ("non_interface", non_iface_mask), ("all", np.ones(len(bind_positions), dtype=bool))]:
        for comp_name, mat_a, mat_b in comparisons:
            rhos = within_position_spearman(mat_a[scope_mask], mat_b[scope_mask])
            agg = aggregate_within_position(rhos)

            print(f"\n{scope_name} / {comp_name}:")
            print(f"  mean={agg['mean']:+.4f}  median={agg['median']:+.4f}  "
                  f"n_usable={agg['n_usable']}  n_neg={agg['n_negative']}  "
                  f"CI=[{agg['ci_lo']:+.4f}, {agg['ci_hi']:+.4f}]")

            for stat_key, stat_val in agg.items():
                results_rows.append({
                    "scope": scope_name,
                    "comparison": comp_name,
                    "analysis": "within_position",
                    "statistic": stat_key,
                    "value": stat_val,
                })

            # Per-position detail
            scope_positions = [p for p, m in zip(bind_positions, scope_mask) if m]
            for pos, rho in zip(scope_positions, rhos):
                results_rows.append({
                    "scope": scope_name,
                    "comparison": comp_name,
                    "analysis": "per_position",
                    "statistic": str(pos),
                    "value": float(rho),
                })

    # --- Pooled correlations for contrast ---
    dms_iface_df = dms_df[dms_df["site_SARS2"].isin(interface_set)]
    pooled_comparisons = [
        ("bind_vs_volume", "bind_avg", RESIDUE_VOLUME),
        ("bind_vs_hydrophobicity", "bind_avg", HYDROPHOBICITY_KD),
    ]
    for comp_name, value_col, prop_dict in pooled_comparisons:
        for scope_name, sub_df in [("all", dms_df), ("interface", dms_iface_df), ("non_interface", dms_df[~dms_df["site_SARS2"].isin(interface_set)])]:
            r = _pooled_property_correlation(sub_df, prop_dict, value_col)
            print(f"\nPooled {scope_name} / {comp_name}: {r:+.4f}")
            results_rows.append({
                "scope": scope_name,
                "comparison": comp_name,
                "analysis": "pooled",
                "statistic": "spearman",
                "value": r,
            })

    # Pooled bind vs expr
    for scope_name, sub_df in [("all", dms_df), ("interface", dms_iface_df), ("non_interface", dms_df[~dms_df["site_SARS2"].isin(interface_set)])]:
        mask = sub_df["bind_avg"].notna() & sub_df["expr_avg"].notna()
        r = spearman(sub_df.loc[mask, "bind_avg"].values, sub_df.loc[mask, "expr_avg"].values)
        print(f"\nPooled {scope_name} / bind_vs_expr: {r:+.4f}")
        results_rows.append({
            "scope": scope_name,
            "comparison": "bind_vs_expr",
            "analysis": "pooled",
            "statistic": "spearman",
            "value": r,
        })

    # Pooled pred vs bind/expr
    pred_long = pred_df.copy()
    pred_long["site"] = pred_long["position"].map(pred_site_map)
    pred_long = pred_long.dropna(subset=["site"])
    pred_long["site"] = pred_long["site"].astype(int)
    merged = pred_long.merge(
        dms_df[["site_SARS2", "mutant", "bind_avg", "expr_avg"]],
        left_on=["site", "mut_aa"],
        right_on=["site_SARS2", "mutant"],
        how="inner",
    )
    for scope_name in ["all", "interface", "non_interface"]:
        if scope_name == "interface":
            sub = merged[merged["site_SARS2"].isin(interface_set)]
        elif scope_name == "non_interface":
            sub = merged[~merged["site_SARS2"].isin(interface_set)]
        else:
            sub = merged
        for target_col, comp_name in [("bind_avg", "pred_vs_bind"), ("expr_avg", "pred_vs_expr")]:
            mask = sub[target_col].notna()
            r = spearman(sub.loc[mask, "score_delta"].values, sub.loc[mask, target_col].values)
            print(f"\nPooled {scope_name} / {comp_name}: {r:+.4f}")
            results_rows.append({
                "scope": scope_name,
                "comparison": comp_name,
                "analysis": "pooled",
                "statistic": "spearman",
                "value": r,
            })

    # --- Write output ---
    out_df = pd.DataFrame(results_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, out_df], ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"\nAppended {len(out_df)} rows to {out_path} (total {len(combined)})")
    else:
        out_df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path} ({len(out_df)} rows)")

    prov_write(
        out_path,
        stage="11_within_position",
        inputs={"pred": str(pred_path), "dms": "single_mut_effects.csv"},
        params={
            "dataset": args.dataset,
            "interface_cutoff": args.interface_cutoff,
        },
        arm={"stage": "within_position", "dataset": args.dataset},
    )


if __name__ == "__main__":
    main()
