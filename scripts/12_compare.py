#!/usr/bin/env python3
"""Stage 12 -- compare attribution predictions against brute-force scan and experiment.

Reports three comparisons that together distinguish "the attribution method
fails" from "Boltz-2 has no binding signal on this complex":

    attribution vs scan       -- is the gradient a faithful proxy for the model?
    attribution vs experiment -- does the gradient predict measured binding?
    scan vs experiment        -- does the model itself predict measured binding?

Sign conventions
----------------
SKEMPI DDG is positive for *weaker* binding (destabilising).
Starr ``bind_avg`` is positive for *tighter* binding; ``igv.dms.binding_ddg``
negates it onto SKEMPI's convention.  All experimental values used here are
on the SKEMPI convention (positive = destabilising).

Score orientation: ``complex_pde`` is lower-is-better (predicted distance
error); the other scores (``complex_iplddt``, ``iptm``, etc.) are
higher-is-better.  When correlating model scores against experiment (where
positive DDG = destabilising = worse binding), we negate higher-is-better
model scores so that positive Spearman always means "model agrees with
experiment."
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.metrics import (  # noqa: E402
    aggregate_within_position,
    bootstrap_ci,
    spearman,
    within_position_spearman,
)
from igv.provenance import read as prov_read, write as prov_write  # noqa: E402

log = logging.getLogger("compare")

# -- Score orientation --------------------------------------------------------
# Scores where a LOWER value means better binding quality.
# All others are higher-is-better.
LOWER_IS_BETTER = {"complex_pde"}


def _orient_for_experiment(values: np.ndarray, score: str) -> np.ndarray:
    """Negate higher-is-better scores so that increase = destabilising = positive DDG."""
    if score in LOWER_IS_BETTER:
        return values
    return -values


# -- Helpers ------------------------------------------------------------------

def _load_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("substitutions", "model_score"):
        if col not in df.columns:
            sys.exit(
                f"ERROR: scan CSV missing column {col!r}. "
                f"Expected schema from 04_scan: {['row_index', 'sequence', 'substitutions', 'n_mut', 'binding_score', 'model_score']}. "
                f"Got: {list(df.columns)}"
            )
    return df


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("position", "mut_aa", "score_delta"):
        if col not in df.columns:
            sys.exit(
                f"ERROR: prediction CSV missing column {col!r}. "
                f"Expected schema from 05_predict (per-substitution path): "
                f"position, mut_aa, score_delta. Got: {list(df.columns)}"
            )
    return df


def _scan_key(row: pd.Series) -> str:
    return str(row["substitutions"]).strip()


def _pred_key(row: pd.Series) -> str:
    return f"{int(row['position'])}{row['mut_aa']}"


def _merge_scan_pred(
    scan_df: pd.DataFrame, pred_df: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    """Inner-join scan and pred on substitution key.

    Returns (merged_df, n_pred_only, n_scan_only).
    """
    scan_keyed = scan_df.copy()
    scan_keyed["_key"] = scan_keyed.apply(_scan_key, axis=1)

    pred_keyed = pred_df.copy()
    pred_keyed["_key"] = pred_keyed.apply(_pred_key, axis=1)

    scan_keys = set(scan_keyed["_key"])
    pred_keys = set(pred_keyed["_key"])

    merged = scan_keyed.merge(pred_keyed, on="_key", suffixes=("_scan", "_pred"))

    n_scan_only = len(scan_keys - pred_keys)
    n_pred_only = len(pred_keys - scan_keys)

    return merged, n_pred_only, n_scan_only


def _load_experiment_skempi(
    dataset: str, chain: str, cache_dir: Path
) -> pd.DataFrame | None:
    """Load per-substitution SKEMPI DDG for a complex.

    Returns a DataFrame with columns [position, mut_aa, ddg] where position
    is 0-based and ddg is on SKEMPI convention (positive = destabilising).
    Returns None if the dataset is not a SKEMPI complex.
    """
    from igv.skempi import (
        SKEMPI_MUTATION_COL,
        add_ddg,
        filter_complex,
        get_complex,
        load_skempi,
        map_mutations_to_indices,
        parse_mutation,
        read_pdb_residue_ids,
        single_point,
    )
    from igv.data import download_rcsb

    try:
        cx = get_complex(dataset)
    except KeyError:
        return None

    pdb_path = download_rcsb(cx.pdb_id, cache_dir)
    residue_ids, sequences = read_pdb_residue_ids(pdb_path)
    chain_ids = residue_ids[chain]
    chain_seq = sequences[chain]

    df = load_skempi(cache_dir)
    df = filter_complex(df, cx.pdb_id)
    df = single_point(df)
    df = add_ddg(df)

    mutations = [parse_mutation(m.strip()) for m in df[SKEMPI_MUTATION_COL]]
    chain_mutations = [(mut, i) for i, mut in enumerate(mutations) if mut.chain == chain]
    if not chain_mutations:
        return None

    chain_muts = [m for m, _ in chain_mutations]
    chain_ddgs = [float(df.iloc[i]["ddg_kcal_mol"]) for _, i in chain_mutations]

    mapped, _mismatches = map_mutations_to_indices(
        chain_muts, chain_ids, chain_seq, allow_mismatch=True,
    )

    rows = []
    for mut, seq_idx in mapped:
        orig_pos = chain_muts.index(mut)
        rows.append({
            "position": seq_idx,
            "mut_aa": mut.mut_aa,
            "ddg": chain_ddgs[orig_pos],
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def _load_experiment_dms(
    dataset: str, cache_dir: Path
) -> pd.DataFrame | None:
    """Load per-substitution DMS DDG for a complex.

    Returns a DataFrame with columns [position, mut_aa, ddg] where position
    is 0-based and ddg is on SKEMPI convention (positive = destabilising).
    Returns None if the dataset is not a DMS complex.
    """
    from igv.dms import (
        binding_ddg,
        get_complex,
        load_starr2020,
        map_sites_to_indices,
        singles,
    )
    from igv.skempi import read_pdb_residue_ids
    from igv.data import download_rcsb

    try:
        cx = get_complex(dataset)
    except KeyError:
        return None

    pdb_path = download_rcsb(cx.pdb_id, cache_dir)
    residue_ids, sequences = read_pdb_residue_ids(pdb_path)
    chain = cx.mutated_chain
    chain_ids = residue_ids[chain]
    chain_seq = sequences[chain]

    raw_df = load_starr2020(cache_dir)
    dms_df = singles(raw_df)
    site_map, _mismatches = map_sites_to_indices(
        dms_df, chain_ids, chain_seq, allow_mismatch=True,
    )

    rows = []
    for _, row in dms_df.iterrows():
        site = int(row["site_SARS2"])
        idx = site_map.get(site)
        if idx is None:
            continue
        ddg_val = float(binding_ddg(row["bind_avg"]))
        rows.append({
            "position": idx,
            "mut_aa": row["mutant"],
            "ddg": ddg_val,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def _get_backward_passes(pred_path: Path) -> int | None:
    """Read the number of backward passes from the pred's provenance arm."""
    try:
        prov = prov_read(pred_path)
        arm = prov.get("arm", {})
        m_steps = arm.get("m_steps")
        if m_steps is not None:
            return int(m_steps)
    except FileNotFoundError:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare attribution predictions against scan and experiment",
    )
    parser.add_argument("--pred", required=True, help="Prediction CSV from 05_predict (per-substitution)")
    parser.add_argument("--scan", required=True, help="Scan CSV from 04_scan")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. spike_rbd, 1JTG)")
    parser.add_argument("--score", default="complex_pde", help="Score name used")
    parser.add_argument("--chain", default=None, help="Chain for SKEMPI lookups")
    parser.add_argument("--cache-dir", default="data/raw", help="Cache directory")
    parser.add_argument("--interface-cutoff", type=float, default=5.0, help="Distance cutoff for interface (A)")
    parser.add_argument("--out", default="results/{dataset}_compare.csv", help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pred_path = Path(args.pred)
    scan_path = Path(args.scan)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out.format(dataset=args.dataset))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        sys.exit(f"ERROR: prediction file not found: {pred_path}")
    if not scan_path.exists():
        sys.exit(f"ERROR: scan file not found: {scan_path}")

    scan_df = _load_scan(scan_path)
    pred_df = _load_pred(pred_path)

    rows: list[dict] = []

    # -- 1. Headline: attribution vs scan (T1) --------------------------------
    merged, n_pred_only, n_scan_only = _merge_scan_pred(scan_df, pred_df)
    n_overlap = len(merged)

    if n_pred_only > 0 or n_scan_only > 0:
        log.warning(
            "OVERLAP ASYMMETRY: %d mutants in pred only, %d in scan only, "
            "%d in both. The two stages may have been run with different "
            "--positions. Results are computed over the %d-mutant intersection.",
            n_pred_only, n_scan_only, n_overlap, n_overlap,
        )
        print(
            f"\n*** WARNING: overlap asymmetry ***\n"
            f"  pred-only:  {n_pred_only}\n"
            f"  scan-only:  {n_scan_only}\n"
            f"  overlap:    {n_overlap}\n"
        )

    # The scan has no wild-type row on the DMS/SKEMPI paths (only mutants
    # where aa != ref_aa are scored).  Spearman is rank-based and invariant
    # to additive shifts, so only rank statistics are available -- not
    # calibrated deltas -- without a wild-type reference.
    has_wt_row = (scan_df["substitutions"].astype(str).str.strip() == "").any()
    if has_wt_row:
        log.info("Wild-type row found in scan; excluding from correlation.")
        merged = merged[merged["_key"] != ""]
        n_overlap = len(merged)

    if n_overlap < 2:
        sys.exit(f"ERROR: only {n_overlap} overlapping mutants between scan and pred.")

    rho_t1 = spearman(merged["score_delta"].values, merged["model_score"].values)
    _, ci_lo_t1, ci_hi_t1 = bootstrap_ci(
        spearman, merged["score_delta"].values, merged["model_score"].values,
    )

    print("\n=== Attribution vs Scan (T1) ===")
    print(f"  n = {n_overlap}")
    print(f"  Spearman = {rho_t1:+.4f}  95% CI [{ci_lo_t1:+.4f}, {ci_hi_t1:+.4f}]")
    if not has_wt_row:
        print("  (no wild-type row in scan; only rank statistics available)")

    rows.append({"comparison": "attribution_vs_scan", "scope": "all", "statistic": "spearman", "value": rho_t1})
    rows.append({"comparison": "attribution_vs_scan", "scope": "all", "statistic": "n", "value": n_overlap})
    rows.append({"comparison": "attribution_vs_scan", "scope": "all", "statistic": "ci_lo", "value": ci_lo_t1})
    rows.append({"comparison": "attribution_vs_scan", "scope": "all", "statistic": "ci_hi", "value": ci_hi_t1})
    rows.append({"comparison": "attribution_vs_scan", "scope": "all", "statistic": "n_pred_only", "value": n_pred_only})
    rows.append({"comparison": "attribution_vs_scan", "scope": "all", "statistic": "n_scan_only", "value": n_scan_only})

    # -- 2. Both sides against experiment -------------------------------------
    exp_df: pd.DataFrame | None = None

    # Try SKEMPI
    if args.chain:
        exp_df = _load_experiment_skempi(args.dataset, args.chain, cache_dir)
        if exp_df is not None:
            log.info("Loaded SKEMPI experimental data for %s chain %s", args.dataset, args.chain)

    # Try DMS
    if exp_df is None:
        exp_df = _load_experiment_dms(args.dataset, cache_dir)
        if exp_df is not None:
            log.info("Loaded DMS experimental data for %s", args.dataset)

    data_source: str | None = None
    if exp_df is not None:
        assert exp_df["ddg"].notna().all(), (
            "Experimental DDG column contains nulls after loading"
        )

        # Determine data source
        try:
            from igv.dms import get_complex as dms_get_complex
            dms_get_complex(args.dataset)
            data_source = "dms"
        except KeyError:
            data_source = "skempi"

        # Join experiment with pred on (position, mut_aa)
        exp_pred = pred_df.merge(
            exp_df, on=["position", "mut_aa"], how="inner",
        )
        # Join experiment with scan
        scan_with_pos = scan_df.copy()
        scan_with_pos["_pos"] = scan_with_pos["substitutions"].apply(
            lambda s: int(str(s).strip()[:-1]) if str(s).strip() else -1
        )
        scan_with_pos["_aa"] = scan_with_pos["substitutions"].apply(
            lambda s: str(s).strip()[-1] if str(s).strip() else ""
        )
        exp_scan = scan_with_pos.merge(
            exp_df, left_on=["_pos", "_aa"], right_on=["position", "mut_aa"], how="inner",
        )

        # Orient model scores for correlation with DDG
        # (positive DDG = destabilising = worse binding)
        pred_oriented = _orient_for_experiment(exp_pred["score_delta"].values, args.score)
        scan_oriented = _orient_for_experiment(exp_scan["model_score"].values, args.score)

        rho_pred_exp = spearman(pred_oriented, exp_pred["ddg"].values)
        rho_scan_exp = spearman(scan_oriented, exp_scan["ddg"].values)

        _, ci_lo_pe, ci_hi_pe = bootstrap_ci(spearman, pred_oriented, exp_pred["ddg"].values)
        _, ci_lo_se, ci_hi_se = bootstrap_ci(spearman, scan_oriented, exp_scan["ddg"].values)

        print("\n=== Attribution vs Experiment ===")
        print(f"  n = {len(exp_pred)}")
        print(f"  Spearman = {rho_pred_exp:+.4f}  95% CI [{ci_lo_pe:+.4f}, {ci_hi_pe:+.4f}]")

        print("\n=== Scan vs Experiment ===")
        print(f"  n = {len(exp_scan)}")
        print(f"  Spearman = {rho_scan_exp:+.4f}  95% CI [{ci_lo_se:+.4f}, {ci_hi_se:+.4f}]")

        print("\n=== The Triple ===")
        print(f"  attribution-vs-experiment: {rho_pred_exp:+.4f}")
        print(f"  scan-vs-experiment:        {rho_scan_exp:+.4f}")
        print(f"  attribution-vs-scan:       {rho_t1:+.4f}")

        for comp, rho, n, ci_l, ci_h in [
            ("attribution_vs_experiment", rho_pred_exp, len(exp_pred), ci_lo_pe, ci_hi_pe),
            ("scan_vs_experiment", rho_scan_exp, len(exp_scan), ci_lo_se, ci_hi_se),
        ]:
            rows.append({"comparison": comp, "scope": "all", "statistic": "spearman", "value": rho})
            rows.append({"comparison": comp, "scope": "all", "statistic": "n", "value": n})
            rows.append({"comparison": comp, "scope": "all", "statistic": "ci_lo", "value": ci_l})
            rows.append({"comparison": comp, "scope": "all", "statistic": "ci_hi", "value": ci_h})
    else:
        print("\n(no experimental data available for this dataset)")

    # -- 3. Within-position analysis for DMS datasets -------------------------
    if data_source == "dms":
        from igv.dms import (
            AA_ORDER,
            get_complex,
            interface_positions,
            load_starr2020,
            map_sites_to_indices,
            singles,
            substitution_matrix,
        )
        from igv.skempi import read_pdb_residue_ids
        from igv.data import download_rcsb

        cx = get_complex(args.dataset)
        pdb_path = download_rcsb(cx.pdb_id, cache_dir)
        residue_ids, sequences = read_pdb_residue_ids(pdb_path)
        chain = cx.mutated_chain
        chain_ids = residue_ids[chain]
        chain_seq = sequences[chain]

        iface_indices = interface_positions(
            pdb_path,
            chain=chain,
            partner_chains=cx.partner_chains,
            residue_ids=chain_ids,
            cutoff=args.interface_cutoff,
        )
        interface_rids = sorted(int(chain_ids[i]) for i in iface_indices)
        interface_set = set(interface_rids)

        raw_dms = load_starr2020(cache_dir)
        dms_df = singles(raw_dms)
        site_map, _ = map_sites_to_indices(
            dms_df, chain_ids, chain_seq, allow_mismatch=True,
        )
        idx_to_rid = {idx: int(rid) for rid, idx in site_map.items()}

        bind_mat, bind_positions, _ = substitution_matrix(dms_df, value_col="bind_avg")
        # Negate bind_avg so positive = destabilising (SKEMPI convention).
        # Spearman is rank-based so the linear scaling from binding_ddg() is irrelevant.
        ddg_mat = -bind_mat

        # Build pred matrix aligned with bind_positions
        aa_to_col = {aa: i for i, aa in enumerate(AA_ORDER)}
        pos_to_row = {p: i for i, p in enumerate(bind_positions)}
        pred_mat = np.full((len(bind_positions), 20), np.nan)
        for _, row in pred_df.iterrows():
            pos0 = int(row["position"])
            site = idx_to_rid.get(pos0)
            if site is None:
                continue
            r = pos_to_row.get(site)
            c = aa_to_col.get(row["mut_aa"])
            if r is not None and c is not None and pd.notna(row["score_delta"]):
                pred_mat[r, c] = row["score_delta"]

        # Build scan matrix aligned with bind_positions
        scan_mat = np.full((len(bind_positions), 20), np.nan)
        for _, row in scan_df.iterrows():
            sub = str(row["substitutions"]).strip()
            if not sub:
                continue
            pos0 = int(sub[:-1])
            aa = sub[-1]
            site = idx_to_rid.get(pos0)
            if site is None:
                continue
            r = pos_to_row.get(site)
            c = aa_to_col.get(aa)
            if r is not None and c is not None:
                scan_mat[r, c] = row["model_score"]

        # Orient pred and scan for experiment comparison
        # pred: score_delta oriented for experiment
        pred_mat_oriented = _orient_for_experiment(pred_mat, args.score)
        scan_mat_oriented = _orient_for_experiment(scan_mat, args.score)

        iface_mask = np.array([p in interface_set for p in bind_positions])
        non_iface_mask = ~iface_mask

        wp_comparisons = [
            ("attribution_vs_experiment", pred_mat_oriented, ddg_mat),
            ("scan_vs_experiment", scan_mat_oriented, ddg_mat),
            ("attribution_vs_scan", pred_mat, scan_mat),
        ]

        print("\n=== Within-position analysis (DMS) ===")
        for scope_name, scope_mask in [
            ("interface", iface_mask),
            ("non_interface", non_iface_mask),
            ("all", np.ones(len(bind_positions), dtype=bool)),
        ]:
            for comp_name, mat_a, mat_b in wp_comparisons:
                rhos = within_position_spearman(mat_a[scope_mask], mat_b[scope_mask])
                agg = aggregate_within_position(rhos)

                print(f"\n  {scope_name} / {comp_name}:")
                print(f"    mean={agg['mean']:+.4f}  median={agg['median']:+.4f}  "
                      f"n_usable={agg['n_usable']}  "
                      f"CI=[{agg['ci_lo']:+.4f}, {agg['ci_hi']:+.4f}]")

                for stat_key, stat_val in agg.items():
                    rows.append({
                        "comparison": comp_name,
                        "scope": f"within_position_{scope_name}",
                        "statistic": stat_key,
                        "value": stat_val,
                    })

    # -- 4. Cost per unit of signal -------------------------------------------
    n_forward = len(scan_df)
    n_backward = _get_backward_passes(pred_path)

    print("\n=== Cost ===")
    print(f"  Forward passes (scan):  {n_forward}")
    if n_backward is not None:
        print(f"  Backward passes (attr): {n_backward}")
        ratio = n_forward / n_backward
        print(f"  Cost ratio (fwd/bwd):   {ratio:.1f}x")
    else:
        print("  Backward passes (attr): unknown (no provenance)")
        n_backward_fallback = 1
        ratio = n_forward / n_backward_fallback
        print(f"  Cost ratio (fwd/bwd):   >={ratio:.0f}x (assuming >=1 backward pass)")

    print("\n  Correlations achieved:")
    print(f"    Scan  (attribution vs scan): {rho_t1:+.4f}")
    if exp_df is not None:
        print(f"    Attr  (attr vs experiment):  {rho_pred_exp:+.4f}")
        print(f"    Scan  (scan vs experiment):  {rho_scan_exp:+.4f}")

    rows.append({"comparison": "cost", "scope": "all", "statistic": "n_forward_passes", "value": n_forward})
    rows.append({"comparison": "cost", "scope": "all", "statistic": "n_backward_passes", "value": n_backward if n_backward is not None else float("nan")})
    if n_backward is not None:
        rows.append({"comparison": "cost", "scope": "all", "statistic": "cost_ratio", "value": ratio})

    # -- Write output ---------------------------------------------------------
    out_df = pd.DataFrame(rows)
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
        stage="12_compare",
        inputs={"pred": str(pred_path), "scan": str(scan_path)},
        params={
            "dataset": args.dataset,
            "score": args.score,
            "chain": args.chain,
            "interface_cutoff": args.interface_cutoff,
        },
        arm={"stage": "compare", "dataset": args.dataset, "score": args.score},
    )


if __name__ == "__main__":
    main()
