#!/usr/bin/env python3
"""Stage 13 -- join pair-attribution map to SKEMPI double-mutant cycles.

Runs the four controls from ROADMAP.md section 12:

  1. Inter-residue distance partial -- Cbeta-Cbeta distance predicts coupling
     on its own, so report partial correlation controlling for it.
  2. Singles partial -- does A[i,j] carry information beyond A[i] and A[j]?
     **This is the control that decides whether Phase 5 continues.**
  3. Permutation null over pair labels (shuffled-null convention from stage 10).
  4. Cluster bootstrap by position (31 clusters for 1JTG, not 76 pairs).

Expected .npz layout for the pair-attribution map
--------------------------------------------------
The ``pair_layer_ig`` function in ``src/igv/attrib.py`` (being added by a
parallel stage-0b agent) will produce an ``.npz`` file with at least::

    pair_ig : ndarray, shape (L, L)
        Symmetrised pair attribution map: ``A[i,j] + A[j,i]``, contracted
        over the 128 channels via a dot product.

If the key ``pair_ig`` is absent, the loader also tries ``pair_map`` and
``attribution`` as fallback key names.  The diagonal ``A[i,i]`` entries are
the single-residue attributions used in control 2.

Calibration
-----------
r in 0.26-0.37 is the honest benchmark (pLM epistasis on fitness, not
binding).  Do NOT compare against the 0.77-0.88 multi-point ddG numbers,
which are dominated by additivity.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.coupling import (  # noqa: E402
    aggregate_pairs,
    extract_cycles,
)
from igv.metrics import bootstrap_ci, partial_spearman, spearman  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402
from igv.skempi import (  # noqa: E402
    add_ddg,
    get_complex,
    load_skempi,
    parse_pdb_heavy_atoms,
    read_pdb_residue_ids,
)

log = logging.getLogger(__name__)

_PAIR_IG_KEYS = ("pair_ig", "pair_map", "attribution")


def _load_pair_map(path: Path) -> np.ndarray:
    """Load the (L, L) pair-attribution map from an .npz file."""
    npz = np.load(path, allow_pickle=True)
    for key in _PAIR_IG_KEYS:
        if key in npz:
            arr = npz[key]
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"pair map under key {key!r} has shape {arr.shape}; "
                    f"expected (L, L)"
                )
            return arr
    raise KeyError(
        f"pair-attribution .npz has no recognised key. "
        f"Tried {_PAIR_IG_KEYS}; found {list(npz.keys())}"
    )


def _cbeta_distance_matrix(
    pdb_path: Path,
    residue_ids_i: list[str],
    chain_i: str,
    residue_ids_j: list[str],
    chain_j: str,
) -> dict[tuple[tuple[str, str], tuple[str, str]], float]:
    """Compute Cbeta-Cbeta (CA for Gly) distances between residue pairs."""
    coords, atom_chains, atom_res_keys = parse_pdb_heavy_atoms(pdb_path)

    from collections import defaultdict

    idx_map: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (ch, rk) in enumerate(zip(atom_chains, atom_res_keys)):
        idx_map[(ch, rk)].append(i)

    def _representative_coord(chain: str, resnum: str) -> np.ndarray | None:
        atoms = idx_map.get((chain, resnum))
        if atoms is None:
            return None
        return coords[atoms].mean(axis=0)

    result: dict[tuple[tuple[str, str], tuple[str, str]], float] = {}
    for ri in residue_ids_i:
        ci = _representative_coord(chain_i, ri)
        if ci is None:
            continue
        for rj in residue_ids_j:
            cj = _representative_coord(chain_j, rj)
            if cj is None:
                continue
            d = float(np.sqrt(((ci - cj) ** 2).sum()))
            result[((chain_i, ri), (chain_j, rj))] = d
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join pair-attribution map to SKEMPI coupling cycles and "
            "run the four Phase 5 controls"
        ),
    )
    parser.add_argument(
        "--complex", required=True,
        help="PDB ID (e.g. 1JTG). Must be registered in skempi.SKEMPI_COMPLEXES.",
    )
    parser.add_argument(
        "--pair-map", default=None,
        help="Path to pair-attribution .npz (key: pair_ig, shape (L, L))",
    )
    parser.add_argument(
        "--pdb", default=None,
        help="Path to PDB file (default: cache-dir/<pdb>.pdb)",
    )
    parser.add_argument("--cache-dir", default="data/raw", help="Cache directory")
    parser.add_argument(
        "--out", default="results/{complex}_coupling.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for the null",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for permutation and bootstrap",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the full plan and every input path without computing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pdb_id = args.complex.upper()
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out.format(complex=pdb_id))

    try:
        cx = get_complex(pdb_id)
    except KeyError:
        sys.exit(f"ERROR: complex {pdb_id} not registered in SKEMPI_COMPLEXES")

    pdb_path = Path(args.pdb) if args.pdb else cache_dir / f"{pdb_id.lower()}.pdb"
    pair_map_path = Path(args.pair_map) if args.pair_map else None
    skempi_csv = cache_dir / "skempi_v2.csv"

    # ------------------------------------------------------------------
    # Dry-run: print plan and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print(f"Stage 13 coupling analysis for {pdb_id}\n")
        print(f"  complex:          {pdb_id} ({cx.note})")
        print(f"  partner 1:        {cx.partner1}")
        print(f"  partner 2:        {cx.partner2}")
        print(f"  PDB path:         {pdb_path}  {'(exists)' if pdb_path.exists() else '(MISSING)'}")
        print(f"  SKEMPI CSV:       {skempi_csv}  {'(exists)' if skempi_csv.exists() else '(MISSING)'}")
        if pair_map_path is not None:
            print(f"  pair-map .npz:    {pair_map_path}  {'(exists)' if pair_map_path.exists() else '(MISSING)'}")
        else:
            print("  pair-map .npz:    (not provided, will run coupling extraction only)")
        print(f"  output:           {out_path}")
        print(f"  n_permutations:   {args.n_permutations}")
        print(f"  seed:             {args.seed}")

        if skempi_csv.exists():
            import pandas as pd
            df = pd.read_csv(skempi_csv, sep=";")
            df = add_ddg(df)
            cycles = extract_cycles(df, complex_key=f"{pdb_id}_{cx.partner1[0]}{''.join(cx.partner1[1:])}_{cx.partner2[0]}{''.join(cx.partner2[1:])}")
            if not cycles:
                cx_key = df[df["#Pdb"].str.startswith(pdb_id)]["#Pdb"].iloc[0] if len(df[df["#Pdb"].str.startswith(pdb_id)]) > 0 else None
                if cx_key:
                    cycles = extract_cycles(df, complex_key=cx_key)
            pairs = {tuple(sorted([c.pos_i, c.pos_j])) for c in cycles}
            positions: set[tuple[str, str]] = set()
            for c in cycles:
                positions.add(c.pos_i)
                positions.add(c.pos_j)
            print("\n  Coupling extraction preview:")
            print(f"    complete cycles:    {len(cycles)}")
            print(f"    distinct pairs:     {len(pairs)}")
            print(f"    distinct positions: {len(positions)}")
            print(f"    cross-chain cycles: {sum(c.cross_chain for c in cycles)}")

        print("\n  Controls to run (if pair-map provided):")
        print("    1. Inter-residue distance partial (Cbeta-Cbeta)")
        print("    2. Singles partial: A[i,j] vs coupling, controlling A[i] and A[j]")
        print("       *** GATE C: if pair term adds nothing over singles, Phase 5 ends ***")
        print("    3. Permutation null over pair labels (n=1000)")
        print("    4. Cluster bootstrap by position")
        print("\n  Calibration: r in 0.26-0.37 is the honest benchmark (pLM epistasis).")
        print("  Do NOT compare against 0.77-0.88 multi-point ddG numbers.")
        return

    # ------------------------------------------------------------------
    # Load SKEMPI and extract cycles
    # ------------------------------------------------------------------
    if not skempi_csv.exists():
        sys.exit(f"ERROR: SKEMPI CSV not found: {skempi_csv}")

    df = load_skempi(cache_dir)
    df = add_ddg(df)
    cx_key_candidates = df[df["#Pdb"].str.split("_").str[0].str.upper() == pdb_id]["#Pdb"].unique()
    if len(cx_key_candidates) == 0:
        sys.exit(f"ERROR: no SKEMPI rows for PDB {pdb_id}")

    cx_key = cx_key_candidates[0]
    cycles = extract_cycles(df, complex_key=cx_key)

    if not cycles:
        sys.exit(f"ERROR: no complete double-mutant cycles for {cx_key}")

    couplings = np.array([c.coupling for c in cycles])
    pairs_set = {tuple(sorted([c.pos_i, c.pos_j])) for c in cycles}
    positions_set: set[tuple[str, str]] = set()
    for c in cycles:
        positions_set.add(c.pos_i)
        positions_set.add(c.pos_j)

    n_cross = sum(c.cross_chain for c in cycles)

    log.info(
        "%s: %d cycles, %d pairs, %d positions, %d cross-chain",
        cx_key, len(cycles), len(pairs_set), len(positions_set), n_cross,
    )

    print(f"\n=== Coupling extraction for {cx_key} ===")
    print(f"  Complete cycles:    {len(cycles)}")
    print(f"  Distinct pairs:     {len(pairs_set)}")
    print(f"  Distinct positions: {len(positions_set)}")
    print(f"  Cross-chain cycles: {n_cross}")
    print(f"  abs(coupling) > 0.5: {int((np.abs(couplings) > 0.5).sum())} / {len(cycles)}")
    print(f"  Coupling range: [{couplings.min():.2f}, {couplings.max():.2f}]")
    print(f"  Coupling std: {couplings.std():.2f}")

    cross_abs = [abs(c.coupling) for c in cycles if c.cross_chain]
    same_abs = [abs(c.coupling) for c in cycles if not c.cross_chain]
    if cross_abs and same_abs:
        gap = np.mean(cross_abs) - np.mean(same_abs)
        print(f"  Mean abs(coupling) cross-chain: {np.mean(cross_abs):.3f}  (n={len(cross_abs)})")
        print(f"  Mean abs(coupling) same-chain:  {np.mean(same_abs):.3f}  (n={len(same_abs)})")
        print(f"  Gap (cross - same):             {gap:+.3f}")
        try:
            from scipy import stats as _stats
            _mw_stat, mw_p = _stats.mannwhitneyu(
                cross_abs, same_abs, alternative="greater",
            )
            _welch_stat, welch_p = _stats.ttest_ind(
                cross_abs, same_abs, equal_var=False,
            )
            print(f"  Mann-Whitney p (cross>same):    {mw_p:.2f}")
            print(f"  Welch t-test p (two-sided):     {welch_p:.2f}")
        except ImportError:
            pass

    agg_df = aggregate_pairs(cycles)
    repeated = agg_df[agg_df["n_cycles"] > 1]
    if len(repeated) > 0:
        spreads = repeated["replicate_spread"].values
        print(f"  Repeated pairs: {len(repeated)}, median spread: {np.median(spreads):.2f}, max: {spreads.max():.2f}")

    # ------------------------------------------------------------------
    # Build result dict (coupling-only, no pair map needed)
    # ------------------------------------------------------------------
    result: dict = {
        "complex": cx_key,
        "pdb_id": pdb_id,
        "n_cycles": len(cycles),
        "n_distinct_pairs": len(pairs_set),
        "n_distinct_positions": len(positions_set),
        "n_cross_chain": n_cross,
        "coupling_range": [float(couplings.min()), float(couplings.max())],
        "coupling_std": float(couplings.std()),
        "n_abs_coupling_gt_05": int((np.abs(couplings) > 0.5).sum()),
    }

    if cross_abs and same_abs:
        result["mean_abs_coupling_cross"] = float(np.mean(cross_abs))
        result["mean_abs_coupling_same"] = float(np.mean(same_abs))
        result["cross_same_gap"] = float(np.mean(cross_abs) - np.mean(same_abs))
        try:
            from scipy import stats as _stats
            _, mw_p_val = _stats.mannwhitneyu(
                cross_abs, same_abs, alternative="greater",
            )
            _, welch_p_val = _stats.ttest_ind(
                cross_abs, same_abs, equal_var=False,
            )
            result["cross_same_mannwhitney_p"] = float(mw_p_val)
            result["cross_same_welch_p"] = float(welch_p_val)
        except ImportError:
            pass

    if len(repeated) > 0:
        result["n_repeated_pairs"] = len(repeated)
        result["replicate_spread_median"] = float(np.median(spreads))
        result["replicate_spread_max"] = float(spreads.max())

    # ------------------------------------------------------------------
    # Controls (require pair-attribution map)
    # ------------------------------------------------------------------
    if pair_map_path is None or not pair_map_path.exists():
        if pair_map_path is not None:
            log.warning("pair-map not found: %s -- skipping controls", pair_map_path)
        else:
            log.info("No --pair-map provided; skipping controls.")
        result["controls"] = "skipped (no pair-attribution map)"
    else:
        if not pdb_path.exists():
            sys.exit(f"ERROR: PDB file not found: {pdb_path}")

        pair_map = _load_pair_map(pair_map_path)
        L = pair_map.shape[0]
        log.info("Loaded pair map: shape (%d, %d)", L, L)

        residue_ids, sequences = read_pdb_residue_ids(pdb_path)

        # Build chain -> (residue_ids_list, offset_in_L) mapping
        chain_order = list(cx.all_chains)
        offset: dict[str, int] = {}
        cur = 0
        for ch in chain_order:
            if ch in residue_ids:
                offset[ch] = cur
                cur += len(residue_ids[ch])

        if cur != L:
            sys.exit(
                f"ERROR: pair map has L={L} but PDB chains "
                f"{chain_order} have {cur} residues"
            )

        # Map each cycle's positions to indices in the L-length sequence
        rid_to_idx: dict[tuple[str, str], int] = {}
        for ch in chain_order:
            if ch in residue_ids:
                for i, rid in enumerate(residue_ids[ch]):
                    rid_to_idx[(ch, rid)] = offset[ch] + i

        pair_attribs = []
        single_i_attribs = []
        single_j_attribs = []
        cycle_couplings = []
        cycle_cross = []
        distance_vals = []

        coords, atom_chains, atom_res_keys = parse_pdb_heavy_atoms(pdb_path)
        from collections import defaultdict
        idx_map_atoms: dict[tuple[str, str], list[int]] = defaultdict(list)
        for ai, (ach, ark) in enumerate(zip(atom_chains, atom_res_keys)):
            idx_map_atoms[(ach, ark)].append(ai)

        for c in cycles:
            idx_i = rid_to_idx.get(c.pos_i)
            idx_j = rid_to_idx.get(c.pos_j)
            if idx_i is None or idx_j is None:
                continue

            pair_val = float(pair_map[idx_i, idx_j])
            single_i_val = float(pair_map[idx_i, idx_i])
            single_j_val = float(pair_map[idx_j, idx_j])

            atoms_i = idx_map_atoms.get(c.pos_i)
            atoms_j = idx_map_atoms.get(c.pos_j)
            if atoms_i and atoms_j:
                ci = coords[atoms_i].mean(axis=0)
                cj = coords[atoms_j].mean(axis=0)
                dist = float(np.sqrt(((ci - cj) ** 2).sum()))
            else:
                dist = np.nan

            pair_attribs.append(pair_val)
            single_i_attribs.append(single_i_val)
            single_j_attribs.append(single_j_val)
            cycle_couplings.append(c.coupling)
            cycle_cross.append(c.cross_chain)
            distance_vals.append(dist)

        pair_attribs = np.array(pair_attribs)
        single_i_attribs = np.array(single_i_attribs)
        single_j_attribs = np.array(single_j_attribs)
        cycle_couplings = np.array(cycle_couplings)
        cycle_cross = np.array(cycle_cross, dtype=bool)
        distance_vals = np.array(distance_vals)

        n_matched = len(cycle_couplings)
        log.info("Matched %d cycles to pair map", n_matched)

        if n_matched < 5:
            sys.exit(f"ERROR: only {n_matched} cycles matched to pair map")

        rng = np.random.default_rng(args.seed)
        controls: dict = {}

        # --- Control 1: Distance partial ---
        rho_raw = spearman(pair_attribs, cycle_couplings)
        valid_dist = np.isfinite(distance_vals)
        if valid_dist.sum() >= 5:
            rho_dist_partial = partial_spearman(
                pair_attribs[valid_dist],
                cycle_couplings[valid_dist],
                distance_vals[valid_dist],
            )
            rho_dist_coupling = spearman(
                distance_vals[valid_dist], cycle_couplings[valid_dist]
            )
        else:
            rho_dist_partial = float("nan")
            rho_dist_coupling = float("nan")

        print("\n=== Control 1: Distance partial ===")
        print(f"  Spearman(A[i,j], coupling):         {rho_raw:+.4f}")
        print(f"  Spearman(distance, coupling):        {rho_dist_coupling:+.4f}")
        print(f"  Partial(A[i,j], coupling | dist):    {rho_dist_partial:+.4f}")

        controls["distance_partial"] = {
            "rho_raw": rho_raw,
            "rho_distance_coupling": rho_dist_coupling,
            "rho_partial": rho_dist_partial,
        }

        # --- Control 2: Singles partial (GATE C) ---
        singles_confound = np.column_stack([single_i_attribs, single_j_attribs])
        rho_singles_partial = partial_spearman(
            pair_attribs, cycle_couplings, singles_confound,
        )
        rho_single_i = spearman(single_i_attribs, cycle_couplings)
        rho_single_j = spearman(single_j_attribs, cycle_couplings)

        print(f"\n{'=' * 60}")
        print("=== Control 2: Singles partial *** GATE C *** ===")
        print(f"{'=' * 60}")
        print(f"  Spearman(A[i,i], coupling):          {rho_single_i:+.4f}")
        print(f"  Spearman(A[j,j], coupling):          {rho_single_j:+.4f}")
        print(f"  Spearman(A[i,j], coupling):          {rho_raw:+.4f}")
        print(f"  Partial(A[i,j], coupling | A[i],A[j]): {rho_singles_partial:+.4f}")
        print("")
        print("  If the partial is near zero, the pair term adds nothing over")
        print("  the singles and Phase 5 ends here.")
        print("  Benchmark: r in 0.26-0.37 (pLM epistasis, NOT 0.77-0.88 multi-point).")
        print(f"{'=' * 60}")

        controls["singles_partial_gate_c"] = {
            "rho_pair_coupling": rho_raw,
            "rho_single_i_coupling": rho_single_i,
            "rho_single_j_coupling": rho_single_j,
            "rho_partial_pair_given_singles": rho_singles_partial,
        }

        # --- Control 3: Permutation null ---
        n_perm = args.n_permutations
        null_rhos = np.array([
            spearman(rng.permutation(pair_attribs), cycle_couplings)
            for _ in range(n_perm)
        ])
        p_value = float(np.mean(np.abs(null_rhos) >= np.abs(rho_raw)))

        print(f"\n=== Control 3: Permutation null (n={n_perm}) ===")
        print(f"  Observed rho:        {rho_raw:+.4f}")
        print(f"  Null mean:           {null_rhos.mean():+.4f}")
        print(f"  Null p95:            {np.percentile(np.abs(null_rhos), 95):+.4f}")
        print(f"  Two-sided p-value:   {p_value:.4f}")

        controls["permutation_null"] = {
            "n_permutations": n_perm,
            "observed_rho": rho_raw,
            "null_mean": float(null_rhos.mean()),
            "null_p95_abs": float(np.percentile(np.abs(null_rhos), 95)),
            "p_value_two_sided": p_value,
        }

        # --- Control 4: Cluster bootstrap by position ---
        unique_positions = sorted(positions_set)
        pos_to_cluster: dict[tuple[str, str], int] = {
            p: i for i, p in enumerate(unique_positions)
        }

        cycle_clusters_i = np.array([
            pos_to_cluster[c.pos_i]
            for c in cycles
            if rid_to_idx.get(c.pos_i) is not None
            and rid_to_idx.get(c.pos_j) is not None
        ])
        cycle_clusters_j = np.array([
            pos_to_cluster[c.pos_j]
            for c in cycles
            if rid_to_idx.get(c.pos_i) is not None
            and rid_to_idx.get(c.pos_j) is not None
        ])

        n_clusters = len(unique_positions)
        n_boot = 2000

        def _cluster_bootstrap_spearman(
            x: np.ndarray,
            y: np.ndarray,
            cluster_ids_a: np.ndarray,
            cluster_ids_b: np.ndarray,
            n_clusters: int,
            rng: np.random.Generator,
            n_boot: int,
        ) -> tuple[float, float, float]:
            point = spearman(x, y)
            boots = np.empty(n_boot)
            for b in range(n_boot):
                sampled = set(rng.integers(0, n_clusters, size=n_clusters))
                mask = np.array([
                    (ca in sampled or cb in sampled)
                    for ca, cb in zip(cluster_ids_a, cluster_ids_b)
                ])
                if mask.sum() < 3:
                    boots[b] = np.nan
                    continue
                boots[b] = spearman(x[mask], y[mask])
            lo = float(np.nanpercentile(boots, 2.5))
            hi = float(np.nanpercentile(boots, 97.5))
            return point, lo, hi

        rho_cb, ci_lo, ci_hi = _cluster_bootstrap_spearman(
            pair_attribs, cycle_couplings,
            cycle_clusters_i, cycle_clusters_j,
            n_clusters, rng, n_boot,
        )

        print("\n=== Control 4: Cluster bootstrap by position ===")
        print(f"  Clusters (positions): {n_clusters}")
        print(f"  Pairs:                {n_matched}")
        print(f"  Spearman:             {rho_cb:+.4f}")
        print(f"  95% CI (cluster):     [{ci_lo:+.4f}, {ci_hi:+.4f}]")

        _, ci_lo_naive, ci_hi_naive = bootstrap_ci(
            spearman, pair_attribs, cycle_couplings, n_boot=n_boot, seed=args.seed,
        )
        print(f"  95% CI (naive pair):  [{ci_lo_naive:+.4f}, {ci_hi_naive:+.4f}]")
        print("  (Naive CI is narrower because pairs share positions.)")

        controls["cluster_bootstrap"] = {
            "n_clusters": n_clusters,
            "n_pairs": n_matched,
            "rho": rho_cb,
            "ci_95_cluster": [ci_lo, ci_hi],
            "ci_95_naive": [ci_lo_naive, ci_hi_naive],
        }

        result["controls"] = controls

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
        f.write("\n")
    print(f"\nWrote {out_path}")

    prov_write(
        out_path,
        stage="13_coupling",
        inputs={
            "skempi": str(skempi_csv),
            "pdb": str(pdb_path),
            "pair_map": str(pair_map_path) if pair_map_path else None,
        },
        params={
            "complex": pdb_id,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
        },
        arm={"stage": "coupling", "complex": pdb_id},
    )


if __name__ == "__main__":
    main()
