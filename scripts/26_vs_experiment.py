#!/usr/bin/env python3
"""Stage 26 -- model vs lab, and shortcut vs lab, on measured SKEMPI DDG.

Two of the project's three comparisons are computed here, per complex, on the
exact mutations SKEMPI measured:

    model vs lab     (B vs C)  -- the brute-force scan delta vs measured DDG
    shortcut vs lab  (A vs C)  -- the res_type attribution delta vs measured DDG

where, for one substitution ``pos -> mut``:

    B = f(mutant) - f(wild type)                        (stage 04 ``delta``)
    A = grad[pos, col(mut)] - grad[pos, col(wt)]        (stage 03 ``grad_res_type``)
    C = R*T*ln(Kd_mut / Kd_wt)                          (SKEMPI, kcal/mol)

Sign convention
---------------
``complex_pde`` is a predicted *distance error*: a LOWER value is a better
complex.  ``delta`` is ``score(mutant) - score(wild type)``, so a POSITIVE
delta means the model thinks the mutant complex is worse.  SKEMPI DDG is
``R*T*ln(Kd_mut/Kd_wt)``, so a POSITIVE DDG means the mutation weakens
binding.  Both are therefore already on the same "positive = destabilising"
axis and must NOT be negated -- agreement shows up as a POSITIVE correlation.
Higher-is-better scores (``iptm``, ``complex_iplddt``, ...) are negated by
:func:`orient_for_experiment`, exactly as in stage 12.  The attribution A is a
first-order Taylor term in the units of the score, so it inherits the score's
orientation.

Every correlation is reported three ways: raw, partial on minimum heavy-atom
distance to the partner chain (this project was fooled by geometry once), and
in magnitude (|A| or |B| vs |DDG|) so the numbers are comparable with the
position-level hot-spot pipeline that worked in |DDG|.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.attrib import CANONICAL_AMINO_ACIDS  # noqa: E402
from igv.metrics import bootstrap_ci, partial_spearman, spearman  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402
from igv.skempi import (  # noqa: E402
    SKEMPI_MUTATION_COL,
    add_ddg,
    compute_distance_to_partner,
    filter_complex,
    get_complex,
    load_skempi,
    parse_mutation,
    parse_pdb_heavy_atoms,
    read_pdb_residue_ids,
    single_point,
)

log = logging.getLogger("vs_experiment")

AA_LIST = list(CANONICAL_AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Scores where a LOWER value means better binding quality.  Kept identical to
# ``12_compare.LOWER_IS_BETTER``; see commit f1a0a0c for why this set is
# written out per score name rather than inferred.
LOWER_IS_BETTER = {"complex_pde", "affinity_pred_value"}

# A mutation at or above this DDG is treated as a hot spot for the orientation
# anchor; at or below the cold threshold it is treated as neutral.
HOTSPOT_DDG = 2.0
NEUTRAL_DDG = 0.5


def orient_for_experiment(values: np.ndarray, score: str) -> np.ndarray:
    """Negate higher-is-better scores so that an increase means destabilising.

    SKEMPI DDG is positive for weaker binding.  After this call a positive
    value of *values* also means weaker binding, so agreement with experiment
    is a positive correlation for every score name.
    """
    values = np.asarray(values, dtype=np.float64)
    if score in LOWER_IS_BETTER:
        return values
    return -values


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, ignoring non-finite pairs. NaN when fewer than 2 remain."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    r, _ = stats.pearsonr(x[mask], y[mask])
    return float(r)


# ---------------------------------------------------------------------------
# Experimental ground truth
# ---------------------------------------------------------------------------


def load_experimental_ddg(
    dataset: str,
    chain: str,
    cache_dir: Path,
    *,
    allow_mismatch: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Return one row per measured single-point mutation on *chain*.

    Uses the repo's SKEMPI loader throughout: ``Mutation(s)_PDB`` for author
    numbering (``Mutation(s)_cleaned`` is sequential and silently wrong on
    1JTG), ``add_ddg`` for ``R*T*ln(Kd_mut/Kd_wt)``, and
    ``read_pdb_residue_ids`` to map author residue IDs onto the 0-based chain
    index the scan and the attribution both use.

    Duplicate measurements of the same mutation (different papers, methods or
    temperatures) are collapsed by MEDIAN, with the spread retained so the
    write-up can quote the experimental noise floor.

    Returns ``(df, raw_df, diagnostics)``.  *df* has columns ``pos``,
    ``mut_aa``, ``wt_aa``, ``residue_id``, ``ddg``, ``n_measurements``,
    ``ddg_min``, ``ddg_max``; *raw_df* keeps one row per measurement so the
    test-retest ceiling can be estimated.
    """
    cache_dir = Path(cache_dir)
    pdb_id = dataset.upper()
    pdb_path = cache_dir / f"{pdb_id.lower()}.pdb"
    if not pdb_path.exists():
        raise SystemExit(f"PDB not found: {pdb_path}")

    residue_ids, sequences = read_pdb_residue_ids(pdb_path)
    if chain not in residue_ids:
        raise SystemExit(
            f"Chain {chain} not in {pdb_path}. Available: {sorted(residue_ids)}"
        )
    chain_ids = residue_ids[chain]
    chain_seq = sequences[chain]
    id_to_idx = {rid: i for i, rid in enumerate(chain_ids)}

    df = add_ddg(single_point(filter_complex(load_skempi(cache_dir), pdb_id)))

    rows: list[dict] = []
    n_other_chain = 0
    n_not_found = 0
    n_mismatch = 0
    for raw, ddg_val in zip(df[SKEMPI_MUTATION_COL], df["ddg_kcal_mol"]):
        mut = parse_mutation(str(raw).strip())
        if mut.chain != chain:
            n_other_chain += 1
            continue
        idx = id_to_idx.get(mut.resnum)
        if idx is None:
            n_not_found += 1
            continue
        if chain_seq[idx] != mut.wt_aa:
            n_mismatch += 1
            if not allow_mismatch:
                raise SystemExit(
                    f"Wild-type mismatch for {raw}: SKEMPI says {mut.wt_aa}, "
                    f"PDB has {chain_seq[idx]} at index {idx}"
                )
            continue
        rows.append(
            {
                "pos": idx,
                "mut_aa": mut.mut_aa,
                "wt_aa": mut.wt_aa,
                "residue_id": mut.resnum,
                "ddg_raw": float(ddg_val),
            }
        )

    if not rows:
        raise SystemExit(f"No SKEMPI single-point mutations on {pdb_id} chain {chain}")

    raw_df = pd.DataFrame(rows)
    agg = (
        raw_df.groupby(["pos", "mut_aa", "wt_aa", "residue_id"], as_index=False)
        .agg(
            ddg=("ddg_raw", "median"),
            n_measurements=("ddg_raw", "size"),
            ddg_min=("ddg_raw", "min"),
            ddg_max=("ddg_raw", "max"),
        )
        .sort_values(["pos", "mut_aa"])
        .reset_index(drop=True)
    )

    spreads = (agg["ddg_max"] - agg["ddg_min"])[agg["n_measurements"] > 1]
    counts = agg["ddg"].round(6).value_counts()
    diagnostics = {
        "n_skempi_rows_chain": int(len(raw_df)),
        "n_unique_mutations": int(len(agg)),
        "n_positions": int(agg["pos"].nunique()),
        "n_rows_other_chain": n_other_chain,
        "n_residue_id_not_found": n_not_found,
        "n_wild_type_mismatch": n_mismatch,
        "n_replicated_mutations": int((agg["n_measurements"] > 1).sum()),
        "replicate_spread_median_kcal": (
            float(spreads.median()) if len(spreads) else None
        ),
        "replicate_spread_max_kcal": float(spreads.max()) if len(spreads) else None,
        "n_distinct_ddg_values": int(agg["ddg"].round(6).nunique()),
        "largest_tie_group": int(counts.iloc[0]) if len(counts) else 0,
        "n_exactly_zero_ddg": int((agg["ddg"].abs() < 1e-9).sum()),
    }
    return agg, raw_df, diagnostics


def experimental_reliability(
    raw_df: pd.DataFrame, *, n_draws: int = 1000, seed: int = 42
) -> dict:
    """Split-half test-retest Spearman of the experimental column itself.

    Mutations measured more than once are split at random into two halves;
    each half is summarised by its median and the two halves are correlated
    across mutations.  This is the ceiling: no predictor can out-correlate
    the measurement's agreement with itself.  Returns ``None`` statistics when
    fewer than three mutations carry replicates.
    """
    counts = raw_df.groupby(["pos", "mut_aa"])["ddg_raw"].size()
    replicated = counts[counts > 1].index
    out: dict = {"n_replicated_mutations": int(len(replicated))}
    if len(replicated) < 3:
        out["test_retest_spearman_median"] = None
        out["note"] = "fewer than 3 replicated mutations; ceiling not estimable"
        return out

    groups = [
        raw_df[(raw_df["pos"] == p) & (raw_df["mut_aa"] == a)]["ddg_raw"].to_numpy()
        for p, a in replicated
    ]
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_draws)
    for i in range(n_draws):
        first, second = [], []
        for vals in groups:
            shuffled = rng.permutation(vals)
            half = max(1, len(shuffled) // 2)
            first.append(float(np.median(shuffled[:half])))
            second.append(float(np.median(shuffled[half:])))
        rhos[i] = spearman(np.array(first), np.array(second))
    out["test_retest_spearman_median"] = float(np.nanmedian(rhos))
    out["test_retest_spearman_p5"] = float(np.nanpercentile(rhos, 5))
    out["test_retest_spearman_p95"] = float(np.nanpercentile(rhos, 95))
    out["n_draws"] = int(n_draws)
    return out


def load_scan_deltas(scan_path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a stage-04 scan CSV into ``pos``/``mut_aa``/``delta`` rows.

    Drops the WT row and any mutant whose delta is not finite (the refeaturise
    arm leaves NaN where the ``token_to_rep_atom`` guard skipped a mutant).
    """
    df = pd.read_csv(scan_path)
    for col in ("mutation_id", "substitutions", "delta"):
        if col not in df.columns:
            raise SystemExit(
                f"Scan CSV {scan_path} missing column {col!r}; expected the "
                "stage-04 schema."
            )
    mutants = df[df["mutation_id"].astype(str) != "WT"].copy()
    n_rows = len(mutants)
    subs = mutants["substitutions"].astype(str).str.strip()
    mutants = mutants[subs.str.len() >= 2].copy()
    subs = mutants["substitutions"].astype(str).str.strip()
    mutants["pos"] = subs.str[:-1].astype(int)
    mutants["mut_aa"] = subs.str[-1]
    mutants["delta"] = pd.to_numeric(mutants["delta"], errors="coerce")
    finite = mutants[np.isfinite(mutants["delta"])].copy()

    modes = sorted(set(df.get("mode", pd.Series(dtype=object)).dropna().astype(str)))
    info = {
        "n_scan_mutants": int(n_rows),
        "n_scan_scored": int(len(finite)),
        "scan_modes": modes,
    }
    return finite[["pos", "mut_aa", "delta"]].reset_index(drop=True), info


def attribution_deltas(npz_path: Path, positions, mut_aas) -> tuple[np.ndarray, str]:
    """A[pos, mut] = grad_res_type[pos, col(mut)] - grad_res_type[pos, col(wt)].

    Returns ``(values, method)``; entries are NaN where the position is out of
    range for the attributed chain slice or the target residue is not one of
    the 20 canonical amino acids.
    """
    data = np.load(npz_path, allow_pickle=True)
    for key in ("grad_res_type", "aa_token_indices", "wt_indices"):
        if key not in data:
            raise SystemExit(
                f"Attribution NPZ {npz_path} has no {key!r}; re-run stage 03 "
                "without --no-res-type-grad."
            )
    grad = data["grad_res_type"]
    aa_token_indices = data["aa_token_indices"]
    wt_indices = data["wt_indices"]
    method = str(data["method"]) if "method" in data else "unknown"

    out = np.full(len(positions), np.nan)
    for i, (pos, aa) in enumerate(zip(positions, mut_aas)):
        pos = int(pos)
        if pos < 0 or pos >= grad.shape[0] or aa not in AA_TO_IDX:
            continue
        mut_col = int(aa_token_indices[AA_TO_IDX[aa]])
        wt_col = int(aa_token_indices[int(wt_indices[pos])])
        out[i] = float(grad[pos, mut_col] - grad[pos, wt_col])
    return out, method


def distance_to_partner_by_position(
    dataset: str, chain: str, cache_dir: Path, positions
) -> np.ndarray:
    """Minimum heavy-atom distance to the partner chain, per requested position."""
    pdb_id = dataset.upper()
    pdb_path = Path(cache_dir) / f"{pdb_id.lower()}.pdb"
    cx = get_complex(pdb_id)
    if chain in cx.partner1:
        partner_chains = cx.partner2
    elif chain in cx.partner2:
        partner_chains = cx.partner1
    else:
        raise SystemExit(f"Chain {chain} is not a registered partner of {pdb_id}")

    residue_ids, _ = read_pdb_residue_ids(pdb_path)
    coords, atom_chains, atom_res_keys = parse_pdb_heavy_atoms(pdb_path)
    all_dist = compute_distance_to_partner(
        coords, atom_chains, atom_res_keys, chain, partner_chains, residue_ids[chain]
    )
    return np.array([all_dist[int(p)] for p in positions], dtype=np.float64)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def permutation_null(
    x: np.ndarray, y: np.ndarray, *, n_perms: int = 10000, seed: int = 42
) -> dict:
    """Shuffled-x null for Spearman: p5, p95 and a two-sided p on |rho|."""
    rng = np.random.default_rng(seed)
    observed = spearman(x, y)
    rhos = np.empty(n_perms)
    for i in range(n_perms):
        rhos[i] = spearman(rng.permutation(x), y)
    return {
        "observed": float(observed),
        "p5": float(np.nanpercentile(rhos, 5)),
        "p95": float(np.nanpercentile(rhos, 95)),
        "p_value_two_sided": float(
            np.sum(np.abs(rhos) >= abs(observed)) / n_perms
        ),
        "n_perms": int(n_perms),
    }


def correlation_block(
    x: np.ndarray,
    y: np.ndarray,
    distance: np.ndarray | None = None,
    *,
    n_perms: int = 10000,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Spearman + Pearson with bootstrap CIs, permutation null and distance partial."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if distance is not None:
        distance = np.asarray(distance, dtype=np.float64)
        mask &= np.isfinite(distance)
    x, y = x[mask], y[mask]
    d = distance[mask] if distance is not None else None

    n = int(len(x))
    block: dict = {"n": n}
    if n < 3:
        block["underpowered"] = True
        block["spearman"] = spearman(x, y) if n >= 2 else float("nan")
        return block

    rho = spearman(x, y)
    _, s_lo, s_hi = bootstrap_ci(spearman, x, y, n_boot=n_boot, seed=seed)
    r = pearson(x, y)
    _, p_lo, p_hi = bootstrap_ci(pearson, x, y, n_boot=n_boot, seed=seed)

    block.update(
        {
            "spearman": rho,
            "spearman_ci": [s_lo, s_hi],
            "pearson": r,
            "pearson_ci": [p_lo, p_hi],
            "permutation_null": permutation_null(x, y, n_perms=n_perms, seed=seed),
        }
    )

    if d is not None:
        partial = partial_spearman(x, y, d)
        _, pa_lo, pa_hi = bootstrap_ci(
            partial_spearman, x, y, d, n_boot=n_boot, seed=seed
        )
        block["partial_spearman_distance"] = partial
        block["partial_spearman_distance_ci"] = [pa_lo, pa_hi]
        block["partial_drop_fraction"] = (
            float((rho - partial) / rho)
            if np.isfinite(rho) and abs(rho) > 1e-12
            else None
        )
        block["distance_vs_x_spearman"] = spearman(d, x)
        block["distance_vs_y_spearman"] = spearman(d, y)
    return block


def orientation_anchor(delta: np.ndarray, ddg: np.ndarray) -> dict:
    """Check the sign convention on the measured hot spots before quoting anything.

    If ``positive delta = destabilising`` is right, the mutations experiment
    calls hot (DDG >= 2) must carry a higher median oriented score than the
    ones it calls neutral (DDG <= 0.5).  Records the verdict rather than
    raising: an inverted anchor is itself a result, but it must be visible.
    """
    delta = np.asarray(delta, dtype=np.float64)
    ddg = np.asarray(ddg, dtype=np.float64)
    hot = delta[ddg >= HOTSPOT_DDG]
    cold = delta[ddg <= NEUTRAL_DDG]
    out = {
        "hotspot_ddg_threshold": HOTSPOT_DDG,
        "neutral_ddg_threshold": NEUTRAL_DDG,
        "n_hot": int(len(hot)),
        "n_neutral": int(len(cold)),
        "median_oriented_score_hot": float(np.median(hot)) if len(hot) else None,
        "median_oriented_score_neutral": float(np.median(cold)) if len(cold) else None,
    }
    if len(hot) and len(cold):
        out["separation"] = float(np.median(hot) - np.median(cold))
        out["sign_as_expected"] = bool(np.median(hot) > np.median(cold))
        if len(hot) >= 2 and len(cold) >= 2:
            u = stats.mannwhitneyu(hot, cold, alternative="two-sided")
            out["mannwhitney_p"] = float(u.pvalue)
    else:
        out["separation"] = None
        out["sign_as_expected"] = None
    return out


def _parse_attribution_arg(spec: str) -> tuple[str | None, Path]:
    """``label=path`` or bare ``path`` (label then taken from the npz method)."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    return None, Path(spec.strip())


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_results(args) -> dict:
    """Join scan, attribution and SKEMPI, and compute every reported statistic."""
    cache_dir = Path(args.cache_dir)
    scan_path = Path(args.scan)
    if not scan_path.exists():
        raise SystemExit(f"Scan file not found: {scan_path}")

    exp_df, exp_raw, exp_diag = load_experimental_ddg(
        args.dataset, args.chain, cache_dir, allow_mismatch=not args.strict_wildtype
    )
    scan_df, scan_info = load_scan_deltas(scan_path)

    joined = exp_df.merge(scan_df, on=["pos", "mut_aa"], how="inner")
    n_join = len(joined)
    log.info(
        "%s chain %s [%s]: %d SKEMPI mutations, %d scored scan rows, %d joined",
        args.dataset, args.chain, args.mode,
        exp_diag["n_unique_mutations"], scan_info["n_scan_scored"], n_join,
    )

    results: dict = {
        "dataset": args.dataset,
        "chain": args.chain,
        "score": args.score,
        "mode": args.mode,
        "scan": str(scan_path),
        "experiment": {
            "source": "SKEMPI 2.0",
            "ddg_definition": "R*T*ln(Kd_mut/Kd_wt), kcal/mol, positive = destabilising",
            "duplicate_aggregation": "median",
            **exp_diag,
        },
        "scan_info": scan_info,
        "orientation": {
            "score": args.score,
            "lower_is_better": args.score in LOWER_IS_BETTER,
            "negated_for_experiment": args.score not in LOWER_IS_BETTER,
            "rule": (
                "complex_pde is a predicted distance error: lower is a better "
                "complex, so delta = f(mut) - f(wt) > 0 means destabilising, "
                "matching SKEMPI DDG > 0. No negation."
            ),
        },
        "n_joined": n_join,
        "n_positions_joined": int(joined["pos"].nunique()) if n_join else 0,
    }

    if n_join < 3:
        results["underpowered"] = True
        results["verdict"] = (
            f"Only {n_join} mutations survive the join; nothing is interpretable."
        )
        return results

    positions = joined["pos"].to_numpy()
    mut_aas = joined["mut_aa"].tolist()
    joined_keys = set(zip(joined["pos"].tolist(), joined["mut_aa"].tolist()))
    raw_joined = exp_raw[
        [
            (p, a) in joined_keys
            for p, a in zip(exp_raw["pos"].tolist(), exp_raw["mut_aa"].tolist())
        ]
    ]
    results["experiment"]["reliability"] = experimental_reliability(
        raw_joined, seed=args.seed
    )
    ddg = joined["ddg"].to_numpy(dtype=np.float64)
    b_signed = orient_for_experiment(joined["delta"].to_numpy(), args.score)
    distance = distance_to_partner_by_position(
        args.dataset, args.chain, cache_dir, positions
    )

    results["distance_to_partner"] = {
        "min": float(np.min(distance)),
        "max": float(np.max(distance)),
        "median": float(np.median(distance)),
        "n_within_6A": int(np.sum(distance <= 6.0)),
    }
    results["orientation_anchor"] = orientation_anchor(b_signed, ddg)

    stat_kwargs = {
        "n_perms": args.n_perms,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }

    results["model_vs_lab"] = {
        "signed": correlation_block(b_signed, ddg, distance, **stat_kwargs),
        "magnitude": correlation_block(
            np.abs(b_signed), np.abs(ddg), distance, **stat_kwargs
        ),
    }
    nonzero = np.abs(ddg) > 1e-9
    results["model_vs_lab"]["signed_excluding_zero_ddg"] = correlation_block(
        b_signed[nonzero], ddg[nonzero], distance[nonzero], **stat_kwargs
    )

    results["shortcut_vs_lab"] = {}
    results["shortcut_vs_model"] = {}
    for spec in args.attribution or []:
        label, path = _parse_attribution_arg(spec)
        if not path.exists():
            raise SystemExit(f"Attribution file not found: {path}")
        a_raw, method = attribution_deltas(path, positions, mut_aas)
        a_signed = orient_for_experiment(a_raw, args.score)
        key = label or method
        results["shortcut_vs_lab"][key] = {
            "attribution": str(path),
            "method": method,
            "signed": correlation_block(a_signed, ddg, distance, **stat_kwargs),
            "magnitude": correlation_block(
                np.abs(a_signed), np.abs(ddg), distance, **stat_kwargs
            ),
        }
        # Same rows as A-vs-C, so the bound "B vs C bounds A vs C" is checked
        # on an identical sample rather than against stage 24's full scan.
        results["shortcut_vs_model"][key] = correlation_block(
            a_signed, b_signed, distance, **stat_kwargs
        )

    b_rho = results["model_vs_lab"]["signed"].get("spearman", float("nan"))
    bound = {}
    for key, blk in results["shortcut_vs_lab"].items():
        a_rho = blk["signed"].get("spearman", float("nan"))
        a_ci = blk["signed"].get("spearman_ci", [float("nan"), float("nan")])
        b_ci = results["model_vs_lab"]["signed"].get(
            "spearman_ci", [float("nan"), float("nan")]
        )
        bound[key] = {
            "a_vs_c_spearman": a_rho,
            "b_vs_c_spearman": b_rho,
            "a_exceeds_b": bool(np.isfinite(a_rho) and np.isfinite(b_rho) and a_rho > b_rho),
            "excess": float(a_rho - b_rho) if np.isfinite(a_rho) and np.isfinite(b_rho) else None,
            "cis_overlap": bool(a_ci[0] <= b_ci[1] and b_ci[0] <= a_ci[1]),
        }
    results["bound_check"] = bound

    results["per_mutation"] = [
        {
            "residue_id": row.residue_id,
            "pos": int(row.pos),
            "wt_aa": row.wt_aa,
            "mut_aa": row.mut_aa,
            "ddg_kcal_mol": float(row.ddg),
            "n_measurements": int(row.n_measurements),
            "ddg_spread": float(row.ddg_max - row.ddg_min),
            "scan_delta_oriented": float(b_signed[i]),
            "distance_to_partner": float(distance[i]),
        }
        for i, row in enumerate(joined.itertuples())
    ]
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--dataset", required=True, help="PDB id, e.g. 3S9D")
    parser.add_argument("--chain", required=True, help="Chain the scan was run on")
    parser.add_argument("--scan", required=True, help="Stage-04 scan CSV")
    parser.add_argument(
        "--attribution", action="append", default=[],
        help="Stage-03 npz, as 'label=path' or 'path'. Repeatable.",
    )
    parser.add_argument("--score", default="complex_pde")
    parser.add_argument("--mode", default="resswap", choices=("resswap", "refeaturise"))
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--n-perms", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-wildtype", action="store_true",
        help="Fail instead of skipping when SKEMPI's wild-type residue "
             "disagrees with the PDB sequence.",
    )
    parser.add_argument(
        "--out", default="results/{dataset}_{score}_vs_experiment_{mode}.json",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    out_path = Path(
        args.out.format(dataset=args.dataset, score=args.score, mode=args.mode)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = build_results(args)

    anchor = results.get("orientation_anchor")
    if anchor and anchor.get("sign_as_expected") is False:
        log.warning(
            "ORIENTATION ANCHOR INVERTED: hot-spot median %.5f <= neutral median "
            "%.5f. The model disagrees with experiment on the mutations "
            "experiment is most sure about; do not read the correlation as a "
            "sign error without checking this first.",
            anchor["median_oriented_score_hot"],
            anchor["median_oriented_score_neutral"],
        )

    mvl = results.get("model_vs_lab", {}).get("signed", {})
    if "spearman" in mvl:
        log.info(
            "model vs lab (signed): n=%d rho=%+.4f CI [%+.4f, %+.4f] "
            "null p95=%+.4f p=%.4f partial=%+.4f",
            mvl["n"], mvl["spearman"], mvl["spearman_ci"][0], mvl["spearman_ci"][1],
            mvl["permutation_null"]["p95"], mvl["permutation_null"]["p_value_two_sided"],
            mvl.get("partial_spearman_distance", float("nan")),
        )
    for key, blk in results.get("shortcut_vs_lab", {}).items():
        s = blk["signed"]
        if "spearman" in s:
            log.info(
                "shortcut vs lab [%s] (signed): n=%d rho=%+.4f CI [%+.4f, %+.4f] "
                "null p95=%+.4f p=%.4f partial=%+.4f",
                key, s["n"], s["spearman"], s["spearman_ci"][0], s["spearman_ci"][1],
                s["permutation_null"]["p95"], s["permutation_null"]["p_value_two_sided"],
                s.get("partial_spearman_distance", float("nan")),
            )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
        f.write("\n")
    log.info("Wrote %s", out_path)

    prov_write(
        out_path,
        stage="26_vs_experiment",
        inputs={
            "scan": str(Path(args.scan)),
            "skempi": "skempi_v2.csv",
            **{
                f"attribution_{i}": spec
                for i, spec in enumerate(args.attribution or [])
            },
        },
        params={
            "dataset": args.dataset,
            "chain": args.chain,
            "score": args.score,
            "mode": args.mode,
            "n_perms": args.n_perms,
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        arm={
            "stage": "vs_experiment",
            "dataset": args.dataset,
            "score": args.score,
            "mode": args.mode,
        },
    )


if __name__ == "__main__":
    main()
