#!/usr/bin/env python3
"""Stage 24 -- faithfulness: does the gradient predict the model's own scan?

Computes the first-order Taylor prediction from the res_type gradient:

    A_grad[pos, mut] = grad[pos, col(mut)] - grad[pos, col(wt)]

and, if IG arrays are present:

    A_ig[pos, mut] = ig[pos, col(mut)] - ig[pos, col(wt)]

Compares these against the scan delta B = f(mut) - f(wt) from stage 04.

Reports Spearman and Pearson with bootstrap 95% CIs, a permutation null,
the scale ratio median|A| / median|B|, per-position aggregates, and a
partial correlation controlling for minimum heavy-atom distance to the
partner chain (and burial/RSA if available).
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

log = logging.getLogger("faithfulness")

AA_LIST = list(CANONICAL_AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    r, _ = stats.pearsonr(x[mask], y[mask])
    return float(r)


def _extract_attribution(
    npz_path: Path,
    scan_df: pd.DataFrame,
) -> dict:
    """Extract A_grad and optionally A_ig aligned to scan rows.

    Returns dict with keys:
      - A_grad: 1D array aligned to scan rows
      - A_ig: 1D array or None
      - B: scan delta
      - positions: 0-based positions
      - mut_aas: mutant amino acids
      - method: 'plain_grad' or 'ig'
    """
    data = np.load(npz_path, allow_pickle=True)

    reference_seq = str(data["reference_seq"])
    method = str(data["method"])

    grad_res_type = data.get("grad_res_type")
    if grad_res_type is None:
        raise SystemExit(
            f"Attribution NPZ {npz_path} does not contain 'grad_res_type'. "
            "Re-run stage 03 without --no-res-type-grad."
        )

    aa_token_indices = data.get("aa_token_indices")
    if aa_token_indices is None:
        raise SystemExit(
            f"Attribution NPZ {npz_path} does not contain 'aa_token_indices'."
        )

    wt_indices = data.get("wt_indices")
    if wt_indices is None:
        raise SystemExit(
            f"Attribution NPZ {npz_path} does not contain 'wt_indices'."
        )

    # grad_res_type is (L_chain, num_tokens) -- the chain slice
    # aa_token_indices is (20,) mapping CANONICAL_AMINO_ACIDS[i] -> column in the token axis
    L_chain = grad_res_type.shape[0]

    A_grad_list = []
    B_list = []
    pos_list = []
    aa_list = []

    for _, row in scan_df.iterrows():
        mid = str(row.get("mutation_id", ""))
        if mid == "WT" or mid == "":
            continue

        sub = str(row["substitutions"]).strip()
        if not sub:
            continue

        pos = int(sub[:-1])
        mut_aa = sub[-1]

        if pos < 0 or pos >= L_chain:
            continue
        if mut_aa not in AA_TO_IDX:
            continue

        mut_col = int(aa_token_indices[AA_TO_IDX[mut_aa]])
        wt_col = int(aa_token_indices[int(wt_indices[pos])])

        a_val = float(grad_res_type[pos, mut_col] - grad_res_type[pos, wt_col])
        b_val = float(row["delta"]) if pd.notna(row.get("delta")) else float("nan")

        if not np.isfinite(b_val):
            continue

        A_grad_list.append(a_val)
        B_list.append(b_val)
        pos_list.append(pos)
        aa_list.append(mut_aa)

    return {
        "A_grad": np.array(A_grad_list, dtype=np.float64),
        "A_ig": None,
        "B": np.array(B_list, dtype=np.float64),
        "positions": np.array(pos_list, dtype=int),
        "mut_aas": aa_list,
        "method": method,
        "reference_seq": reference_seq,
    }


def _permutation_null(
    a: np.ndarray,
    b: np.ndarray,
    n_perms: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test for Spearman correlation."""
    rng = np.random.default_rng(seed)
    observed = spearman(a, b)
    perm_rhos = np.empty(n_perms)
    for i in range(n_perms):
        perm_rhos[i] = spearman(rng.permutation(a), b)
    p5 = float(np.percentile(perm_rhos, 5))
    p95 = float(np.percentile(perm_rhos, 95))
    n_extreme = np.sum(np.abs(perm_rhos) >= np.abs(observed))
    p_value = float(n_extreme / n_perms)
    return {
        "observed": float(observed),
        "p5": p5,
        "p95": p95,
        "p_value_two_sided": p_value,
        "n_perms": n_perms,
    }


def _per_position_aggregates(
    a: np.ndarray, b: np.ndarray, positions: np.ndarray,
) -> dict:
    """Sum and max of |A| vs |B| per position."""
    unique_pos = np.unique(positions)
    results = {}
    sum_a = []
    sum_b = []
    max_a = []
    max_b = []
    for p in unique_pos:
        mask = positions == p
        sum_a.append(float(np.sum(np.abs(a[mask]))))
        sum_b.append(float(np.sum(np.abs(b[mask]))))
        max_a.append(float(np.max(np.abs(a[mask]))))
        max_b.append(float(np.max(np.abs(b[mask]))))
    sum_a = np.array(sum_a)
    sum_b = np.array(sum_b)
    max_a = np.array(max_a)
    max_b = np.array(max_b)

    results["n_positions"] = len(unique_pos)
    results["sum_spearman"] = spearman(sum_a, sum_b) if len(sum_a) >= 2 else float("nan")
    results["max_spearman"] = spearman(max_a, max_b) if len(max_a) >= 2 else float("nan")

    if len(sum_a) >= 2:
        _, lo, hi = bootstrap_ci(spearman, sum_a, sum_b)
        results["sum_spearman_ci_lo"] = lo
        results["sum_spearman_ci_hi"] = hi
    if len(max_a) >= 2:
        _, lo, hi = bootstrap_ci(spearman, max_a, max_b)
        results["max_spearman_ci_lo"] = lo
        results["max_spearman_ci_hi"] = hi

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--attribution", required=True,
        help="Stage-03 attribution .npz (must contain grad_res_type)",
    )
    parser.add_argument(
        "--scan", required=True,
        help="Stage-04 scan CSV (must contain mutation_id, substitutions, delta)",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--chain", default="H")
    parser.add_argument("--score", default="complex_pde")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument(
        "--out",
        default="results/{dataset}_{score}_faithfulness.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    attr_path = Path(args.attribution)
    scan_path = Path(args.scan)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out.format(dataset=args.dataset, score=args.score))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not attr_path.exists():
        raise SystemExit(f"Attribution file not found: {attr_path}")
    if not scan_path.exists():
        raise SystemExit(f"Scan file not found: {scan_path}")

    scan_df = pd.read_csv(scan_path)
    for col in ("substitutions", "delta"):
        if col not in scan_df.columns:
            raise SystemExit(
                f"Scan CSV missing column {col!r}. "
                "Expected schema from 04_scan with delta column."
            )

    extracted = _extract_attribution(attr_path, scan_df)
    A = extracted["A_grad"]
    B = extracted["B"]
    positions = extracted["positions"]
    method = extracted["method"]

    n = len(A)
    if n < 2:
        raise SystemExit(f"Only {n} overlapping mutations between attribution and scan.")

    log.info("Matched %d mutations between attribution and scan", n)

    results: dict = {"n": n, "method": method, "dataset": args.dataset, "score": args.score}

    # Spearman and Pearson with bootstrap CIs
    rho_s = spearman(A, B)
    _, s_lo, s_hi = bootstrap_ci(spearman, A, B)
    rho_p = pearson(A, B)
    _, p_lo, p_hi = bootstrap_ci(pearson, A, B)

    results["spearman"] = rho_s
    results["spearman_ci_lo"] = s_lo
    results["spearman_ci_hi"] = s_hi
    results["pearson"] = rho_p
    results["pearson_ci_lo"] = p_lo
    results["pearson_ci_hi"] = p_hi

    log.info("Spearman = %.4f [%.4f, %.4f]", rho_s, s_lo, s_hi)
    log.info("Pearson  = %.4f [%.4f, %.4f]", rho_p, p_lo, p_hi)

    # Permutation null
    perm = _permutation_null(A, B)
    results["permutation_null"] = perm
    log.info(
        "Permutation null: p5=%.4f p95=%.4f p_two_sided=%.4f",
        perm["p5"], perm["p95"], perm["p_value_two_sided"],
    )

    # Scale ratio
    med_a = float(np.median(np.abs(A)))
    med_b = float(np.median(np.abs(B)))
    scale_ratio = med_a / med_b if med_b > 0 else float("nan")
    results["scale_ratio"] = scale_ratio
    results["median_abs_A"] = med_a
    results["median_abs_B"] = med_b
    log.info("Scale ratio median|A|/median|B| = %.4f", scale_ratio)

    # Per-position aggregates
    pos_agg = _per_position_aggregates(A, B, positions)
    results["per_position"] = pos_agg
    log.info(
        "Per-position (n=%d): sum_spearman=%.4f max_spearman=%.4f",
        pos_agg["n_positions"],
        pos_agg["sum_spearman"],
        pos_agg["max_spearman"],
    )

    # Partial correlation controlling for distance to partner chain
    try:
        from igv.dms import resolve_pdb_complex
        from igv.skempi import compute_confounds, read_pdb_residue_ids

        resolved = resolve_pdb_complex(
            args.dataset, args.chain, cache_dir,
        )
        pdb_path = resolved.pdb_path
        residue_ids, sequences = read_pdb_residue_ids(pdb_path)
        chain_residue_ids = residue_ids[args.chain]
        chain_seq = sequences[args.chain]

        partner_chains = tuple(
            c for c in resolved.chains if c != args.chain
        )

        unique_positions = sorted(set(positions.tolist()))
        confounds = compute_confounds(
            pdb_path,
            args.chain,
            chain_residue_ids,
            chain_seq,
            partner_chains,
            unique_positions,
        )

        pos_to_confound_row = {p: i for i, p in enumerate(unique_positions)}
        confound_names = sorted(confounds.keys())

        if "distance_to_partner" in confound_names:
            dist_vals = np.array([
                confounds["distance_to_partner"][pos_to_confound_row[p]]
                for p in positions
            ])
            partial_rho = partial_spearman(A, B, dist_vals)
            results["partial_spearman_distance_to_partner"] = partial_rho
            log.info("Partial Spearman (controlling distance_to_partner) = %.4f", partial_rho)

        available_confounds = []
        for name in confound_names:
            vals = np.array([
                confounds[name][pos_to_confound_row[p]]
                for p in positions
            ])
            if np.all(np.isfinite(vals)):
                available_confounds.append((name, vals))

        if available_confounds:
            confound_matrix = np.column_stack([v for _, v in available_confounds])
            partial_rho_all = partial_spearman(A, B, confound_matrix)
            results["partial_spearman_all_confounds"] = partial_rho_all
            results["confound_names"] = [n for n, _ in available_confounds]
            log.info(
                "Partial Spearman (all %d confounds) = %.4f",
                len(available_confounds), partial_rho_all,
            )

    except (KeyError, FileNotFoundError, ImportError) as exc:
        log.info("Confound analysis skipped: %s", exc)
        results["confound_analysis"] = "skipped"

    # Write JSON output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    log.info("Wrote %s", out_path)

    prov_write(
        out_path,
        stage="24_faithfulness",
        inputs={
            "attribution": str(attr_path),
            "scan": str(scan_path),
        },
        params={
            "dataset": args.dataset,
            "chain": args.chain,
            "score": args.score,
        },
        arm={
            "stage": "faithfulness",
            "dataset": args.dataset,
            "score": args.score,
            "method": method,
        },
    )


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
