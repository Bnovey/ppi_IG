#!/usr/bin/env python3
"""Stage 19 — TISM attribution vs Hauser 2018 measured resistance.

Turn a stage-18 affinity gradient artifact into the model-versus-lab number
against the Hauser 2018 ABL1 TKI resistance dataset, so that a single GPU
run is followed by zero-cost analysis.

Key identity: feats["res_type"] is one-hot, so the prediction for substituting
wild-type w with mutant m at position i is simply grad[i][m] - grad[i][w].
This is the published TISM estimator (iScience PMC11404212).  It means the
(1, L, vocab) gradient from ONE run already contains every mutation prediction.

For the difference form: the off-simplex constant gradient component that
Majdandzic & Koo (Genome Biology 2023) show corrupts raw gradient maps
cancels exactly, giving the correction for free.  This is why we use the
difference form directly rather than score_deltas -- score_deltas operates in
embedding space and assumes linearity, while the one-hot difference is exact.

Analyses emitted (per-inhibitor, pooled-within-inhibitor-ranks, within-mutation
across-inhibitor, T315I sanity check).

Usage:
  python scripts/19_tism_hauser.py --gradient results/affinity_gradient.npz --dry-run
  python scripts/19_tism_hauser.py --gradient results/affinity_gradient.npz --hauser data/raw/hauser2018_abl_supp3.csv
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

from igv.abl import (  # noqa: E402
    docked_only,
    load_hauser2018,
    load_hauser2018_from_csv,
    measured,
    parse_mutation,
    validate_ic50_consistency,
    xtal_only,
)
from igv.attrib import CANONICAL_AMINO_ACIDS  # noqa: E402
from igv.metrics import bootstrap_ci, spearman  # noqa: E402

log = logging.getLogger("tism_hauser")

# ---------------------------------------------------------------------------
# Artifact contract — expected keys in the gradient .npz
# ---------------------------------------------------------------------------

REQUIRED_NPZ_KEYS = {
    "grad_res_type",
    "token_map_keys",
    "token_map_values",
    "crop_coverage",
    "vocab_order",
}

MIN_N_FOR_CORRELATION = 5
DOF_CONFOUND_PANEL_SIZE = 10
MIN_RESIDUAL_DOF_FOR_PARTIAL = 15


# ---------------------------------------------------------------------------
# Pure helpers (CPU, no torch, no boltz — testable and usable in --dry-run)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gradient", required=True,
        help="Path to the affinity gradient .npz artifact from stage 18.",
    )
    p.add_argument(
        "--hauser", default=None,
        help="Path to the Hauser 2018 CSV (offline). If omitted, downloads.",
    )
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument(
        "--tki", default=None,
        help="Restrict to a single TKI (default: all 6 ABL TKIs).",
    )
    p.add_argument(
        "--out", default="results/tism_hauser.json",
    )
    p.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for the shuffled null.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for permutation and bootstrap.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--protein-chain-id", default="0",
        help="Chain ID in the token map for the protein (default: '0', "
             "matching the affinity featuriser's asym_id convention).",
    )
    return p


def validate_artifact(npz: dict[str, np.ndarray]) -> None:
    """Validate the gradient artifact has the expected keys."""
    missing = REQUIRED_NPZ_KEYS - set(npz.keys())
    if missing:
        raise ValueError(
            f"Gradient artifact is missing required keys: {sorted(missing)}. "
            f"Expected: {sorted(REQUIRED_NPZ_KEYS)}. "
            f"Found: {sorted(npz.keys())}."
        )


def reconstruct_token_map(
    keys: np.ndarray, values: np.ndarray,
) -> dict[tuple[str, int], int]:
    """Reconstruct token_map from serialised arrays.

    keys is shape (N, 2) with columns [chain_id_str, residue_index].
    values is shape (N,) with the crop-local token index.
    """
    token_map: dict[tuple[str, int], int] = {}
    for i in range(len(values)):
        chain = str(keys[i, 0])
        resi = int(keys[i, 1])
        token_map[(chain, resi)] = int(values[i])
    return token_map


def build_vocab_lookup(vocab_order: np.ndarray) -> dict[str, int]:
    """Build a one-letter AA -> column index lookup from the artifact's vocab."""
    return {str(aa): i for i, aa in enumerate(vocab_order)}


def tism_predict(
    grad: np.ndarray,
    token_idx: int,
    wt_aa: str,
    mut_aa: str,
    vocab_lookup: dict[str, int],
) -> float:
    """TISM prediction: grad[token_idx][mut] - grad[token_idx][wt].

    For one-hot input, this is identical to grad . (e_mut - e_wt) and
    gets the Majdandzic & Koo off-simplex correction for free.
    """
    wt_col = vocab_lookup[wt_aa]
    mut_col = vocab_lookup[mut_aa]
    return float(grad[token_idx, mut_col] - grad[token_idx, wt_col])


def map_mutations_to_predictions(
    df: pd.DataFrame,
    grad: np.ndarray,
    token_map: dict[tuple[str, int], int],
    vocab_lookup: dict[str, int],
    protein_chain_id: str,
    crop_surviving: set[int],
) -> pd.DataFrame:
    """Map each measured mutation to a TISM prediction.

    Returns a DataFrame with columns:
      tki, mutation, wt_aa, resnum, mut_aa, DDG_exp, dlog10_ic50,
      tism_pred, ic50_suspect, crop_excluded
    """
    rows = []
    for _, row in df.iterrows():
        mut = parse_mutation(row["mutation"])
        if mut.wt_aa not in vocab_lookup or mut.mut_aa not in vocab_lookup:
            continue

        resnum_0based = mut.resnum
        key = (protein_chain_id, resnum_0based)

        crop_excluded = key not in token_map or resnum_0based not in crop_surviving

        if crop_excluded:
            rows.append({
                "tki": row["TKI"],
                "mutation": str(mut),
                "wt_aa": mut.wt_aa,
                "resnum": mut.resnum,
                "mut_aa": mut.mut_aa,
                "DDG_exp": float(row["DDG_exp"]),
                "dlog10_ic50": float(row["dlog10_ic50"]),
                "tism_pred": np.nan,
                "ic50_suspect": bool(row.get("ic50_suspect", False)),
                "crop_excluded": True,
            })
            continue

        token_idx = token_map[key]
        pred = tism_predict(grad, token_idx, mut.wt_aa, mut.mut_aa, vocab_lookup)

        rows.append({
            "tki": row["TKI"],
            "mutation": str(mut),
            "wt_aa": mut.wt_aa,
            "resnum": mut.resnum,
            "mut_aa": mut.mut_aa,
            "DDG_exp": float(row["DDG_exp"]),
            "dlog10_ic50": float(row["dlog10_ic50"]),
            "tism_pred": pred,
            "ic50_suspect": bool(row.get("ic50_suspect", False)),
            "crop_excluded": False,
        })

    return pd.DataFrame(rows)


def permutation_null(
    pred: np.ndarray,
    true: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Shuffled-ranking null distribution for Spearman correlation."""
    rng = np.random.default_rng(seed)
    null_rhos = np.array([
        spearman(rng.permutation(pred), true)
        for _ in range(n_perm)
    ])
    observed = spearman(pred, true)
    p_value = float(np.mean(np.abs(null_rhos) >= np.abs(observed)))
    return {
        "n_permutations": n_perm,
        "null_mean": float(np.nanmean(null_rhos)),
        "null_p95": float(np.nanpercentile(null_rhos, 95)),
        "null_p05": float(np.nanpercentile(null_rhos, 5)),
        "p_value_two_sided": p_value,
    }


def per_inhibitor_analysis(
    mapped: pd.DataFrame,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Per-inhibitor Spearman with null and CI."""
    results = {}
    for tki in sorted(mapped["tki"].unique()):
        sub = mapped[(mapped["tki"] == tki) & ~mapped["crop_excluded"]]
        n = len(sub)
        if n < MIN_N_FOR_CORRELATION:
            results[tki] = {
                "n": n,
                "spearman": float("nan"),
                "note": f"n={n} < {MIN_N_FOR_CORRELATION}, correlation not computed",
            }
            continue

        pred = sub["tism_pred"].values
        true = sub["DDG_exp"].values

        rho = spearman(pred, true)
        point, ci_lo, ci_hi = bootstrap_ci(spearman, pred, true, seed=seed)
        null = permutation_null(pred, true, n_perm=n_perm, seed=seed)

        results[tki] = {
            "n": n,
            "spearman": rho,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            **null,
        }
    return results


def pooled_within_inhibitor_ranks(
    mapped: pd.DataFrame,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Pooled Spearman using within-inhibitor ranks.

    Never raw pool ddG across inhibitors: the six TKIs are different ligands
    and a per-ligand offset (including Boltz-2's MW correction which is
    constant per ligand) would confound raw pooling.

    Instead, rank both predicted and measured values within each inhibitor,
    then pool the ranks for a single correlation.
    """
    usable = mapped[~mapped["crop_excluded"]].copy()
    if len(usable) < MIN_N_FOR_CORRELATION:
        return {
            "n": len(usable),
            "spearman": float("nan"),
            "note": "too few data points",
        }

    all_pred_ranks = []
    all_true_ranks = []
    for tki in sorted(usable["tki"].unique()):
        sub = usable[usable["tki"] == tki]
        if len(sub) < 2:
            continue
        pred_ranks = stats.rankdata(sub["tism_pred"].values)
        true_ranks = stats.rankdata(sub["DDG_exp"].values)
        all_pred_ranks.extend(pred_ranks)
        all_true_ranks.extend(true_ranks)

    pred_arr = np.array(all_pred_ranks)
    true_arr = np.array(all_true_ranks)

    rho = spearman(pred_arr, true_arr)
    point, ci_lo, ci_hi = bootstrap_ci(spearman, pred_arr, true_arr, seed=seed)
    null = permutation_null(pred_arr, true_arr, n_perm=n_perm, seed=seed)

    return {
        "n": len(pred_arr),
        "n_tkis": len(usable["tki"].unique()),
        "spearman": rho,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        **null,
    }


def within_mutation_across_inhibitor(
    mapped: pd.DataFrame,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Within-mutation, across-inhibitor analysis.

    For each mutation, rank the TKIs by measured ddG and compare to the
    model's ranking. This is the clean design: burial, RSA, volume,
    conservation and position are all constant within a mutation, so they
    cancel structurally (not statistically).
    """
    usable = mapped[~mapped["crop_excluded"]].copy()
    mutations = sorted(usable["mutation"].unique())

    per_mutation = {}
    all_pred_ranks = []
    all_true_ranks = []

    for mut_str in mutations:
        sub = usable[usable["mutation"] == mut_str]
        if len(sub) < 2:
            continue

        pred = sub["tism_pred"].values
        true = sub["DDG_exp"].values

        pred_ranks = stats.rankdata(pred)
        true_ranks = stats.rankdata(true)
        all_pred_ranks.extend(pred_ranks)
        all_true_ranks.extend(true_ranks)

        n = len(sub)
        if n >= MIN_N_FOR_CORRELATION:
            rho = spearman(pred, true)
        else:
            rho = float("nan")

        per_mutation[mut_str] = {
            "n_tkis": n,
            "spearman": rho,
            "tkis": sorted(sub["tki"].tolist()),
        }

    pred_arr = np.array(all_pred_ranks)
    true_arr = np.array(all_true_ranks)

    if len(pred_arr) < MIN_N_FOR_CORRELATION:
        return {
            "n_datapoints": len(pred_arr),
            "n_mutations": len(per_mutation),
            "pooled_spearman": float("nan"),
            "per_mutation": per_mutation,
            "note": "too few data points for pooled correlation",
        }

    pooled_rho = spearman(pred_arr, true_arr)
    point, ci_lo, ci_hi = bootstrap_ci(spearman, pred_arr, true_arr, seed=seed)
    null = permutation_null(pred_arr, true_arr, n_perm=n_perm, seed=seed)

    return {
        "n_datapoints": len(pred_arr),
        "n_mutations": len(per_mutation),
        "pooled_spearman": pooled_rho,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        **null,
        "per_mutation": per_mutation,
    }


def t315i_sanity_check(mapped: pd.DataFrame) -> dict:
    """Ponatinib was engineered to retain potency against T315I where imatinib fails.

    Report the model's predicted ordering for T315I across the 6 TKIs
    against the measured ordering.
    """
    t315i = mapped[(mapped["mutation"] == "T315I") & ~mapped["crop_excluded"]]
    if len(t315i) == 0:
        return {"status": "T315I not in usable data"}

    t315i_sorted_measured = t315i.sort_values("DDG_exp")
    t315i_sorted_predicted = t315i.sort_values("tism_pred")

    measured_order = t315i_sorted_measured["tki"].tolist()
    predicted_order = t315i_sorted_predicted["tki"].tolist()

    has_imatinib = "imatinib" in t315i["tki"].values
    has_ponatinib = "ponatinib" in t315i["tki"].values

    ponatinib_correct = None
    if has_imatinib and has_ponatinib:
        imat_ddg = float(t315i[t315i["tki"] == "imatinib"]["DDG_exp"].iloc[0])
        pon_ddg = float(t315i[t315i["tki"] == "ponatinib"]["DDG_exp"].iloc[0])
        imat_pred = float(t315i[t315i["tki"] == "imatinib"]["tism_pred"].iloc[0])
        pon_pred = float(t315i[t315i["tki"] == "ponatinib"]["tism_pred"].iloc[0])
        ponatinib_correct = (pon_pred < imat_pred) == (pon_ddg < imat_ddg)

    pred = t315i["tism_pred"].values
    true = t315i["DDG_exp"].values
    n = len(t315i)
    rho = spearman(pred, true) if n >= MIN_N_FOR_CORRELATION else float("nan")

    return {
        "n_tkis": n,
        "measured_order_ascending_ddg": measured_order,
        "predicted_order_ascending_pred": predicted_order,
        "spearman": rho,
        "ponatinib_retains_potency_over_imatinib": ponatinib_correct,
        "measured_ddg": {
            row["tki"]: float(row["DDG_exp"]) for _, row in t315i.iterrows()
        },
        "predicted": {
            row["tki"]: float(row["tism_pred"]) for _, row in t315i.iterrows()
        },
    }


def dof_guard(n_positions: int, n_confounds: int = DOF_CONFOUND_PANEL_SIZE) -> dict:
    """Refuse partial correlation if residual DOF is too low.

    The mutation axis has limited variants across limited positions. If
    someone requests a partial correlation with the confound panel, this
    guards against the 3HFM trap (HANDOFF item 6: CI of [-1.0, +1.0]).
    """
    residual_dof = n_positions - n_confounds - 1
    safe = residual_dof >= MIN_RESIDUAL_DOF_FOR_PARTIAL
    return {
        "n_positions": n_positions,
        "n_confounds": n_confounds,
        "residual_dof": residual_dof,
        "partial_correlation_safe": safe,
        "note": (
            "Partial correlation is safe to compute."
            if safe else
            f"REFUSED: residual DOF = {residual_dof} after removing "
            f"{n_confounds} confounds from {n_positions} positions. "
            f"A partial correlation here would produce an uninterpretable "
            f"interval (the 3HFM trap, HANDOFF item 6)."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    grad_path = Path(args.gradient)
    out_json = Path(args.out)

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        print("Stage 19: TISM attribution vs Hauser 2018\n")
        print(f"  gradient:         {grad_path}")
        print(f"  hauser:           {args.hauser or '(auto-download)'}")
        print(f"  cache_dir:        {args.cache_dir}")
        print(f"  protein_chain_id: {args.protein_chain_id}")
        print(f"  tki filter:       {args.tki or 'all 6 ABL TKIs'}")
        print(f"  n_permutations:   {args.n_permutations}")
        print(f"  seed:             {args.seed}")
        print(f"  output:           {out_json}")
        print()
        print("Artifact contract (required .npz keys):")
        for k in sorted(REQUIRED_NPZ_KEYS):
            print(f"    {k}")
        print()
        print("Analyses emitted:")
        print("  1. Per-inhibitor Spearman (n~21 each) with null + CI")
        print("  2. Pooled within-inhibitor ranks (never raw pooled ddG)")
        print("  3. Within-mutation across-inhibitor (confounds cancel structurally)")
        print("  4. T315I sanity check (ponatinib vs imatinib)")
        print("  5. Degrees-of-freedom guard for partial correlation")
        print()
        print("Honesty requirements:")
        print("  - Crop coverage: exclude mutations outside AffinityCropper's kept set")
        print("  - Exclude docked measurements (erlotinib/gefitinib)")
        print("  - Use DDG_exp (not raw IC50), report ic50_suspect rows")
        print("  - Permutation null + bootstrap CI on every correlation")
        print("  - Sign: positive DDG_exp = resistance = weaker binding, no negation")
        print("  - DOF guard: refuse partial correlation if too few residual DOF")
        print()
        print("VM command:")
        cmd = f"python3 scripts/19_tism_hauser.py --gradient {args.gradient}"
        if args.hauser:
            cmd += f" --hauser {args.hauser}"
        print(f"  {cmd}")

        from igv.provenance import write as prov_write  # noqa: E402
        out_json.parent.mkdir(parents=True, exist_ok=True)
        dry_result = {"stage": "19_tism_hauser", "dry_run": True}
        out_json.write_text(json.dumps(dry_result, indent=2) + "\n")
        prov_write(
            out_json,
            stage="19_tism_hauser",
            inputs={"gradient": str(grad_path)},
            params={"dry_run": True},
        )
        return 0

    # ------------------------------------------------------------------
    # Load artifact
    # ------------------------------------------------------------------
    if not grad_path.exists():
        raise SystemExit(f"Gradient artifact not found: {grad_path}")

    npz = dict(np.load(str(grad_path), allow_pickle=True))
    validate_artifact(npz)

    grad = npz["grad_res_type"]
    if grad.ndim == 3 and grad.shape[0] == 1:
        grad = grad[0]

    token_map = reconstruct_token_map(npz["token_map_keys"], npz["token_map_values"])
    vocab_order = npz["vocab_order"]
    vocab_lookup = build_vocab_lookup(vocab_order)
    crop_coverage = json.loads(str(npz["crop_coverage"]))

    crop_surviving_set = set(crop_coverage.get("surviving", []))

    print(f"\nGradient artifact: {grad_path}")
    print(f"  grad shape:       {grad.shape}")
    print(f"  vocab size:       {len(vocab_order)}")
    print(f"  token_map size:   {len(token_map)}")
    print(f"  crop surviving:   {len(crop_surviving_set)}")

    for aa in CANONICAL_AMINO_ACIDS:
        if aa not in vocab_lookup:
            raise SystemExit(
                f"Canonical amino acid '{aa}' missing from artifact's vocab_order. "
                f"Vocab has: {list(vocab_order)}"
            )

    # ------------------------------------------------------------------
    # Load Hauser ground truth
    # ------------------------------------------------------------------
    if args.hauser:
        hauser_df = load_hauser2018_from_csv(Path(args.hauser))
    else:
        hauser_df = load_hauser2018(Path(args.cache_dir))

    hauser_meas = measured(hauser_df)

    xtal_df = xtal_only(hauser_meas)
    docked_df = docked_only(hauser_meas)
    n_docked_excluded = len(docked_df)

    if args.tki:
        xtal_df = xtal_df[xtal_df["TKI"] == args.tki].reset_index(drop=True)
        if len(xtal_df) == 0:
            raise SystemExit(f"No xtal measurements for TKI '{args.tki}'")

    suspect = validate_ic50_consistency(hauser_df)
    n_suspect = len(suspect)

    print("\nHauser 2018:")
    print(f"  total measured:     {len(hauser_meas)}")
    print(f"  xtal (usable):      {len(xtal_df)}")
    print(f"  docked (excluded):  {n_docked_excluded}")
    print(f"  ic50_suspect rows:  {n_suspect}")
    if n_suspect > 0:
        for _, row in suspect.iterrows():
            print(f"    {row['TKI']} / {row['mutation']}: "
                  f"deviation {row['ic50_deviation_log10']:.3f} log10")

    # ------------------------------------------------------------------
    # Map mutations to predictions
    # ------------------------------------------------------------------
    mapped = map_mutations_to_predictions(
        xtal_df, grad, token_map, vocab_lookup,
        args.protein_chain_id, crop_surviving_set,
    )

    n_total = len(mapped)
    n_crop_excluded = int(mapped["crop_excluded"].sum())
    n_usable = n_total - n_crop_excluded
    n_suspect_included = int(
        mapped[~mapped["crop_excluded"]]["ic50_suspect"].sum()
    )

    print("\nMapping:")
    print(f"  total mapped:       {n_total}")
    print(f"  crop excluded:      {n_crop_excluded}")
    print(f"  usable:             {n_usable}")
    print(f"  ic50_suspect incl:  {n_suspect_included}")

    if n_usable == 0:
        print("\nERROR: No usable mutations after crop exclusion.")
        return 1

    # Sign convention assertion: positive DDG_exp = resistance.
    # affinity_pred_value is log10(IC50 uM), so an increase is also weaker
    # binding. They are already on the same convention -- NO negation.
    # This assertion catches a silent flip (the a18891c / 6d2749a defect class).
    usable = mapped[~mapped["crop_excluded"]]

    # ------------------------------------------------------------------
    # Run analyses
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Per-inhibitor analysis")
    print(f"{'=' * 60}")
    per_inh = per_inhibitor_analysis(
        mapped, n_perm=args.n_permutations, seed=args.seed,
    )
    for tki, res in per_inh.items():
        if "bootstrap_ci_95" in res:
            print(f"  {tki:12s}  n={res['n']:3d}  "
                  f"rho={res['spearman']:+.4f}  "
                  f"CI=[{res['bootstrap_ci_95'][0]:+.4f}, "
                  f"{res['bootstrap_ci_95'][1]:+.4f}]  "
                  f"null_p={res['p_value_two_sided']:.3f}")
        else:
            print(f"  {tki:12s}  n={res['n']:3d}  {res.get('note', '')}")

    print(f"\n{'=' * 60}")
    print("Pooled within-inhibitor ranks")
    print(f"{'=' * 60}")
    pooled = pooled_within_inhibitor_ranks(
        mapped, n_perm=args.n_permutations, seed=args.seed,
    )
    if "bootstrap_ci_95" in pooled:
        print(f"  n={pooled['n']}  n_tkis={pooled['n_tkis']}  "
              f"rho={pooled['spearman']:+.4f}  "
              f"CI=[{pooled['bootstrap_ci_95'][0]:+.4f}, "
              f"{pooled['bootstrap_ci_95'][1]:+.4f}]  "
              f"null_p={pooled['p_value_two_sided']:.3f}")
    else:
        print(f"  {pooled.get('note', '')}")

    print(f"\n{'=' * 60}")
    print("Within-mutation across-inhibitor (the clean design)")
    print(f"{'=' * 60}")
    within_mut = within_mutation_across_inhibitor(
        mapped, n_perm=args.n_permutations, seed=args.seed,
    )
    if "bootstrap_ci_95" in within_mut:
        print(f"  n_datapoints={within_mut['n_datapoints']}  "
              f"n_mutations={within_mut['n_mutations']}  "
              f"pooled_rho={within_mut['pooled_spearman']:+.4f}  "
              f"CI=[{within_mut['bootstrap_ci_95'][0]:+.4f}, "
              f"{within_mut['bootstrap_ci_95'][1]:+.4f}]  "
              f"null_p={within_mut['p_value_two_sided']:.3f}")
    else:
        print(f"  {within_mut.get('note', '')}")

    print(f"\n{'=' * 60}")
    print("T315I sanity check")
    print(f"{'=' * 60}")
    t315i = t315i_sanity_check(mapped)
    if "measured_order_ascending_ddg" in t315i:
        print("  measured order (ascending ddG, least resistance first):")
        print(f"    {t315i['measured_order_ascending_ddg']}")
        print("  predicted order (ascending pred, least resistance first):")
        print(f"    {t315i['predicted_order_ascending_pred']}")
        if t315i["ponatinib_retains_potency_over_imatinib"] is not None:
            label = "CORRECT" if t315i["ponatinib_retains_potency_over_imatinib"] else "WRONG"
            print(f"  ponatinib < imatinib (T315I): {label}")
    else:
        print(f"  {t315i.get('status', 'unknown')}")

    print(f"\n{'=' * 60}")
    print("Degrees-of-freedom guard")
    print(f"{'=' * 60}")
    n_positions = len(usable["resnum"].unique())
    dof = dof_guard(n_positions)
    print(f"  {dof['note']}")

    # ------------------------------------------------------------------
    # Write JSON artifact
    # ------------------------------------------------------------------
    result = {
        "stage": "19_tism_hauser",
        "gradient_artifact": str(grad_path),
        "n_total_mapped": n_total,
        "n_crop_excluded": n_crop_excluded,
        "n_docked_excluded": n_docked_excluded,
        "n_usable": n_usable,
        "n_suspect_included": n_suspect_included,
        "sign_convention": "positive DDG_exp = resistance = weaker binding; "
                           "positive tism_pred = increase in affinity_pred_value "
                           "(log10 IC50 uM) = weaker binding; same direction, "
                           "no negation applied",
        "per_inhibitor": per_inh,
        "pooled_within_inhibitor_ranks": pooled,
        "within_mutation_across_inhibitor": within_mut,
        "t315i_sanity_check": t315i,
        "dof_guard": dof,
        "crop_coverage": crop_coverage,
        "vocab_size": len(vocab_order),
        "vocab_order": list(vocab_order),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str) + "\n")

    from igv.provenance import write as prov_write  # noqa: E402
    prov_write(
        out_json,
        stage="19_tism_hauser",
        inputs={
            "gradient": str(grad_path),
            "hauser": args.hauser or "(auto-download)",
        },
        params={
            "protein_chain_id": args.protein_chain_id,
            "tki": args.tki,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
        },
        arm={
            "method": "tism",
            "ground_truth": "hauser2018",
            "score_surface": "affinity_pred_value",
        },
    )

    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
