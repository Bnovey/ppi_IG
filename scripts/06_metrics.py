#!/usr/bin/env python3
"""Compute T1/T2/T3 metrics and baselines for attribution evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.metrics import spearman, stratify_by_n_mut, summary
from igv.provenance import assert_provenance, write as prov_write

PUBLISHED_BASELINES = {
    "ProteinMPNN": 0.30,
    "ESM-IF1": 0.28,
    "AntiFold": 0.21,
    "Boltz-2_scan": 0.13,
    "FoldX": 0.12,
    "AF3": -0.02,
}

KNOWN_HARD_DATASETS = {"1mlc", "1n8z"}


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("pred", "binding_score", "n_mut"):
        if col not in df.columns:
            sys.exit(f"ERROR: prediction CSV missing column {col!r}")
    return df


def _load_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("scan_score", "binding_score"):
        if col not in df.columns:
            sys.exit(f"ERROR: scan CSV missing column {col!r}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute three-term evaluation metrics")
    parser.add_argument("--pred", required=True, help="Prediction CSV from 05_predict")
    parser.add_argument("--scan", default=None, help="Optional scan CSV for T1/T2")
    parser.add_argument("--dataset", default="4fqi_h1", help="Dataset name for labelling")
    parser.add_argument("--method", default="plain_grad", help="Attribution method label")
    parser.add_argument("--out", default="results/metrics.csv", help="Output metrics CSV")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    out_path = Path(args.out)

    if not pred_path.exists():
        sys.exit(f"ERROR: prediction file not found: {pred_path}")

    assert_provenance(pred_path)

    pred_df = _load_pred(pred_path)
    pred = pred_df["pred"].values
    true = pred_df["binding_score"].values
    n_mut = pred_df["n_mut"].values

    rows = []

    t3_rho = spearman(pred, true)
    print(f"T3 (attribution vs experiment): Spearman = {t3_rho:.4f}")
    rows.append({
        "dataset": args.dataset,
        "method": args.method,
        "term": "T3",
        "score": "spearman",
        "value": t3_rho,
    })

    s = summary(pred, true, n_mut=n_mut)
    rows.append({
        "dataset": args.dataset,
        "method": args.method,
        "term": "T3",
        "score": "precision_at_10",
        "value": s["precision_at_10"],
    })

    if args.scan:
        scan_path = Path(args.scan)
        if not scan_path.exists():
            sys.exit(f"ERROR: scan file not found: {scan_path}")
        assert_provenance(scan_path)

        scan_df = _load_scan(scan_path)

        if "substitutions" in pred_df.columns and "substitutions" in scan_df.columns:
            merged = pred_df.merge(scan_df, on="substitutions", suffixes=("_pred", "_scan"))
        else:
            merged = pred_df.iloc[: len(scan_df)].copy()
            merged["scan_score"] = scan_df["scan_score"].values[: len(merged)]

        t1_rho = spearman(merged["pred"].values if "pred" in merged.columns
                          else merged["pred_pred"].values,
                          merged["scan_score"].values if "scan_score" in merged.columns
                          else merged["scan_score_scan"].values)
        print(f"T1 (attribution vs model scan): Spearman = {t1_rho:.4f}")
        rows.append({
            "dataset": args.dataset,
            "method": args.method,
            "term": "T1",
            "score": "spearman",
            "value": t1_rho,
        })

        scan_true = (merged["binding_score"].values if "binding_score" in merged.columns
                     else merged["binding_score_pred"].values)
        scan_scores = (merged["scan_score"].values if "scan_score" in merged.columns
                       else merged["scan_score_scan"].values)
        t2_rho = spearman(scan_scores, scan_true)
        print(f"T2 (model scan vs experiment):  Spearman = {t2_rho:.4f}")
        rows.append({
            "dataset": args.dataset,
            "method": args.method,
            "term": "T2",
            "score": "spearman",
            "value": t2_rho,
        })

    strat = stratify_by_n_mut(pred, true, n_mut)
    print("\nStratified Spearman by n_mut:")
    for _, row in strat.iterrows():
        nm = int(row["n_mut"])
        print(f"  n_mut={nm:2d}: rho={row['metric']:.4f}  (n={int(row['count'])})")
        rows.append({
            "dataset": args.dataset,
            "method": args.method,
            "term": "T3_stratified",
            "score": f"spearman_nmut{nm}",
            "value": row["metric"],
        })

    for baseline_name, baseline_val in PUBLISHED_BASELINES.items():
        rows.append({
            "dataset": args.dataset,
            "method": baseline_name,
            "term": "baseline",
            "score": "spearman",
            "value": baseline_val,
        })

    new_df = pd.DataFrame(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"\nAppended {len(new_df)} rows to {out_path} (total {len(combined)})")
    else:
        new_df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path} ({len(new_df)} rows)")

    if args.dataset in KNOWN_HARD_DATASETS:
        print(
            f"\nNOTE: {args.dataset} is a known-hard dataset where all published "
            "methods score near zero or negative. Low scores here are expected."
        )

    print("\nBaseline comparison (published per-dataset Spearman, averaged):")
    for name, val in PUBLISHED_BASELINES.items():
        marker = " <-- line to beat" if name == "Boltz-2_scan" else ""
        print(f"  {name:15s}: {val:+.2f}{marker}")
    print(f"  {'Ours (T3)':15s}: {t3_rho:+.4f}")

    prov_write(
        out_path,
        stage="06_metrics",
        inputs={"pred": str(pred_path), "scan": str(args.scan) if args.scan else None},
        params={"dataset": args.dataset, "method": args.method},
        arm={"stage": "metrics", "dataset": args.dataset},
    )


if __name__ == "__main__":
    main()
