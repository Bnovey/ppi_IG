#!/usr/bin/env python3
"""Does the resswap/refeaturise decorrelation reproduce?

Section 24's most surprising negative: on 3S9D, over the 103 mutations both
scans scored, the one-hot swap and the fully re-featurised mutant were
*uncorrelated* (Spearman -0.0744), and the refeaturise delta was 36x larger.
That bounds the headline, so it is the finding that most needs replication.

Usage:
    python3 scripts/25_decorrelation.py 3S9D 1JTG 2JEL 3HFM
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RESULTS = Path("results")


def one(ds: str) -> dict | None:
    a = RESULTS / f"{ds}_complex_pde_scan.csv"
    b = RESULTS / f"{ds}_complex_pde_scan_refeaturise.csv"
    if not (a.exists() and b.exists()):
        return None
    ra = pd.read_csv(a)
    rb = pd.read_csv(b)
    ra = ra[ra.mutation_id != "WT"]
    rb = rb[(rb.mutation_id != "WT") & (rb["mode"] == "refeaturise")]
    m = ra.merge(rb, on="mutation_id", suffixes=("_swap", "_ref"))
    if len(m) < 5:
        return {"dataset": ds, "n_overlap": len(m), "note": "too few to test"}
    x = m["delta_swap"].to_numpy(float)
    y = m["delta_ref"].to_numpy(float)
    return {
        "dataset": ds,
        "n_overlap": len(m),
        "spearman": spearmanr(x, y).statistic,
        "pearson": pearsonr(x, y)[0],
        "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
        "median_abs_diff": float(np.median(np.abs(x - y))),
        "max_abs_diff": float(np.max(np.abs(x - y))),
        "median_abs_swap": float(np.median(np.abs(x))),
        "median_abs_ref": float(np.median(np.abs(y))),
        "scale_ratio_ref_over_swap": float(
            np.median(np.abs(y)) / np.median(np.abs(x))
        ),
        "sd_ref_delta": float(np.std(y, ddof=1)),
    }


def main() -> None:
    names = sys.argv[1:] or ["3S9D", "1JTG", "2JEL", "3HFM"]
    rows = [r for ds in names if (r := one(ds)) is not None]
    if not rows:
        print("no complex has both scans yet")
        return
    print("| complex | n both scored | Spearman(swap, refeat) | Pearson | "
          "sign agree | median abs swap | median abs refeat | ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "note" in r:
            print(f"| {r['dataset']} | {r['n_overlap']} | {r['note']} "
                  "| -- | -- | -- | -- | -- |")
            continue
        print(
            f"| {r['dataset']} | {r['n_overlap']} | **{r['spearman']:.4f}** "
            f"| {r['pearson']:.4f} | {r['sign_agreement']:.3f} "
            f"| {r['median_abs_swap']:.4f} | {r['median_abs_ref']:.4f} "
            f"| **{r['scale_ratio_ref_over_swap']:.1f}x** |"
        )


if __name__ == "__main__":
    main()
