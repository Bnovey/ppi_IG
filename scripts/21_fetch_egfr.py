#!/usr/bin/env python3
"""Fetch the Robichaux 2021 EGFR resistance dataset and 1M17 structure.

Downloads Supplementary Table 4 (XLSX) from Springer, parses Panel A to
extract 27 single-missense substitutions across 20 positions and 18 TKIs,
and writes a tidy CSV to ``data/processed/egfr_ground_truth.csv``.

Guards:
  - WT residue identity at all 20 positions against 1M17 (offset -24)
  - Sign convention anchored on T790M (erlotinib +1.56, osimertinib -0.76)

Requires openpyxl at fetch time for XLSX parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import download_rcsb
from igv.egfr import (
    EGFR_1M17_OFFSET,
    ROBICHAUX_CACHE_NAME,
    download_robichaux2021,
    load_egfr_ground_truth,
    verify_sign_convention,
    verify_wt_residues_1m17,
)
from igv.provenance import write as prov_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EGFR resistance data")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download XLSX
    xlsx_path = download_robichaux2021(cache_dir)
    print(f"  dataset    {ROBICHAUX_CACHE_NAME:30s} -> {xlsx_path}  "
          f"({xlsx_path.stat().st_size / 1024:.1f} KB)")

    # Download 1M17 PDB
    pdb_path = download_rcsb("1M17", cache_dir)
    print(f"  structure  {'1M17':30s} -> {pdb_path}  "
          f"({pdb_path.stat().st_size / 1024:.1f} KB)")

    # Parse Panel A
    df = load_egfr_ground_truth(cache_dir)
    n_singles = df["mutation"].nunique()
    n_drugs = df["drug"].nunique()
    n_positions = df["position"].nunique()
    n_cells = len(df)
    print(f"\n  Parsed: {n_singles} single mutations, {n_positions} positions, "
          f"{n_drugs} drugs, {n_cells} cells")

    # Guard 1: WT residue identity
    mutations = sorted(df["mutation"].unique())
    n_match, n_total, mismatches = verify_wt_residues_1m17(mutations, pdb_path)
    print(f"\n  WT residue guard (1M17, offset {EGFR_1M17_OFFSET}): "
          f"{n_match}/{n_total}")
    if mismatches:
        for mm in mismatches:
            print(f"    MISMATCH: {mm}")
        raise SystemExit("WT residue guard FAILED")
    print("    PASSED")

    # Guard 2: sign convention
    sign_ok, sign_problems = verify_sign_convention(df)
    print(f"\n  Sign convention guard: {'PASSED' if sign_ok else 'FAILED'}")
    if not sign_ok:
        for p in sign_problems:
            print(f"    {p}")
        raise SystemExit("Sign convention guard FAILED")

    # Write ground truth CSV
    out_csv = out_dir / "egfr_ground_truth.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Wrote {out_csv} ({len(df)} rows)")

    # Noise floor
    median_sd = df["replicate_sd"].median()
    mean_sd = df["replicate_sd"].mean()
    p90_sd = df["replicate_sd"].quantile(0.9)
    response_range = (df["mean_response"].min(), df["mean_response"].max())
    print("\n  Noise floor (replicate SD, log units):")
    print(f"    median: {median_sd:.3f}")
    print(f"    mean:   {mean_sd:.3f}")
    print(f"    90th %%: {p90_sd:.3f}")
    print(f"    response range: {response_range[0]:.2f} to {response_range[1]:.2f}")

    # Manifest
    manifest_path = cache_dir / "egfr_fetch_manifest.json"
    xlsx_sha256 = hashlib.sha256(xlsx_path.read_bytes()).hexdigest()
    manifest = {
        "source": "Robichaux et al. 2021, doi:10.1038/s41586-021-03898-1",
        "xlsx": str(xlsx_path),
        "pdb": str(pdb_path),
        "pdb_id": "1M17",
        "output": str(out_csv),
        "n_singles": n_singles,
        "n_positions": n_positions,
        "n_drugs": n_drugs,
        "n_cells": n_cells,
        "offset": EGFR_1M17_OFFSET,
        "wt_residue_guard": f"{n_match}/{n_total}",
        "sign_guard": "PASSED",
        "log_base": "unknown",
        "median_replicate_sd": round(median_sd, 4),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    prov_write(
        out_csv,
        stage="21_fetch_egfr",
        inputs={
            "dataset": "robichaux2021_egfr",
            "source_doi": "10.1038/s41586-021-03898-1",
            "source_pmid": "34526717",
            "source_pmc": "PMC8481125",
            "xlsx_sha256": xlsx_sha256,
        },
        params={
            "cache_dir": str(cache_dir),
            "offset": EGFR_1M17_OFFSET,
            "n_rows": n_cells,
            "n_mutations": n_singles,
            "n_positions": n_positions,
            "n_drugs": n_drugs,
            "sign_convention": (
                "log(IC50_mut / IC50_WT), positive = resistance; "
                "anchored on T790M: erlotinib +1.56, osimertinib -0.76"
            ),
            "log_base": "unknown",
            "median_replicate_sd": round(median_sd, 4),
            "structure": "1M17 (monomer, chain A, offset -24)",
            "ligands": {"erlotinib": "AQ4", "gefitinib": "IRE"},
            "wt_residue_guard": f"{n_match}/{n_total}",
        },
        arm={"stage": "fetch", "target": "EGFR"},
    )

    if args.summary:
        print("\n--- Per-drug summary ---")
        for drug in sorted(df["drug"].unique()):
            sub = df[df["drug"] == drug]
            print(f"  {drug:15s}: {len(sub)} mutations, "
                  f"response {sub['mean_response'].min():+.2f} to "
                  f"{sub['mean_response'].max():+.2f}")


if __name__ == "__main__":
    main()
