#!/usr/bin/env python3
"""Fetch PimA ground truth from the Platinum database.

Downloads the Platinum flat file CSV, extracts PimA mannosyltransferase
(C. glutamicum, UniProt Q8NNK8) for ligands GDD (3OKP) and GDP (3OKC),
and writes ground truth CSVs to ``data/processed/``.

Response: log10(k_mt) - log10(k_wt), positive = resistance (affinity loss).

Guards:
  - WT residue identity at all measured positions against PDB structures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import download_rcsb
from igv.platinum import (
    PLATINUM_CACHE_NAME,
    download_platinum,
    extract_target,
    load_platinum,
    verify_wt_residues,
)
from igv.provenance import write as prov_write

PIMA_TARGETS = [
    {
        "name": "PimA-GDD",
        "uniprot": "Q8NNK8",
        "lig_id": "GDD",
        "pdb_id": "3OKP",
        "out_name": "pima_gdd_ground_truth.csv",
    },
    {
        "name": "PimA-GDP",
        "uniprot": "Q8NNK8",
        "lig_id": "GDP",
        "pdb_id": "3OKC",
        "out_name": "pima_gdp_ground_truth.csv",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PimA ground truth from Platinum")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download Platinum CSV
    plat_path = download_platinum(cache_dir)
    print(f"  dataset    {PLATINUM_CACHE_NAME:30s} -> {plat_path}  "
          f"({plat_path.stat().st_size / 1024:.1f} KB)")

    all_rows = load_platinum(cache_dir)
    print(f"  Total Platinum rows: {len(all_rows)}")

    for target in PIMA_TARGETS:
        print(f"\n{'='*60}")
        print(f"  Target: {target['name']}")
        print(f"{'='*60}")

        # Download PDB
        pdb_path = download_rcsb(target["pdb_id"], cache_dir)
        print(f"  structure  {target['pdb_id']:30s} -> {pdb_path}  "
              f"({pdb_path.stat().st_size / 1024:.1f} KB)")

        # Extract target
        df = extract_target(all_rows, target["uniprot"], target["lig_id"])
        n_mutations = len(df)
        n_positions = df["position"].nunique()
        print(f"\n  Extracted: {n_mutations} mutations, {n_positions} positions")

        # WT residue guard
        n_match, n_total, mismatches = verify_wt_residues(df, pdb_path)
        print(f"  WT residue guard ({target['pdb_id']}): {n_match}/{n_total}")
        if mismatches:
            for mm in mismatches:
                print(f"    MISMATCH: {mm}")
            raise SystemExit(f"WT residue guard FAILED for {target['name']}")
        print("    PASSED")

        # Write ground truth
        out_csv = out_dir / target["out_name"]
        df.to_csv(out_csv, index=False)
        print(f"\n  Wrote {out_csv} ({len(df)} rows)")

        # Summary
        print("\n  Response (log10 ratio):")
        print(f"    min:    {df['response_log10'].min():+.4f}")
        print(f"    max:    {df['response_log10'].max():+.4f}")
        print(f"    median: {df['response_log10'].median():+.4f}")

        censored = df["censored"].sum()
        if censored:
            print(f"    censored (right): {censored}")

        prov_write(
            out_csv,
            stage="22_fetch_pima",
            inputs={
                "dataset": "platinum",
                "target": target["name"],
                "source_doi": "10.1093/nar/gku966",
                "source_pmc": "PMC4384026",
            },
            params={
                "uniprot": target["uniprot"],
                "lig_id": target["lig_id"],
                "pdb_id": target["pdb_id"],
                "n_rows": n_mutations,
                "n_positions": n_positions,
                "sign_convention": (
                    "log10(Kd_mut / Kd_wt), positive = weaker binding = resistance; "
                    "Platinum's k_mt and k_wt are raw dissociation constants"
                ),
                "log_base": "log10",
                "wt_residue_guard": f"{n_match}/{n_total}",
            },
            arm={"stage": "fetch", "target": target["name"]},
        )

    # Manifest
    manifest_path = cache_dir / "pima_fetch_manifest.json"
    manifest = {
        "source": "Platinum database, doi:10.1093/nar/gku966",
        "platinum_csv": str(plat_path),
        "targets": [
            {
                "name": t["name"],
                "pdb": t["pdb_id"],
                "lig": t["lig_id"],
                "out": str(out_dir / t["out_name"]),
            }
            for t in PIMA_TARGETS
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    prov_write(
        manifest_path,
        stage="22_fetch_pima",
        inputs={"dataset": "platinum"},
        params={"cache_dir": str(cache_dir)},
        arm={"stage": "fetch"},
    )


if __name__ == "__main__":
    main()
