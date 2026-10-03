#!/usr/bin/env python3
"""Fetch the Hauser et al. 2018 ABL1 resistance dataset and co-crystal PDBs.

Downloads Supplementary Data 3 (XLSX) from the authors' GitHub, converts to
CSV, and fetches the 6 ABL co-crystal PDB structures from RCSB. All files
land under ``data/raw/`` (gitignored) and are re-fetchable.

Requires openpyxl at fetch time for XLSX parsing (``pip install openpyxl``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.abl import (
    ABL_TKIS,
    HAUSER2018_CACHE_NAME,
    TKI_PDB,
    download_hauser2018,
    load_hauser2018,
    summary,
)
from igv.data import download_rcsb
from igv.provenance import write as prov_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Hauser 2018 ABL1 resistance data")
    parser.add_argument("--cache-dir", default="data/raw", help="Directory to cache downloads")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary after fetch")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    csv_path = download_hauser2018(cache_dir)
    size_kb = csv_path.stat().st_size / 1024
    print(f"  dataset    {HAUSER2018_CACHE_NAME:30s} -> {csv_path}  ({size_kb:.1f} KB)")

    fetched = [("dataset", HAUSER2018_CACHE_NAME, str(csv_path), f"{size_kb:.1f} KB")]

    for tki in sorted(ABL_TKIS):
        pdb_id = TKI_PDB[tki]
        pdb_path = download_rcsb(pdb_id, cache_dir)
        pdb_size = pdb_path.stat().st_size / 1024
        fetched.append(("structure", pdb_id, str(pdb_path), f"{pdb_size:.1f} KB"))
        print(f"  structure  {pdb_id:30s} -> {pdb_path}  ({pdb_size:.1f} KB)")

    print(f"\nFetched {len(fetched)} files to {cache_dir}")

    manifest = cache_dir / "abl_fetch_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump(
            [{"kind": k, "name": n, "path": p, "size": s} for k, n, p, s in fetched],
            f,
            indent=2,
        )

    prov_write(
        manifest,
        stage="17_fetch_abl",
        inputs={"dataset": "hauser2018_abl"},
        params={"cache_dir": str(cache_dir)},
        arm={"stage": "fetch"},
    )

    if args.summary:
        df = load_hauser2018(cache_dir)
        stats = summary(df)
        print("\n--- Hauser 2018 ABL1 resistance dataset summary ---")
        print(f"Total rows: {stats['total_rows']}")
        print(f"Measured (IC50 + DDG_exp): {stats['total_measured']}")
        print(f"  ABL TKIs: {stats['abl_measured']}")
        print(f"  EGFR controls: {stats['egfr_measured']}")
        print(f"Distinct mutations: {stats['n_distinct_mutations']}")
        print(f"Distinct positions: {stats['n_distinct_positions']}")
        print("\nPer-inhibitor counts:")
        for tki, count in stats["per_tki"].items():
            print(f"  {tki:12s}: {count}")
        print("\nDDG_exp distribution (kcal/mol):")
        print(f"  mean:   {stats['ddg_mean']:+.3f}")
        print(f"  std:    {stats['ddg_std']:.3f}")
        print(f"  min:    {stats['ddg_min']:+.3f}")
        print(f"  max:    {stats['ddg_max']:+.3f}")
        print(f"  median: {stats['ddg_median']:+.3f}")
        print(f"  resistance (>0): {stats['n_resistance']}")
        print(f"  sensitising (<0): {stats['n_sensitising']}")
        print(f"  neutral (=0): {stats['n_neutral']}")


if __name__ == "__main__":
    main()
