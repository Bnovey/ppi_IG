#!/usr/bin/env python3
"""Download AbBiBench affinity CSVs and structure PDBs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import _AFFINITY_NAMES, _STRUCTURE_NAMES, download
from igv.provenance import write as prov_write

ALL_AFFINITY = list(_AFFINITY_NAMES)

STRUCTURE_FOR_AFFINITY = {
    "1mhp_LC": "1mhp",
    "1mlc": "1mlc_bae",
    "1n8z": "1n8z_bac",
    "2fjg": "2fjg_hlv",
    "3gbn_h1": "3gbn_hlab",
    "3gbn_h9": "3gbn_hlab",
    "5a12_ang2": "4zfg_hla",
    "5a12_vegf": "4zfg_hla",
    "aayl49": "AAYL49_bca",
    "aayl49_ML": "AAYL49_bca",
    "aayl50_LC": "AAYL50_bca",
    "aayl51": "AAYL51_bca",
    "aayl52_LC": "AAYL52_bca",
    "g6_LC": "4zff_hld",
    "4fqi_h1": "4fqi_hlab",
    "4fqi_h3": "4fqi_hlab",
    "4d5_her2": "1mhp_hla",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AbBiBench data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--datasets", nargs="+", help="Dataset names to download")
    group.add_argument("--all", action="store_true", help="Download all 17 datasets")
    parser.add_argument(
        "--cache-dir", default="data/raw", help="Directory to cache downloads"
    )
    args = parser.parse_args()

    datasets = ALL_AFFINITY if args.all else args.datasets
    cache_dir = Path(args.cache_dir)

    fetched = []

    for name in datasets:
        if name not in ALL_AFFINITY:
            print(f"WARNING: unknown dataset {name!r}, skipping")
            continue

        aff_path = download(name, "affinity", cache_dir)
        size_kb = aff_path.stat().st_size / 1024
        fetched.append(("affinity", name, str(aff_path), f"{size_kb:.1f} KB"))
        print(f"  affinity  {name:20s} -> {aff_path}  ({size_kb:.1f} KB)")

        struct_name = STRUCTURE_FOR_AFFINITY.get(name)
        if struct_name and struct_name in _STRUCTURE_NAMES:
            pdb_path = download(struct_name, "structure", cache_dir)
            pdb_size = pdb_path.stat().st_size / 1024
            fetched.append(("structure", struct_name, str(pdb_path), f"{pdb_size:.1f} KB"))
            print(f"  structure {struct_name:20s} -> {pdb_path}  ({pdb_size:.1f} KB)")

    print(f"\nFetched {len(fetched)} files to {cache_dir}")

    manifest = cache_dir / "fetch_manifest.json"
    import json

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump(
            [{"kind": k, "name": n, "path": p, "size": s} for k, n, p, s in fetched],
            f,
            indent=2,
        )

    prov_write(
        manifest,
        stage="00_fetch_data",
        inputs={"datasets": datasets},
        params={"cache_dir": str(cache_dir)},
        arm={"stage": "fetch"},
    )


if __name__ == "__main__":
    main()
