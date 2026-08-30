#!/usr/bin/env python3
"""Build a MutantLibrary from an AbBiBench affinity dataset and write to parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import (
    _STRUCTURE_NAMES,
    align_reference_to_structure,
    build_library,
    download,
    read_pdb_chains,
)
from igv.provenance import write as prov_write

STRUCTURE_FOR_DATASET = {
    "4fqi_h1": "4fqi_hlab",
    "4fqi_h3": "4fqi_hlab",
}

CHAIN_FOR_DATASET = {
    "4fqi_h1": {"ab_heavy": "H", "ab_light": "L", "antigen": ["A", "B"]},
}


def _assert_4fqi_h1(lib, alignment):
    """Hard assertions for the 4fqi_h1 dataset based on verified facts."""
    assert len(lib.frame) == 65094, f"Expected 65094 rows, got {len(lib.frame)}"
    assert len(lib.reference_seq) == 121, (
        f"Expected heavy chain 121 aa, got {len(lib.reference_seq)}"
    )

    light_lens = lib.frame["light_chain_seq"].str.len().unique()
    assert len(light_lens) == 1 and light_lens[0] == 109, (
        f"Expected constant light chain 109 aa, got {sorted(light_lens)}"
    )

    expected_varpos = [28, 29, 30, 51, 56, 57, 58, 70, 73, 74, 75, 76, 83, 86, 94, 105]
    assert lib.variable_positions == expected_varpos, (
        f"Expected variable positions {expected_varpos}, got {lib.variable_positions}"
    )
    assert len(lib.variable_positions) == 16

    for pos in lib.variable_positions:
        assert len(lib.alphabet_at[pos]) == 2, (
            f"Position {pos} has {len(lib.alphabet_at[pos])} states, expected 2 (binary)"
        )

    scores = lib.frame["binding_score"]
    assert abs(scores.min() - 7.0) < 0.01, f"Min score {scores.min()}, expected ~7.0"
    assert abs(scores.max() - 9.835) < 0.01, f"Max score {scores.max()}, expected ~9.835"

    n_floor = (scores == 7.0).sum()
    assert n_floor == 1675, f"Expected 1675 rows at floor 7.0, got {n_floor}"
    pct = n_floor / len(lib.frame) * 100
    assert abs(pct - 2.6) < 0.15, f"Floor percentage {pct:.1f}%, expected ~2.6%"

    assert alignment["same_length"] is True
    assert alignment["offset"] == 0
    mm_indices = sorted(m[0] for m in alignment["mismatches"])
    # Mismatches at *variable* positions are expected and depend on which
    # library member we picked as the reference -- the consensus differs from
    # the deposited structure at whichever variable sites disagree.  The
    # invariant that actually matters is the scaffold: at positions the library
    # never varies, the structure and the library must agree except at the two
    # known fixed substitutions.  If that set changes, the structure no longer
    # corresponds to this library and the fixed geometry is invalid.
    variable = set(lib.variable_positions)
    fixed_mm = sorted(i for i in mm_indices if i not in variable)
    assert fixed_mm == [23, 45], (
        f"Expected fixed-position mismatches at [23, 45], got {fixed_mm} "
        f"(all mismatches: {mm_indices})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mutant library from affinity data")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. 4fqi_h1)")
    parser.add_argument("--chain", default="H", choices=["H", "L"])
    parser.add_argument(
        "--out",
        default="data/processed/{dataset}_library.parquet",
        help="Output parquet path",
    )
    parser.add_argument("--cache-dir", default="data/raw")
    args = parser.parse_args()

    out_path = Path(args.out.format(dataset=args.dataset))
    cache_dir = Path(args.cache_dir)

    print(f"Building library for {args.dataset} (chain={args.chain})...")
    lib = build_library(args.dataset, cache_dir, chain=args.chain)

    print(f"  rows:              {len(lib.frame)}")
    print(f"  reference length:  {len(lib.reference_seq)}")
    print(f"  variable positions:{len(lib.variable_positions)}: {lib.variable_positions}")
    for pos in lib.variable_positions:
        print(f"    pos {pos:3d}: {lib.alphabet_at[pos]}")

    alignment = None
    struct_name = STRUCTURE_FOR_DATASET.get(args.dataset)
    if struct_name:
        pdb_path = download(struct_name, "structure", cache_dir)
        pdb_chains = read_pdb_chains(pdb_path)
        print(f"\n  Structure {struct_name}:")
        for ch, seq in pdb_chains.items():
            print(f"    chain {ch}: {len(seq)} aa")

        chain_id = args.chain
        if chain_id in pdb_chains:
            alignment = align_reference_to_structure(lib.reference_seq, pdb_chains[chain_id])
            print(f"\n  Alignment (library vs PDB chain {chain_id}):")
            print(f"    same_length: {alignment['same_length']}")
            print(f"    offset:      {alignment['offset']}")
            if alignment["mismatches"]:
                print(f"    mismatches:  {len(alignment['mismatches'])}")
                for idx, pdb_aa, lib_aa in alignment["mismatches"]:
                    print(f"      index {idx}: PDB={pdb_aa}, library={lib_aa}")

    if args.dataset == "4fqi_h1":
        _assert_4fqi_h1(lib, alignment)
        print("\n  All 4fqi_h1 assertions passed.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = lib.frame.copy()
    frame["substitutions_str"] = [
        ";".join(f"{pos}{aa}" for pos, aa in subs) for subs in lib.substitutions
    ]
    frame.to_parquet(out_path, index=False)
    print(f"\n  Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    companion = {
        "dataset": lib.name,
        "chain": lib.chain,
        "reference_seq": lib.reference_seq,
        "variable_positions": lib.variable_positions,
        "alphabet_at": {str(k): v for k, v in lib.alphabet_at.items()},
        "n_rows": len(lib.frame),
        "structure_alignment": alignment,
    }
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(companion, f, indent=2)
    print(f"  Wrote {json_path}")

    prov_write(
        out_path,
        stage="01_build_library",
        inputs={"dataset": args.dataset, "cache_dir": str(cache_dir)},
        params={"chain": args.chain},
        arm={"stage": "build_library", "dataset": args.dataset, "chain": args.chain},
    )


if __name__ == "__main__":
    main()
