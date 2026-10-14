#!/usr/bin/env python3
"""Exact refeaturise skip-count and reason breakdown for section 25.

The scan CSV lumps every skip under mode='refeaturise_SKIPPED_SHAPE', so the
reason has to come from the scan log, where 04_scan.py emits one of three
distinct guard messages.  Section 24 found 847/950 skips on 3S9D, all on
TOKEN_TO_REP_ATOM CONTENT DIFFERS and none on either shape check; this script
is what lets that be checked rather than assumed.

Usage:
    python3 scripts/25_skip_breakdown.py --csv results/X_scan_refeaturise.csv \
        --log logs/X_scan_refeat.log
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Note the ordering/anchoring: "TOKEN_TO_REP_ATOM SHAPE MISMATCH" contains
# "ATOM SHAPE MISMATCH" as a substring, so the bare atom-shape check must be
# matched with a non-underscore character in front of it or it over-counts.
REASONS = {
    "TOKEN_TO_REP_ATOM CONTENT DIFFERS": r"TOKEN_TO_REP_ATOM CONTENT DIFFERS",
    "TOKEN_TO_REP_ATOM SHAPE MISMATCH": r"TOKEN_TO_REP_ATOM SHAPE MISMATCH",
    "ATOM SHAPE MISMATCH (atom_pad_mask)": r"(?:^|[^_])ATOM SHAPE MISMATCH",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    mut = df[df["mutation_id"] != "WT"]
    scored = int((mut["mode"] == "refeaturise").sum())
    skipped = int((mut["mode"] == "refeaturise_SKIPPED_SHAPE").sum())
    total = len(mut)

    text = Path(args.log).read_text(errors="replace")
    counts = {
        name: len(re.findall(pat, text, flags=re.MULTILINE))
        for name, pat in REASONS.items()
    }
    # The bare atom-shape pattern still matches the token_to_rep_atom line when
    # preceded by a space, so subtract that overlap explicitly.
    counts["ATOM SHAPE MISMATCH (atom_pad_mask)"] -= counts[
        "TOKEN_TO_REP_ATOM SHAPE MISMATCH"
    ]

    print(f"csv = {args.csv}")
    print(f"log = {args.log}")
    print("\n| quantity | count | share |")
    print("|---|---|---|")
    print(f"| mutants attempted | {total} | 100% |")
    for name, c in counts.items():
        share = f"{100 * c / total:.1f}%" if total else "--"
        print(f"| skipped, `{name}` | {c} | {share} |")
    print(f"| **scored** | **{scored}** | "
          f"{100 * scored / total:.1f}% |" if total else "| scored | 0 | -- |")

    acct = sum(counts.values())
    status = "OK" if acct == skipped else "MISMATCH"
    print(f"\nskip rows in CSV = {skipped}; reasons found in log = {acct} "
          f"-> {status}")
    if acct != skipped:
        print("  The log may be truncated by a forced VM stop, or a skip "
              "reason is unaccounted for. Do not quote the breakdown.")


if __name__ == "__main__":
    main()
