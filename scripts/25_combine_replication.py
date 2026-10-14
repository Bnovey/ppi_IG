#!/usr/bin/env python3
"""Combine section-25 faithfulness JSONs across complexes into one table.

Reads results/{DATASET}_faith_{method}_{scan}.json for every complex that has
them and emits the four-complex table plus the across-complex summary that
section 25 reports.  Pure CPU, no model, no GPU.

Usage:
    python3 scripts/25_combine_replication.py --method ig64 --scan resswap
    python3 scripts/25_combine_replication.py --all
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COMPLEXES = [
    ("3S9D", "A", 50),
    ("1JTG", "B", 28),
    ("2JEL", "P", 32),
    ("3HFM", "Y", 15),
]

RESULTS = Path("results")


def load(dataset: str, method: str, scan: str) -> dict | None:
    p = RESULTS / f"{dataset}_faith_{method}_{scan}.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def clears_null(d: dict) -> bool:
    """True iff the observed Spearman lies outside its own permutation null."""
    pn = d.get("permutation_null", {})
    obs = pn.get("observed", d.get("spearman"))
    p5, p95 = pn.get("p5"), pn.get("p95")
    if obs is None or p5 is None or p95 is None:
        return False
    return bool(obs > p95 or obs < p5)


def table(method: str, scan: str) -> tuple[str, list[float]]:
    hdr = (
        "| complex | chain | positions | n | Spearman | CI | null p95 | "
        "clears null | partial vs distance | partial vs all 8 | scale ratio | "
        "per-position sum |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    rhos: list[float] = []
    for ds, ch, npos in COMPLEXES:
        d = load(ds, method, scan)
        if d is None:
            rows.append(f"| {ds} {ch} | {ch} | {npos} | -- | **not run** "
                        "| -- | -- | -- | -- | -- | -- | -- |")
            continue
        pn = d.get("permutation_null", {})
        rho = d.get("spearman")
        rhos.append(rho)
        pp = d.get("per_position", {})
        # 3HFM is unidentifiable at the position level against 8 confounds.
        partial_all = d.get("partial_spearman_all_confounds")
        partial_all_s = fmt(partial_all)
        # Confounds are constant within a position, so the confound block has
        # only `npos` distinct rows however many mutations there are. With 8
        # confounds and 15 positions, 3HFM's confound matrix is near-saturated
        # and the partial is not identifiable -- section 14 measured its CI as
        # [-1, +1]. Flag it rather than letting it be read as a result.
        n_conf = len(d.get("confound_names", []) or [])
        if n_conf and npos < 3 * n_conf:
            partial_all_s += f" [UNIDENTIFIABLE: {n_conf} confounds, {npos} positions]"
        rows.append(
            f"| {ds} {ch} | {ch} | {npos} | {d.get('n')} | **{fmt(rho)}** "
            f"| [{fmt(d.get('spearman_ci_lo'))}, {fmt(d.get('spearman_ci_hi'))}] "
            f"| {fmt(pn.get('p95'))} "
            f"| {'**yes**' if clears_null(d) else 'no'} "
            f"| {fmt(d.get('partial_spearman_distance_to_partner'))} "
            f"| {partial_all_s} "
            f"| {fmt(d.get('scale_ratio'))} "
            f"| {fmt(pp.get('sum_spearman'))} |"
        )
    return hdr + "\n".join(rows), rhos


def summarise(rhos: list[float], method: str, scan: str) -> str:
    if not rhos:
        return "No complexes available.\n"
    n = len(rhos)
    mean = sum(rhos) / n
    if n > 1:
        sd = math.sqrt(sum((r - mean) ** 2 for r in rhos) / (n - 1))
    else:
        sd = float("nan")
    cleared = sum(
        1
        for ds, _, _ in COMPLEXES
        if (d := load(ds, method, scan)) is not None and clears_null(d)
    )
    return (
        f"\n**Across {n} complexes ({method}, {scan}):** "
        f"mean Spearman **{fmt(mean)}**, sd **{fmt(sd)}**, "
        f"range [{fmt(min(rhos))}, {fmt(max(rhos))}]. "
        f"**{cleared} of {n}** cleared their own permutation null.\n"
    )


def decorrelation() -> str:
    """Resswap-vs-refeaturise scale contrast, per complex."""
    out = [
        "\n| complex | median abs delta resswap | median abs delta refeaturise "
        "| ratio |",
        "|---|---|---|---|",
    ]
    for ds, _, _ in COMPLEXES:
        a = load(ds, "ig64", "resswap") or load(ds, "ig32", "resswap")
        b = load(ds, "ig64", "refeat") or load(ds, "ig32", "refeat")
        if a is None or b is None:
            out.append(f"| {ds} | -- | -- | not run |")
            continue
        ma, mb = a.get("median_abs_B"), b.get("median_abs_B")
        ratio = (mb / ma) if (ma and mb) else None
        out.append(
            f"| {ds} | {fmt(ma)} | {fmt(mb)} | "
            f"{fmt(ratio, 1) if ratio else '--'}x |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ig64")
    ap.add_argument("--scan", default="resswap")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    combos = (
        [(m, s) for m in ("ig64", "ig32", "plaingrad") for s in ("resswap", "refeat")]
        if args.all
        else [(args.method, args.scan)]
    )
    for m, s in combos:
        t, rhos = table(m, s)
        if not rhos and args.all:
            continue
        print(f"\n### {m} vs {s}\n")
        print(t)
        print(summarise(rhos, m, s))
    if args.all:
        print("\n### resswap vs refeaturise effect size\n")
        print(decorrelation())


if __name__ == "__main__":
    main()
