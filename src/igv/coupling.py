"""SKEMPI 2.0 double-mutant-cycle extraction.

A double-mutant cycle measures whether two residues interact::

    coupling = ddG(AB) - ddG(A) - ddG(B)

Zero if the two positions act independently.

Within-reference matching (default)
------------------------------------
A coupling value is only meaningful when all three ddGs (double, single-A,
single-B) share an experimental context: same lab, buffer, method and
wild-type Kd.  SKEMPI's ``Reference`` column is the best available proxy.
The default ``matching="within_reference"`` mode requires both constituent
singles to appear in the same ``Reference`` as the double.  When a single
has several measurements inside that reference (e.g. a Van't Hoff
temperature series), we prefer the measurement at the same ``temperature_k``
as the double, falling back to the mean within-reference.

The legacy ``matching="mean_all"`` mode averages each single's ddG across
*all* available measurements regardless of reference.  This injects
between-lab offset straight into the coupling term.

1DAN note
---------
``1DAN_HL_UT`` collapses from 14 pairs (mean-all) to 4 (within-reference)
because its cycles were almost entirely cross-lab.  With only 4 pairs it is
statistically useless for Phase 5 and falls below the default
``min_cycles=8`` threshold in ``extract_all_complexes``.

Temperature note
----------------
``add_ddg`` applies ``_parse_temperature`` per row, so ddG uses the
measurement's actual temperature.  Reference 17430899 (1JTG) is a Van't
Hoff series spanning 279--303 K; those ddG values differ from what a fixed
298 K would give.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from igv.skempi import (
    SKEMPI_MUTATION_COL,
    add_ddg,
    filter_complex,
    parse_mutation,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cycle:
    """One complete double-mutant cycle."""

    complex: str
    pos_i: tuple[str, str]  # (chain, resnum)
    pos_j: tuple[str, str]
    cross_chain: bool
    ddg_double: float
    ddg_i: float
    ddg_j: float
    coupling: float


def _partner_chain_sets(
    complex_key: str,
) -> tuple[set[str], set[str]]:
    """Parse the SKEMPI ``#Pdb`` key into two partner chain sets."""
    parts = complex_key.split("_")
    p1 = set(parts[1]) if len(parts) > 1 else set()
    p2 = set(parts[2]) if len(parts) > 2 else set()
    return p1, p2


def _pick_ddg(
    measurements: list[tuple[float, float]],
    target_temp: float,
) -> float:
    """Select ddG from *measurements* preferring the target temperature.

    *measurements* is a list of ``(ddg, temperature_k)`` tuples, all from
    the same reference.  If any measurement matches *target_temp* exactly,
    return their mean; otherwise return the mean of all measurements in the
    reference.
    """
    exact = [d for d, t in measurements if t == target_temp]
    if exact:
        return float(np.mean(exact))
    return float(np.mean([d for d, _ in measurements]))


def extract_cycles(
    df: pd.DataFrame,
    complex_key: str | None = None,
    *,
    matching: str = "within_reference",
) -> list[Cycle]:
    """Extract complete double-mutant cycles from a SKEMPI dataframe.

    Parameters
    ----------
    df : DataFrame
        SKEMPI rows (may contain multiple complexes). Must already have
        ``ddg_kcal_mol`` and ``temperature_k`` columns (call ``add_ddg``
        first) or raw affinity columns.
    complex_key : str, optional
        If given, filter to this complex first.
    matching : ``"within_reference"`` (default) or ``"mean_all"``
        How to select the single-mutant ddG for each cycle.

        * ``"within_reference"`` — require both singles to share the
          double's ``Reference``.  Prefer same ``temperature_k``; fall
          back to the within-reference mean.
        * ``"mean_all"`` — average each single's ddG across all available
          measurements regardless of reference (legacy behaviour).

    Returns
    -------
    list[Cycle]
        One entry per double-mutant row whose constituent singles are
        both present (under the chosen matching rule).
    """
    if matching not in ("within_reference", "mean_all"):
        raise ValueError(
            f"matching must be 'within_reference' or 'mean_all', "
            f"got {matching!r}"
        )

    if "ddg_kcal_mol" not in df.columns:
        df = add_ddg(df)

    if complex_key is not None:
        pdb = complex_key.split("_")[0]
        df = filter_complex(df, pdb)

    mut_counts = df[SKEMPI_MUTATION_COL].str.count(",") + 1

    singles = df[mut_counts == 1].copy()
    doubles = df[mut_counts == 2].copy()

    if matching == "within_reference":
        single_by_ref: dict[tuple[str, str], list[tuple[float, float]]] = (
            defaultdict(list)
        )
        for _, row in singles.iterrows():
            key = row[SKEMPI_MUTATION_COL].strip()
            ref = str(row["Reference"]).strip()
            single_by_ref[(key, ref)].append(
                (float(row["ddg_kcal_mol"]), float(row["temperature_k"]))
            )
    else:
        single_ddg: dict[str, list[float]] = {}
        for _, row in singles.iterrows():
            key = row[SKEMPI_MUTATION_COL].strip()
            single_ddg.setdefault(key, []).append(float(row["ddg_kcal_mol"]))

    cx_key = complex_key
    if cx_key is None and len(df) > 0:
        cx_key = df["#Pdb"].iloc[0]

    p1, p2 = _partner_chain_sets(cx_key) if cx_key else (set(), set())

    cycles: list[Cycle] = []
    for _, drow in doubles.iterrows():
        parts = drow[SKEMPI_MUTATION_COL].split(",")
        if len(parts) != 2:
            continue
        s1_key = parts[0].strip()
        s2_key = parts[1].strip()

        if matching == "within_reference":
            dbl_ref = str(drow["Reference"]).strip()
            dbl_temp = float(drow["temperature_k"])

            s1_in_ref = single_by_ref.get((s1_key, dbl_ref))
            s2_in_ref = single_by_ref.get((s2_key, dbl_ref))
            if s1_in_ref is None or s2_in_ref is None:
                continue

            ddg_i = _pick_ddg(s1_in_ref, dbl_temp)
            ddg_j = _pick_ddg(s2_in_ref, dbl_temp)
        else:
            if s1_key not in single_ddg or s2_key not in single_ddg:
                continue
            ddg_i = float(np.mean(single_ddg[s1_key]))
            ddg_j = float(np.mean(single_ddg[s2_key]))

        m1 = parse_mutation(s1_key)
        m2 = parse_mutation(s2_key)

        ddg_double = float(drow["ddg_kcal_mol"])
        coupling = ddg_double - ddg_i - ddg_j

        pos_i = (m1.chain, m1.resnum)
        pos_j = (m2.chain, m2.resnum)

        cross = (
            (m1.chain in p1 and m2.chain in p2)
            or (m1.chain in p2 and m2.chain in p1)
        )

        cycles.append(Cycle(
            complex=drow["#Pdb"],
            pos_i=pos_i,
            pos_j=pos_j,
            cross_chain=cross,
            ddg_double=ddg_double,
            ddg_i=ddg_i,
            ddg_j=ddg_j,
            coupling=coupling,
        ))

    return cycles


def _canonical_pair(
    pos_i: tuple[str, str],
    pos_j: tuple[str, str],
) -> tuple[tuple[str, str], tuple[str, str]]:
    return tuple(sorted([pos_i, pos_j]))  # type: ignore[return-value]


def cycles_to_dataframe(cycles: list[Cycle]) -> pd.DataFrame:
    """Convert a list of cycles to a flat DataFrame."""
    rows = []
    for c in cycles:
        pair = _canonical_pair(c.pos_i, c.pos_j)
        rows.append({
            "complex": c.complex,
            "chain_i": c.pos_i[0],
            "resnum_i": c.pos_i[1],
            "chain_j": c.pos_j[0],
            "resnum_j": c.pos_j[1],
            "pair": f"{pair[0][0]}{pair[0][1]}_{pair[1][0]}{pair[1][1]}",
            "cross_chain": c.cross_chain,
            "ddg_double": c.ddg_double,
            "ddg_i": c.ddg_i,
            "ddg_j": c.ddg_j,
            "coupling": c.coupling,
        })
    return pd.DataFrame(rows)


def aggregate_pairs(cycles: list[Cycle]) -> pd.DataFrame:
    """Aggregate cycles to distinct position pairs.

    When multiple cycles hit the same pair (via different substitutions),
    reports the mean coupling plus the replicate spread (max - min).

    Returns a DataFrame with one row per distinct pair.
    """
    pair_couplings: dict[tuple, list[float]] = {}
    pair_meta: dict[tuple, dict] = {}

    for c in cycles:
        pair = _canonical_pair(c.pos_i, c.pos_j)
        pair_couplings.setdefault(pair, []).append(c.coupling)
        if pair not in pair_meta:
            pair_meta[pair] = {
                "complex": c.complex,
                "chain_i": pair[0][0],
                "resnum_i": pair[0][1],
                "chain_j": pair[1][0],
                "resnum_j": pair[1][1],
                "cross_chain": c.cross_chain,
            }

    rows = []
    for pair in sorted(pair_couplings):
        vals = pair_couplings[pair]
        meta = pair_meta[pair]
        rows.append({
            **meta,
            "pair": f"{pair[0][0]}{pair[0][1]}_{pair[1][0]}{pair[1][1]}",
            "coupling_mean": float(np.mean(vals)),
            "coupling_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "coupling_min": float(np.min(vals)),
            "coupling_max": float(np.max(vals)),
            "replicate_spread": float(np.max(vals) - np.min(vals)),
            "n_cycles": len(vals),
        })

    return pd.DataFrame(rows)


def extract_all_complexes(
    df: pd.DataFrame,
    min_cycles: int = 8,
    *,
    matching: str = "within_reference",
) -> dict[str, list[Cycle]]:
    """Extract cycles for every complex meeting a minimum cycle count.

    Parameters
    ----------
    df : DataFrame
        Full SKEMPI dataframe with ddG already computed.
    min_cycles : int
        Only return complexes with at least this many complete cycles.
    matching : str
        Passed to :func:`extract_cycles`.

    Returns
    -------
    dict mapping complex key to its list of cycles.
    """
    if "ddg_kcal_mol" not in df.columns:
        df = add_ddg(df)

    results: dict[str, list[Cycle]] = {}
    dropped: dict[str, int] = {}
    for cx_key in df["#Pdb"].unique():
        cx_cycles = extract_cycles(df, complex_key=cx_key, matching=matching)
        if len(cx_cycles) >= min_cycles:
            results[cx_key] = cx_cycles
        elif len(cx_cycles) > 0:
            dropped[cx_key] = len(cx_cycles)

    if dropped:
        log.info(
            "Complexes below min_cycles=%d threshold: %s",
            min_cycles,
            ", ".join(f"{k} ({n})" for k, n in sorted(dropped.items())),
        )

    return results
