"""Tests for igv.coupling -- double-mutant-cycle extraction from SKEMPI 2.0.

Pinned numbers use within-reference matching (the default). The legacy
mean-all mode is retained but not the default and not regression-pinned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from igv.coupling import (
    Cycle,
    aggregate_pairs,
    cycles_to_dataframe,
    extract_all_complexes,
    extract_cycles,
)
from igv.skempi import add_ddg, ddg as compute_ddg, filter_complex

_SKEMPI_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "skempi_v2.csv"

_need_csv = pytest.mark.skipif(
    not _SKEMPI_CSV.exists(), reason="SKEMPI CSV not on disk"
)


# ---------------------------------------------------------------------------
# Unit tests (no CSV required)
# ---------------------------------------------------------------------------


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal SKEMPI-like DataFrame for unit tests."""
    base = {
        "#Pdb": "1XYZ_A_B",
        "Mutation(s)_PDB": "",
        "Affinity_mut_parsed": 1e-9,
        "Affinity_wt_parsed": 1e-9,
        "Temperature": "298",
        "Reference": "99999999",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_extract_cycles_simple():
    """Two singles and one double give exactly one cycle."""
    df = _make_df([
        {"Mutation(s)_PDB": "AA10G", "Affinity_mut_parsed": 1e-8},
        {"Mutation(s)_PDB": "GB20A", "Affinity_mut_parsed": 1e-7},
        {"Mutation(s)_PDB": "AA10G,GB20A", "Affinity_mut_parsed": 1e-5},
    ])
    df = add_ddg(df)
    cycles = extract_cycles(df, complex_key="1XYZ_A_B")
    assert len(cycles) == 1
    c = cycles[0]
    assert c.pos_i == ("A", "10")
    assert c.pos_j == ("B", "20")
    assert c.cross_chain is True
    assert abs(c.coupling - (c.ddg_double - c.ddg_i - c.ddg_j)) < 1e-10


def test_extract_cycles_missing_single():
    """A double with no matching single gives no cycle."""
    df = _make_df([
        {"Mutation(s)_PDB": "AA10G", "Affinity_mut_parsed": 1e-8},
        {"Mutation(s)_PDB": "AA10G,GB20A", "Affinity_mut_parsed": 1e-5},
    ])
    df = add_ddg(df)
    cycles = extract_cycles(df, complex_key="1XYZ_A_B")
    assert len(cycles) == 0


def test_extract_cycles_same_chain():
    """Both mutations on chain A -> not cross-chain."""
    df = _make_df([
        {"Mutation(s)_PDB": "AA10G", "Affinity_mut_parsed": 1e-8},
        {"Mutation(s)_PDB": "GA20A", "Affinity_mut_parsed": 1e-7},
        {"Mutation(s)_PDB": "AA10G,GA20A", "Affinity_mut_parsed": 1e-5},
    ])
    df = add_ddg(df)
    cycles = extract_cycles(df, complex_key="1XYZ_A_B")
    assert len(cycles) == 1
    assert cycles[0].cross_chain is False


def test_aggregate_pairs_replicate_spread():
    """Two cycles at the same pair produce a spread."""
    c1 = Cycle(
        complex="1X_A_B", pos_i=("A", "1"), pos_j=("B", "2"),
        cross_chain=True, ddg_double=3.0, ddg_i=1.0, ddg_j=1.0, coupling=1.0,
    )
    c2 = Cycle(
        complex="1X_A_B", pos_i=("A", "1"), pos_j=("B", "2"),
        cross_chain=True, ddg_double=4.0, ddg_i=1.5, ddg_j=1.5, coupling=1.0,
    )
    c3 = Cycle(
        complex="1X_A_B", pos_i=("A", "1"), pos_j=("B", "2"),
        cross_chain=True, ddg_double=5.0, ddg_i=1.0, ddg_j=1.5, coupling=2.5,
    )
    agg = aggregate_pairs([c1, c2, c3])
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["n_cycles"] == 3
    assert abs(row["replicate_spread"] - 1.5) < 1e-10


def test_cycles_to_dataframe():
    c = Cycle(
        complex="1X_A_B", pos_i=("A", "1"), pos_j=("B", "2"),
        cross_chain=True, ddg_double=3.0, ddg_i=1.0, ddg_j=1.0, coupling=1.0,
    )
    df = cycles_to_dataframe([c])
    assert len(df) == 1
    assert "coupling" in df.columns
    assert "cross_chain" in df.columns


# ---------------------------------------------------------------------------
# Integration tests on the real SKEMPI CSV
# ---------------------------------------------------------------------------


@_need_csv
class Test1JTG:
    """Pinned regression numbers for 1JTG_A_B (within-reference matching)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        df = pd.read_csv(_SKEMPI_CSV, sep=";")
        df = add_ddg(df)
        self.cycles = extract_cycles(df, complex_key="1JTG_A_B")
        self.couplings = np.array([c.coupling for c in self.cycles])

    def test_complete_cycles(self):
        assert len(self.cycles) == 80

    def test_distinct_pairs(self):
        pairs = {
            tuple(sorted([c.pos_i, c.pos_j])) for c in self.cycles
        }
        assert len(pairs) == 74

    def test_distinct_positions(self):
        positions: set[tuple[str, str]] = set()
        for c in self.cycles:
            positions.add(c.pos_i)
            positions.add(c.pos_j)
        assert len(positions) == 31

    def test_cross_chain_cycles(self):
        assert sum(c.cross_chain for c in self.cycles) == 66

    def test_same_chain_cycles(self):
        assert sum(not c.cross_chain for c in self.cycles) == 14

    def test_coupling_std(self):
        assert abs(np.std(self.couplings, ddof=1) - 0.92) < 0.005

    def test_coupling_range(self):
        assert abs(self.couplings.min() - (-3.53)) < 0.005
        assert abs(self.couplings.max() - 1.96) < 0.005

    def test_abs_coupling_gt_half(self):
        assert int((np.abs(self.couplings) > 0.5).sum()) == 33

    def test_coupling_self_consistent(self):
        for c in self.cycles:
            expected = c.ddg_double - c.ddg_i - c.ddg_j
            assert abs(c.coupling - expected) < 1e-10

    def test_cross_vs_same_is_null(self):
        """The cross-chain vs same-chain gap is not significant."""
        cross = [abs(c.coupling) for c in self.cycles if c.cross_chain]
        same = [abs(c.coupling) for c in self.cycles if not c.cross_chain]
        assert len(cross) == 66 and len(same) == 14
        gap = np.mean(cross) - np.mean(same)
        assert abs(gap) < 0.5, f"gap {gap:.3f} unexpectedly large"


@_need_csv
class TestPerComplexTable:
    """Pin per-complex distinct-pair counts (within-reference matching)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        df = pd.read_csv(_SKEMPI_CSV, sep=";")
        self.df = add_ddg(df)

    _EXPECTED_PAIRS = {
        "1JTG_A_B": 74,
        "1BRS_A_D": 33,
        "3S9D_A_B": 53,
        "4G0N_A_B": 32,
        "1LFD_A_B": 26,
        "1AO7_ABC_DE": 21,
        "3HFM_HL_Y": 11,
        "1DAN_HL_UT": 4,
        "1VFB_AB_C": 14,
        "1DQJ_AB_C": 13,
        "1DVF_AB_CD": 13,
    }

    @pytest.mark.parametrize("cx_key", list(_EXPECTED_PAIRS.keys()))
    def test_per_complex_pairs(self, cx_key: str):
        exp_pairs = self._EXPECTED_PAIRS[cx_key]
        cycles = extract_cycles(self.df, complex_key=cx_key)
        pairs = {tuple(sorted([c.pos_i, c.pos_j])) for c in cycles}
        assert len(pairs) == exp_pairs, (
            f"{cx_key}: expected {exp_pairs} pairs, got {len(pairs)}"
        )

    def test_1dan_below_threshold(self):
        """1DAN collapses to 4 pairs under within-reference -- below min_cycles=8."""
        cycles = extract_cycles(self.df, complex_key="1DAN_HL_UT")
        assert len(cycles) == 4
        all_cx = extract_all_complexes(self.df, min_cycles=8)
        assert "1DAN_HL_UT" not in all_cx

    def test_total_distinct_pairs(self):
        """294 total distinct pairs across the 11-complex replication set."""
        total = sum(
            len({
                tuple(sorted([c.pos_i, c.pos_j]))
                for c in extract_cycles(self.df, complex_key=cx_key)
            })
            for cx_key in self._EXPECTED_PAIRS
        )
        assert total == 294, f"Expected 294 total distinct pairs, got {total}"


@_need_csv
class TestTemperatureRegression:
    """Verify that per-row temperature is used, not a fixed 298 K.

    Reference 17430899 is a Van't Hoff temperature series for 1JTG spanning
    279--303 K.  A regression to fixed 298 K would produce different ddG
    values for the non-298 K measurements.
    """

    def test_ddg_differs_from_fixed_298(self):
        df = pd.read_csv(_SKEMPI_CSV, sep=";")
        df = add_ddg(df)
        cdf = filter_complex(df, "1JTG")

        ref_rows = cdf[
            (cdf["Reference"].astype(str).str.strip() == "17430899")
            & (cdf["temperature_k"] != 298.0)
        ]
        assert len(ref_rows) > 0, "no non-298K rows in reference 17430899"

        row = ref_rows.iloc[0]
        ddg_actual = float(row["ddg_kcal_mol"])
        ddg_fixed = compute_ddg(
            float(row["Affinity_mut_parsed"]),
            float(row["Affinity_wt_parsed"]),
            298.0,
        )
        assert ddg_actual != pytest.approx(ddg_fixed, abs=1e-6), (
            "ddG should differ from fixed-298K value for a non-298K row; "
            "per-row temperature is not being used"
        )
