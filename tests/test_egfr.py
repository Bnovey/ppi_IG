"""Tests for igv.egfr -- CPU-only, no network access."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from igv.egfr import (
    EGFR_1M17_OFFSET,
    EGFR_DRUGS,
    EGFRMutation,
    parse_egfr_mutation,
    parse_panel_a,
    verify_sign_convention,
    verify_wt_residues_1m17,
)

_XLSX_PATH = Path("data/raw/robichaux2021_supp_table4.xlsx")
_PDB_PATH = Path("data/raw/1m17.pdb")
_HAS_RAW_DATA = _XLSX_PATH.exists() and _PDB_PATH.exists()


# ---------------------------------------------------------------------------
# Inline fixture -- representative subset of the Robichaux Panel A
# ---------------------------------------------------------------------------

_FIXTURE_XLSX_DATA = {
    "Exon 1": [21, 20, 21, 20, 20],
    "Exon 2": [None, None, None, None, None],
    "Exon 3": [None, None, None, None, None],
    "Group": ["CL", "CL", "CL", "CL", "CL"],
    "EGFR Mutation": ["T790M", "L858R", "L718Q", "S811F", "Ex19del/T790M"],
    "Erlotinib": [1.56, -1.17, 0.79, -1.35, 0.5],
    "Erlotinib.1": [1.56, -1.17, 0.79, -1.07, 0.5],
    "Erlotinib.2": [1.56, -1.17, 0.79, -1.35, 0.5],
    "Osimertinib": [-0.53, -1.58, 1.03, -2.46, 0.1],
    "Osimertinib.1": [-0.41, -1.58, 1.03, -2.46, 0.1],
    "Osimertinib.2": [-1.35, -1.58, 1.03, -2.46, 0.1],
}


@pytest.fixture
def fixture_xlsx(tmp_path):
    df = pd.DataFrame(_FIXTURE_XLSX_DATA)
    path = tmp_path / "test_panel_a.xlsx"
    df.to_excel(path, sheet_name="Panel A", index=False, engine="openpyxl")
    return path


@pytest.fixture
def fixture_df(fixture_xlsx):
    return parse_panel_a(fixture_xlsx)


# ---------------------------------------------------------------------------
# parse_egfr_mutation
# ---------------------------------------------------------------------------


class TestParseEGFRMutation:
    def test_normal(self):
        m = parse_egfr_mutation("T790M")
        assert m == EGFRMutation(wt_aa="T", resnum=790, mut_aa="M")

    def test_str_roundtrip(self):
        m = parse_egfr_mutation("L858R")
        assert str(m) == "L858R"

    def test_pdb_resnum(self):
        m = parse_egfr_mutation("T790M")
        assert m.pdb_resnum_1m17 == 790 + EGFR_1M17_OFFSET

    def test_malformed_raises(self):
        with pytest.raises(ValueError, match="Malformed"):
            parse_egfr_mutation("Ex19del")


# ---------------------------------------------------------------------------
# 1M17 offset
# ---------------------------------------------------------------------------


class TestOffset:
    def test_offset_is_minus_24(self):
        assert EGFR_1M17_OFFSET == -24

    def test_t790_maps_to_766(self):
        assert 790 + EGFR_1M17_OFFSET == 766


# ---------------------------------------------------------------------------
# Panel A parsing
# ---------------------------------------------------------------------------


class TestParsePanelA:
    def test_filters_to_singles(self, fixture_df):
        assert "Ex19del/T790M" not in fixture_df["mutation"].values

    def test_keeps_singles(self, fixture_df):
        assert "T790M" in fixture_df["mutation"].values
        assert "L858R" in fixture_df["mutation"].values

    def test_has_required_columns(self, fixture_df):
        required = {"mutation", "drug", "rep1", "rep2", "rep3",
                     "mean_response", "replicate_sd", "position",
                     "wt_aa", "mut_aa"}
        assert required.issubset(set(fixture_df.columns))

    def test_mean_response_computed(self, fixture_df):
        t790m_erl = fixture_df[
            (fixture_df["mutation"] == "T790M") & (fixture_df["drug"] == "Erlotinib")
        ]
        assert len(t790m_erl) == 1
        assert t790m_erl.iloc[0]["mean_response"] == pytest.approx(1.56, abs=0.01)

    def test_replicate_sd_present(self, fixture_df):
        row = fixture_df[
            (fixture_df["mutation"] == "S811F") & (fixture_df["drug"] == "Erlotinib")
        ]
        assert len(row) == 1
        assert row.iloc[0]["replicate_sd"] > 0


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------


class TestSignConvention:
    def test_t790m_erlotinib_positive(self, fixture_df):
        row = fixture_df[
            (fixture_df["mutation"] == "T790M") & (fixture_df["drug"] == "Erlotinib")
        ]
        assert row.iloc[0]["mean_response"] > 0

    def test_t790m_osimertinib_negative(self, fixture_df):
        row = fixture_df[
            (fixture_df["mutation"] == "T790M") & (fixture_df["drug"] == "Osimertinib")
        ]
        assert row.iloc[0]["mean_response"] < 0

    def test_l858r_erlotinib_negative(self, fixture_df):
        row = fixture_df[
            (fixture_df["mutation"] == "L858R") & (fixture_df["drug"] == "Erlotinib")
        ]
        assert row.iloc[0]["mean_response"] < 0

    def test_l718q_osimertinib_positive(self, fixture_df):
        row = fixture_df[
            (fixture_df["mutation"] == "L718Q") & (fixture_df["drug"] == "Osimertinib")
        ]
        assert row.iloc[0]["mean_response"] > 0

    def test_verify_sign_convention_passes(self, fixture_df):
        ok, problems = verify_sign_convention(fixture_df)
        assert ok, f"Sign guard failed: {problems}"


# ---------------------------------------------------------------------------
# Drug list
# ---------------------------------------------------------------------------


class TestDrugList:
    def test_18_drugs(self):
        assert len(EGFR_DRUGS) == 18

    def test_erlotinib_present(self):
        assert "Erlotinib" in EGFR_DRUGS

    def test_osimertinib_present(self):
        assert "Osimertinib" in EGFR_DRUGS


# ---------------------------------------------------------------------------
# Integration tests -- real downloaded files (skip if absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_RAW_DATA, reason="raw EGFR data not downloaded")
class TestEGFRIntegration:
    @pytest.fixture(scope="class")
    def real_df(self):
        return parse_panel_a(_XLSX_PATH)

    def test_row_count(self, real_df):
        assert len(real_df) == 486

    def test_27_mutations(self, real_df):
        assert real_df["mutation"].nunique() == 27

    def test_20_positions(self, real_df):
        assert real_df["position"].nunique() == 20

    def test_18_drugs(self, real_df):
        assert real_df["drug"].nunique() == 18

    def test_wt_residue_guard_all_match(self, real_df):
        mutations = sorted(real_df["mutation"].unique())
        n_match, n_total, mismatches = verify_wt_residues_1m17(mutations, _PDB_PATH)
        assert n_match == n_total, f"Mismatches: {mismatches}"
        assert n_total == 27

    def test_sign_convention_guard(self, real_df):
        ok, problems = verify_sign_convention(real_df)
        assert ok, f"Sign guard failed: {problems}"

    def test_t790m_erlotinib_anchor(self, real_df):
        row = real_df[
            (real_df["mutation"] == "T790M") & (real_df["drug"] == "Erlotinib")
        ]
        assert abs(row.iloc[0]["mean_response"] - 1.56) < 0.05

    def test_t790m_osimertinib_anchor(self, real_df):
        row = real_df[
            (real_df["mutation"] == "T790M") & (real_df["drug"] == "Osimertinib")
        ]
        assert abs(row.iloc[0]["mean_response"] - (-0.76)) < 0.05

    def test_median_replicate_sd(self, real_df):
        median_sd = real_df["replicate_sd"].median()
        assert abs(median_sd - 0.080) < 0.005
