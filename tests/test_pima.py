"""Tests for igv.platinum (PimA ground truth) -- CPU-only, no network."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd
import pytest

from igv.platinum import (
    PlatinumMutation,
    extract_target,
    parse_mutation,
    verify_wt_residues,
)

_PLATINUM_PATH = Path("data/raw/platinum_flat_file.csv")
_GDD_PDB_PATH = Path("data/raw/3okp.pdb")
_GDP_PDB_PATH = Path("data/raw/3okc.pdb")
_HAS_RAW_DATA = (
    _PLATINUM_PATH.exists() and _GDD_PDB_PATH.exists() and _GDP_PDB_PATH.exists()
)


# ---------------------------------------------------------------------------
# Inline fixture -- representative subset of Platinum flat file
# ---------------------------------------------------------------------------

_FIXTURE_ROWS = [
    {
        "mutation": "D13A",
        "affin.chain": "A",
        "affin.lig_id": "GDD",
        "affin.lig_name": "GDP-mannose",
        "affin.pdb_id": "3OKP",
        "mut.uniprot": "Q8NNK8",
        "affin.exptal_method": "fluorescence",
        "affin.temperature": "25",
        "affin.ph": "7.5",
        "affin.k_wt": "19000",
        "affin.k_mt": "42800",
        "affin.delta_k": "23800",
        "affin.fold_change": "2.25",
        "affin.unit": "KD",
        "mut.mt_pdb": "NULL",
        "mut.wt_pdb": "3OKP",
        "mut.in_binding_site": "NO",
        "mut.distance_to_lig": "13.9336",
        "mut.doi": "10.1074/jbc.M110.165407",
        "mut.pmid": "20843801",
        "mut.is_single_point": "YES",
        "mut.mcsm_ppi": "NULL",
        "mut.duet_stability": "-0.5",
        "mut.residue_rsa": "0.15",
        "mut.residue_sst": "H",
        "mut.residue_hbond": "2",
        "prot.molecule_name": "PimA",
        "prot.organism": "Corynebacterium glutamicum",
        "prot.resolution": "2.0",
        "prot.r_value": "0.18",
    },
    {
        "mutation": "K211Q",
        "affin.chain": "A",
        "affin.lig_id": "GDD",
        "affin.lig_name": "GDP-mannose",
        "affin.pdb_id": "3OKP",
        "mut.uniprot": "Q8NNK8",
        "affin.exptal_method": "fluorescence",
        "affin.temperature": "25",
        "affin.ph": "7.5",
        "affin.k_wt": "19000",
        "affin.k_mt": "30500",
        "affin.delta_k": "11500",
        "affin.fold_change": "1.61",
        "affin.unit": "KD",
        "mut.mt_pdb": "NULL",
        "mut.wt_pdb": "3OKP",
        "mut.in_binding_site": "YES",
        "mut.distance_to_lig": "2.7737",
        "mut.doi": "10.1074/jbc.M110.165407",
        "mut.pmid": "20843801",
        "mut.is_single_point": "YES",
        "mut.mcsm_ppi": "NULL",
        "mut.duet_stability": "-0.3",
        "mut.residue_rsa": "0.05",
        "mut.residue_sst": "E",
        "mut.residue_hbond": "1",
        "prot.molecule_name": "PimA",
        "prot.organism": "Corynebacterium glutamicum",
        "prot.resolution": "2.0",
        "prot.r_value": "0.18",
    },
    {
        "mutation": "D13A",
        "affin.chain": "A",
        "affin.lig_id": "GDP",
        "affin.lig_name": "GDP",
        "affin.pdb_id": "3OKC",
        "mut.uniprot": "Q8NNK8",
        "affin.exptal_method": "fluorescence",
        "affin.temperature": "25",
        "affin.ph": "7.5",
        "affin.k_wt": "36500",
        "affin.k_mt": "61300",
        "affin.delta_k": "24800",
        "affin.fold_change": "1.68",
        "affin.unit": "KD",
        "mut.mt_pdb": "NULL",
        "mut.wt_pdb": "3OKC",
        "mut.in_binding_site": "NO",
        "mut.distance_to_lig": "13.9",
        "mut.doi": "10.1074/jbc.M110.165407",
        "mut.pmid": "20843801",
        "mut.is_single_point": "YES",
        "mut.mcsm_ppi": "NULL",
        "mut.duet_stability": "-0.5",
        "mut.residue_rsa": "0.15",
        "mut.residue_sst": "H",
        "mut.residue_hbond": "2",
        "prot.molecule_name": "PimA",
        "prot.organism": "Corynebacterium glutamicum",
        "prot.resolution": "2.4",
        "prot.r_value": "0.20",
    },
    {
        "mutation": "AB12C",
        "affin.chain": "A",
        "affin.lig_id": "GDD",
        "affin.pdb_id": "3OKP",
        "mut.uniprot": "Q8NNK8",
        "affin.k_wt": "19000",
        "affin.k_mt": "20000",
        "affin.unit": "KD",
        "mut.wt_pdb": "3OKP",
        "mut.is_single_point": "NO",
        "mut.distance_to_lig": "5.0",
        "affin.exptal_method": "fluorescence",
        "affin.temperature": "25",
        "affin.ph": "7.5",
        "affin.delta_k": "1000",
        "affin.fold_change": "1.05",
        "mut.mt_pdb": "NULL",
        "mut.in_binding_site": "NO",
        "mut.doi": "10.1074/jbc.M110.165407",
        "mut.pmid": "20843801",
        "affin.lig_name": "GDP-mannose",
        "mut.mcsm_ppi": "NULL",
        "mut.duet_stability": "-0.1",
        "mut.residue_rsa": "0.10",
        "mut.residue_sst": "C",
        "mut.residue_hbond": "0",
        "prot.molecule_name": "PimA",
        "prot.organism": "Corynebacterium glutamicum",
        "prot.resolution": "2.0",
        "prot.r_value": "0.18",
    },
]


@pytest.fixture
def fixture_rows():
    return _FIXTURE_ROWS


# ---------------------------------------------------------------------------
# parse_mutation
# ---------------------------------------------------------------------------


class TestParseMutation:
    def test_normal(self):
        m = parse_mutation("D13A")
        assert m == PlatinumMutation(wt_aa="D", resnum=13, mut_aa="A")

    def test_str_roundtrip(self):
        assert str(parse_mutation("K211Q")) == "K211Q"

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_mutation("AB12C")


# ---------------------------------------------------------------------------
# extract_target
# ---------------------------------------------------------------------------


class TestExtractTarget:
    def test_filters_by_uniprot_and_ligand(self, fixture_rows):
        df = extract_target(fixture_rows, "Q8NNK8", "GDD")
        assert len(df) == 2
        assert set(df["mutation"]) == {"D13A", "K211Q"}

    def test_filters_to_single_point(self, fixture_rows):
        df = extract_target(fixture_rows, "Q8NNK8", "GDD")
        assert "AB12C" not in df["mutation"].values

    def test_gdp_separate(self, fixture_rows):
        df = extract_target(fixture_rows, "Q8NNK8", "GDP")
        assert len(df) == 1
        assert df.iloc[0]["mutation"] == "D13A"

    def test_response_log10(self, fixture_rows):
        df = extract_target(fixture_rows, "Q8NNK8", "GDD")
        d13a = df[df["mutation"] == "D13A"].iloc[0]
        expected = math.log10(42800) - math.log10(19000)
        assert d13a["response_log10"] == pytest.approx(expected, abs=1e-6)

    def test_positive_means_weaker_binding(self, fixture_rows):
        df = extract_target(fixture_rows, "Q8NNK8", "GDD")
        d13a = df[df["mutation"] == "D13A"].iloc[0]
        assert d13a["response_log10"] > 0

    def test_has_distance(self, fixture_rows):
        df = extract_target(fixture_rows, "Q8NNK8", "GDD")
        d13a = df[df["mutation"] == "D13A"].iloc[0]
        assert d13a["distance_to_ligand"] == pytest.approx(13.9336, abs=0.01)


# ---------------------------------------------------------------------------
# WT residue guard (on fixture PDB)
# ---------------------------------------------------------------------------


class TestVerifyWTResidues:
    def test_all_match_on_fixture(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ASP A  13       1.0  2.0  3.0  1.00  0.00\n"
            "ATOM      2  CA  LYS A 211       4.0  5.0  6.0  1.00  0.00\n"
        )
        df = pd.DataFrame({
            "mutation": ["D13A", "K211Q"],
            "position": [13, 211],
            "wt_aa": ["D", "K"],
        })
        n_match, n_total, mismatches = verify_wt_residues(df, pdb)
        assert n_match == 2
        assert n_total == 2
        assert mismatches == []

    def test_mismatch_detected(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A  13       1.0  2.0  3.0  1.00  0.00\n"
        )
        df = pd.DataFrame({
            "mutation": ["D13A"],
            "position": [13],
            "wt_aa": ["D"],
        })
        n_match, n_total, mismatches = verify_wt_residues(df, pdb)
        assert n_match == 0
        assert len(mismatches) == 1


# ---------------------------------------------------------------------------
# Integration tests -- real downloaded files (skip if absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_RAW_DATA, reason="raw Platinum/PimA data not downloaded")
class TestPimAIntegration:
    @pytest.fixture(scope="class")
    def all_rows(self):
        with open(_PLATINUM_PATH, newline="") as f:
            return list(csv.DictReader(f))

    @pytest.fixture(scope="class")
    def gdd_df(self, all_rows):
        return extract_target(all_rows, "Q8NNK8", "GDD")

    @pytest.fixture(scope="class")
    def gdp_df(self, all_rows):
        return extract_target(all_rows, "Q8NNK8", "GDP")

    def test_gdd_row_count(self, gdd_df):
        assert len(gdd_df) == 16

    def test_gdp_row_count(self, gdp_df):
        assert len(gdp_df) == 16

    def test_gdd_13_positions(self, gdd_df):
        assert gdd_df["position"].nunique() == 13

    def test_gdp_13_positions(self, gdp_df):
        assert gdp_df["position"].nunique() == 13

    def test_gdd_wt_residue_guard(self, gdd_df):
        n_match, n_total, mismatches = verify_wt_residues(gdd_df, _GDD_PDB_PATH)
        assert n_match == n_total, f"Mismatches: {mismatches}"

    def test_gdp_wt_residue_guard(self, gdp_df):
        n_match, n_total, mismatches = verify_wt_residues(gdp_df, _GDP_PDB_PATH)
        assert n_match == n_total, f"Mismatches: {mismatches}"

    def test_positive_is_weaker_binding(self, gdd_df):
        d13a = gdd_df[gdd_df["mutation"] == "D13A"].iloc[0]
        assert d13a["k_mt"] > d13a["k_wt"]
        assert d13a["response_log10"] > 0
