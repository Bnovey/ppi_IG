"""Tests for igv.skempi — CPU-only unit tests plus one CSV integration test."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from igv.skempi import (
    Mutation,
    SkempiComplex,
    add_ddg,
    antibody_antigen,
    ddg,
    filter_complex,
    get_complex,
    map_mutations_to_indices,
    parse_mutation,
    parse_mutations,
    read_pdb_residue_ids,
    single_point,
)

_R = 1.987204259e-3


# ---------------------------------------------------------------------------
# parse_mutation
# ---------------------------------------------------------------------------

def test_parse_mutation_normal():
    m = parse_mutation("LI38G")
    assert m == Mutation(wt_aa="L", chain="I", resnum="38", mut_aa="G")


def test_parse_mutation_insertion_code():
    m = parse_mutation("SA100AG")
    assert m == Mutation(wt_aa="S", chain="A", resnum="100A", mut_aa="G")


def test_parse_mutation_malformed():
    with pytest.raises(ValueError, match="Malformed"):
        parse_mutation("38")


# ---------------------------------------------------------------------------
# parse_mutations
# ---------------------------------------------------------------------------

def test_parse_mutations_multi():
    muts = parse_mutations("LI38G,SA100AG")
    assert len(muts) == 2
    assert muts[0] == Mutation(wt_aa="L", chain="I", resnum="38", mut_aa="G")
    assert muts[1] == Mutation(wt_aa="S", chain="A", resnum="100A", mut_aa="G")


# ---------------------------------------------------------------------------
# ddg
# ---------------------------------------------------------------------------

def test_ddg_weaker_binding():
    result = ddg(kd_mut=1e-6, kd_wt=1e-9)
    assert result > 0


def test_ddg_identical():
    assert ddg(kd_mut=1e-9, kd_wt=1e-9) == 0.0


def test_ddg_non_positive_raises():
    with pytest.raises(ValueError, match="positive"):
        ddg(kd_mut=0.0, kd_wt=1e-9)
    with pytest.raises(ValueError, match="positive"):
        ddg(kd_mut=1e-9, kd_wt=-1.0)


def test_ddg_hand_computed():
    kd_mut = 1e-6
    kd_wt = 1e-9
    expected = _R * 298.0 * math.log(kd_mut / kd_wt)
    assert abs(ddg(kd_mut, kd_wt) - expected) < 1e-10


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def skempi_fixture():
    return pd.DataFrame({
        "#Pdb": ["1VFB_AB_C", "1vfb_AB_C", "3HFM_HL_Y", "1MHP_A_B"],
        "Mutation(s)_cleaned": ["LA38G", "LA38G,SA100AG", "LH50A", "MA10G"],
        "Hold_out_type": ["AB/AG", "", "AB/AG,Pr/PI", "Pr/PI"],
        "Affinity_mut_parsed": [1e-6, 1e-7, 1e-8, 1e-5],
        "Affinity_wt_parsed": [1e-9, 1e-9, 1e-9, 1e-9],
        "Temperature": ["298", "298 (assumed)", "310", "foo"],
    })


def test_add_ddg(skempi_fixture):
    df = add_ddg(skempi_fixture)
    assert "ddg_kcal_mol" in df.columns
    assert "temperature_k" in df.columns
    assert len(df) == 4
    assert df["temperature_k"].iloc[0] == 298.0
    assert df["temperature_k"].iloc[2] == 310.0
    assert df["temperature_k"].iloc[3] == 298.0  # unparseable -> default


def test_single_point(skempi_fixture):
    df = single_point(skempi_fixture)
    assert len(df) == 3
    assert all(df["Mutation(s)_cleaned"].str.count(",") == 0)


def test_filter_complex(skempi_fixture):
    df = filter_complex(skempi_fixture, "1VFB")
    assert len(df) == 2


def test_antibody_antigen(skempi_fixture):
    df = antibody_antigen(skempi_fixture)
    assert len(df) == 2
    assert all(df["Hold_out_type"].str.contains("AB/AG"))


# ---------------------------------------------------------------------------
# Integration test (requires real CSV on disk)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Complex registry
# ---------------------------------------------------------------------------

def test_get_complex_known():
    c = get_complex("3HFM")
    assert isinstance(c, SkempiComplex)
    assert c.pdb_id == "3HFM"
    assert c.ab_chains == ("H", "L")
    assert c.ag_chains == ("Y",)


def test_get_complex_case_insensitive():
    assert get_complex("3hfm") == get_complex("3HFM")


def test_get_complex_unknown():
    with pytest.raises(KeyError, match="Unknown SKEMPI complex"):
        get_complex("9ZZZ")


# ---------------------------------------------------------------------------
# Integration test (requires real CSV on disk)
# ---------------------------------------------------------------------------

_SKEMPI_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "skempi_v2.csv"


@pytest.mark.skipif(not _SKEMPI_CSV.exists(), reason="SKEMPI CSV not on disk")
def test_load_real_csv():
    df = pd.read_csv(_SKEMPI_CSV, sep=";")
    assert len(df) > 7000
    assert "#Pdb" in df.columns
    assert "Mutation(s)_cleaned" in df.columns

    df = add_ddg(df)
    sp = single_point(df)
    assert len(sp) > 4000


# ---------------------------------------------------------------------------
# Inline PDB fixture (with insertion code)
# ---------------------------------------------------------------------------

_MINI_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  N   GLY A   2       3.000   4.000   5.000  1.00  0.00           N
ATOM      4  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      5  N   SER A   3       5.000   6.000   7.000  1.00  0.00           N
ATOM      6  CA  SER A   3       6.000   7.000   8.000  1.00  0.00           C
ATOM      7  N   LEU A   3A      7.000   8.000   9.000  1.00  0.00           N
ATOM      8  CA  LEU A   3A      8.000   9.000  10.000  1.00  0.00           C
ATOM      9  N   VAL A   4       9.000  10.000  11.000  1.00  0.00           N
ATOM     10  CA  VAL A   4      10.000  11.000  12.000  1.00  0.00           C
ATOM     11  N   MET B   5      11.000  12.000  13.000  1.00  0.00           N
ATOM     12  CA  MET B   5      12.000  13.000  14.000  1.00  0.00           C
ATOM     13  N   TRP B   6      13.000  14.000  15.000  1.00  0.00           N
ATOM     14  CA  TRP B   6      14.000  15.000  16.000  1.00  0.00           C
END
"""


@pytest.fixture
def mini_pdb_path(tmp_path):
    p = tmp_path / "mini.pdb"
    p.write_text(_MINI_PDB)
    return p


# ---------------------------------------------------------------------------
# read_pdb_residue_ids
# ---------------------------------------------------------------------------

def test_read_pdb_residue_ids_basic(mini_pdb_path):
    ids, seqs = read_pdb_residue_ids(mini_pdb_path)
    assert list(ids.keys()) == ["A", "B"]
    assert ids["A"] == ["1", "2", "3", "3A", "4"]
    assert ids["B"] == ["5", "6"]
    assert seqs["A"] == "AGSLV"
    assert seqs["B"] == "MW"


def test_read_pdb_residue_ids_insertion_code(mini_pdb_path):
    ids, _ = read_pdb_residue_ids(mini_pdb_path)
    assert "3A" in ids["A"]
    idx_3 = ids["A"].index("3")
    idx_3a = ids["A"].index("3A")
    assert idx_3a == idx_3 + 1


def test_read_pdb_residue_ids_length_matches_sequence(mini_pdb_path):
    from igv.data import read_pdb_chains
    ids, seqs = read_pdb_residue_ids(mini_pdb_path)
    ref = read_pdb_chains(mini_pdb_path)
    for ch in ref:
        assert len(ids[ch]) == len(ref[ch])
        assert seqs[ch] == ref[ch]


# ---------------------------------------------------------------------------
# map_mutations_to_indices — wild-type guard
# ---------------------------------------------------------------------------

def test_map_mutations_correct():
    residue_ids = ["1", "2", "3", "3A", "4"]
    sequence = "AGSLV"
    muts = [Mutation(wt_aa="A", chain="A", resnum="1", mut_aa="G")]
    mapped, mismatches = map_mutations_to_indices(muts, residue_ids, sequence)
    assert len(mapped) == 1
    assert mapped[0] == (muts[0], 0)
    assert len(mismatches) == 0


def test_map_mutations_insertion_code():
    residue_ids = ["1", "2", "3", "3A", "4"]
    sequence = "AGSLV"
    muts = [Mutation(wt_aa="L", chain="A", resnum="3A", mut_aa="G")]
    mapped, mismatches = map_mutations_to_indices(muts, residue_ids, sequence)
    assert len(mapped) == 1
    assert mapped[0][1] == 3  # index of "3A"


def test_map_mutations_wt_mismatch_raises():
    residue_ids = ["1", "2", "3"]
    sequence = "AGS"
    muts = [Mutation(wt_aa="X", chain="A", resnum="1", mut_aa="G")]
    with pytest.raises(ValueError, match="wild-type mismatch"):
        map_mutations_to_indices(muts, residue_ids, sequence)


def test_map_mutations_wt_mismatch_allow():
    residue_ids = ["1", "2", "3"]
    sequence = "AGS"
    muts = [Mutation(wt_aa="X", chain="A", resnum="1", mut_aa="G")]
    mapped, mismatches = map_mutations_to_indices(
        muts, residue_ids, sequence, allow_mismatch=True
    )
    assert len(mapped) == 0
    assert len(mismatches) == 1
    assert mismatches[0] == (muts[0], "X", "A")


def test_map_mutations_resnum_not_found():
    residue_ids = ["1", "2", "3"]
    sequence = "AGS"
    muts = [Mutation(wt_aa="A", chain="A", resnum="99", mut_aa="G")]
    with pytest.raises(ValueError, match="not found in the PDB"):
        map_mutations_to_indices(muts, residue_ids, sequence)


# ---------------------------------------------------------------------------
# Aggregation of multiple mutations at one position
# ---------------------------------------------------------------------------

def test_aggregation_multiple_mutations_at_position():
    residue_ids = ["1", "2", "3"]
    sequence = "AGS"
    muts = [
        Mutation(wt_aa="A", chain="A", resnum="1", mut_aa="G"),
        Mutation(wt_aa="A", chain="A", resnum="1", mut_aa="V"),
        Mutation(wt_aa="G", chain="A", resnum="2", mut_aa="A"),
    ]
    mapped, _ = map_mutations_to_indices(muts, residue_ids, sequence)
    assert len(mapped) == 3

    pos_ddgs: dict[int, list[float]] = {}
    ddg_vals = [3.0, 5.0, 1.0]
    for (mut, idx), ddg_val in zip(mapped, ddg_vals):
        pos_ddgs.setdefault(idx, []).append(ddg_val)

    assert pos_ddgs[0] == [3.0, 5.0]
    assert max(pos_ddgs[0]) == 5.0
    assert np.mean(pos_ddgs[0]) == 4.0
    assert pos_ddgs[1] == [1.0]


# ---------------------------------------------------------------------------
# Integration test with real PDB
# ---------------------------------------------------------------------------

_PDB_3HFM = Path(__file__).resolve().parent.parent / "data" / "raw" / "3hfm.pdb"


@pytest.mark.skipif(not _PDB_3HFM.exists(), reason="3hfm.pdb not on disk")
def test_read_pdb_residue_ids_real_3hfm():
    from igv.data import read_pdb_chains
    ids, seqs = read_pdb_residue_ids(_PDB_3HFM)
    ref = read_pdb_chains(_PDB_3HFM)
    for ch in ("H", "L", "Y"):
        assert ch in ids
        assert len(ids[ch]) == len(ref[ch])
        assert seqs[ch] == ref[ch]
