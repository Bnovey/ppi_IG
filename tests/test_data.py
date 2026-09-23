"""Tests for igv.data — CPU-only unit tests plus one network integration test."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from igv.data import (
    _build_consensus,
    _derive_substitutions,
    _rcsb_url,
    align_reference_to_structure,
    build_library,
    download_rcsb,
    find_variable_positions,
    read_pdb_chains,
)


# ---------------------------------------------------------------------------
# read_pdb_chains
# ---------------------------------------------------------------------------

def test_read_pdb_chains_basic(tmp_path):
    pdb_text = textwrap.dedent("""\
        ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  9.67           N
        ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00 10.38           C
        ATOM     10  N   GLY A   2      25.000  24.000   3.000  1.00  8.00           N
        ATOM     11  CA  GLY A   2      24.000  23.000   2.500  1.00  9.00           C
        ATOM     20  N   VAL B   1      20.000  20.000   1.000  1.00  7.00           N
        ATOM     21  CA  VAL B   1      19.000  19.000   0.500  1.00  8.00           C
        ATOM     30  N   LEU B   2      18.000  18.000   0.000  1.00  6.00           N
        END
    """)
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_text)

    chains = read_pdb_chains(pdb_file)
    assert chains == {"A": "AG", "B": "VL"}


def test_read_pdb_chains_deduplicates_atoms(tmp_path):
    pdb_text = textwrap.dedent("""\
        ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
        ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
        ATOM      3  C   MET A   1      26.700  26.800   2.300  1.00  9.00           C
        ATOM      4  N   SER A   2      25.000  24.000   3.000  1.00  8.00           N
        END
    """)
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_text)

    chains = read_pdb_chains(pdb_file)
    assert chains["A"] == "MS"


def test_read_pdb_chains_unknown_residue(tmp_path):
    pdb_text = textwrap.dedent("""\
        ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  9.67           N
        ATOM      2  N   UNK A   2      25.000  24.000   3.000  1.00  8.00           N
        END
    """)
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_text)

    chains = read_pdb_chains(pdb_file)
    assert chains["A"] == "AX"


# ---------------------------------------------------------------------------
# find_variable_positions
# ---------------------------------------------------------------------------

def test_find_variable_positions_basic():
    seqs = ["ACDE", "ACFE", "ACDE"]
    assert find_variable_positions(seqs) == [2]


def test_find_variable_positions_all_same():
    seqs = ["ACDE", "ACDE", "ACDE"]
    assert find_variable_positions(seqs) == []


def test_find_variable_positions_multiple():
    seqs = ["ABCD", "XBCY", "ABCY"]
    assert find_variable_positions(seqs) == [0, 3]


def test_find_variable_positions_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        find_variable_positions(["AB", "ABC"])


def test_find_variable_positions_empty():
    assert find_variable_positions([]) == []


# ---------------------------------------------------------------------------
# _derive_substitutions
# ---------------------------------------------------------------------------

def test_derive_substitutions_basic():
    reference = "ACDE"
    seqs = ["ACDE", "AXDE", "ACYE"]
    var_pos = [1, 2]
    subs = _derive_substitutions(seqs, reference, var_pos)
    assert subs[0] == ()
    assert subs[1] == ((1, "X"),)
    assert subs[2] == ((2, "Y"),)


def test_derive_substitutions_multiple():
    reference = "ABCD"
    seqs = ["XBCY"]
    var_pos = [0, 3]
    subs = _derive_substitutions(seqs, reference, var_pos)
    assert subs[0] == ((0, "X"), (3, "Y"))


# ---------------------------------------------------------------------------
# _build_consensus
# ---------------------------------------------------------------------------

def test_build_consensus():
    seqs = ["ABCD", "XBCY", "ABCY"]
    var_pos = [0, 3]
    consensus = _build_consensus(seqs, var_pos)
    assert consensus[0] == "A"  # A appears 2x, X appears 1x
    assert consensus[1] == "B"  # invariant
    assert consensus[2] == "C"  # invariant
    assert consensus[3] == "Y"  # Y appears 2x, D appears 1x


# ---------------------------------------------------------------------------
# align_reference_to_structure
# ---------------------------------------------------------------------------

def test_align_same_length_exact():
    result = align_reference_to_structure("ACDE", "ACDE")
    assert result["same_length"] is True
    assert result["offset"] == 0
    assert result["mismatches"] == []


def test_align_same_length_mismatches():
    result = align_reference_to_structure("ACDE", "AXDY")
    assert result["same_length"] is True
    assert len(result["mismatches"]) == 2
    assert result["mismatches"][0] == (1, "X", "C")
    assert result["mismatches"][1] == (3, "Y", "E")


def test_align_different_length_good():
    ref = "BCDE"
    chain = "ABCDEF"
    result = align_reference_to_structure(ref, chain)
    assert result["same_length"] is False
    assert result["offset"] == 1
    assert result["mismatches"] == []


def test_align_different_length_bad():
    with pytest.raises(NotImplementedError, match="95%"):
        align_reference_to_structure("AAAA", "BBBBBB")


# ---------------------------------------------------------------------------
# build_library logic (using a fake DataFrame, no network)
# ---------------------------------------------------------------------------

def test_build_library_logic():
    import pandas as pd

    seqs_h = ["ACGT", "AXGT", "ACYT", "AXYT"]
    seqs_l = ["MMMM"] * 4
    scores = [1.0, 2.0, 3.0, 4.0]
    df = pd.DataFrame({
        "heavy_chain_seq": seqs_h,
        "light_chain_seq": seqs_l,
        "binding_score": scores,
    })

    var_pos = find_variable_positions(seqs_h)
    assert var_pos == [1, 2]

    consensus = _build_consensus(seqs_h, var_pos)
    assert consensus[1] in ("C", "X")
    assert consensus[2] in ("G", "Y")

    subs = _derive_substitutions(seqs_h, consensus, var_pos)
    n_muts = [len(s) for s in subs]
    assert min(n_muts) == 0
    assert max(n_muts) == 2


# ---------------------------------------------------------------------------
# Network integration test
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_4fqi_h1_integration(tmp_path):
    lib = build_library("4fqi_h1", cache_dir=tmp_path, chain="H")

    assert len(lib.frame) == 65094

    heavy_lengths = lib.frame["heavy_chain_seq"].str.len().unique()
    assert len(heavy_lengths) == 1
    assert heavy_lengths[0] == 121

    light_lengths = lib.frame["light_chain_seq"].str.len().unique()
    assert len(light_lengths) == 1
    assert light_lengths[0] == 109

    assert lib.frame["light_chain_seq"].nunique() == 1

    assert lib.variable_positions == [
        28, 29, 30, 51, 56, 57, 58, 70, 73, 74, 75, 76, 83, 86, 94, 105,
    ]

    for pos in lib.variable_positions:
        assert len(lib.alphabet_at[pos]) == 2, f"Position {pos} is not binary"

    n_mut_counts = lib.frame["n_mut"].value_counts().sort_index()
    assert n_mut_counts.index.min() == 0
    assert n_mut_counts.index.max() == 16
    # Unimodal: the peak should be near 8 (half of 16)
    peak = n_mut_counts.idxmax()
    assert 5 <= peak <= 11, f"Distribution peak at {peak}, expected near 8"

    censored = (lib.frame["binding_score"] == 7.0).sum()
    assert censored == 1675


# ---------------------------------------------------------------------------
# RCSB download
# ---------------------------------------------------------------------------

def test_rcsb_url_uppercase():
    assert _rcsb_url("3hfm") == "https://files.rcsb.org/download/3HFM.pdb"
    assert _rcsb_url("3HFM") == "https://files.rcsb.org/download/3HFM.pdb"


def test_download_rcsb_caches(tmp_path):
    pdb_content = b"ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00           N\nEND\n"

    class FakeResp:
        content = pdb_content
        def raise_for_status(self):
            pass

    call_count = 0

    def fake_get(url, timeout=None):
        nonlocal call_count
        call_count += 1
        return FakeResp()

    with patch("igv.data.requests.get", side_effect=fake_get):
        p1 = download_rcsb("3hfm", tmp_path)
        p2 = download_rcsb("3hfm", tmp_path)

    assert p1 == p2
    assert p1.name == "3hfm.pdb"
    assert p1.read_bytes() == pdb_content
    assert call_count == 1


def test_download_rcsb_creates_parent(tmp_path):
    nested = tmp_path / "a" / "b"

    class FakeResp:
        content = b"END\n"
        def raise_for_status(self):
            pass

    with patch("igv.data.requests.get", return_value=FakeResp()):
        p = download_rcsb("1abc", nested)

    assert p.exists()
    assert p.name == "1abc.pdb"


# ---------------------------------------------------------------------------
# build_library rejects unrecognised chains
# ---------------------------------------------------------------------------

def test_build_library_rejects_non_HL_chain():
    import pandas as pd

    df = pd.DataFrame({
        "heavy_chain_seq": ["ACGT"],
        "light_chain_seq": ["MMMM"],
        "binding_score": [1.0],
    })

    with patch("igv.data.load_affinity", return_value=df):
        with pytest.raises(ValueError, match="must be 'H' or 'L'"):
            build_library("fake", Path("/tmp"), chain="Y")


def test_build_library_accepts_H_and_L():
    import pandas as pd

    df = pd.DataFrame({
        "heavy_chain_seq": ["ACGT", "AXGT"],
        "light_chain_seq": ["MMMM", "MMMM"],
        "binding_score": [1.0, 2.0],
    })

    with patch("igv.data.load_affinity", return_value=df):
        lib_h = build_library("fake", Path("/tmp"), chain="H")
        assert lib_h.chain == "H"

    with patch("igv.data.load_affinity", return_value=df):
        lib_l = build_library("fake", Path("/tmp"), chain="L")
        assert lib_l.chain == "L"
