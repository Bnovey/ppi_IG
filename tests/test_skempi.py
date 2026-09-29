"""Tests for igv.skempi — CPU-only unit tests plus one CSV integration test."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from igv.skempi import (
    HYDROPHOBICITY_KD,
    RESIDUE_VOLUME,
    SKEMPI_COMPLEXES,
    SKEMPI_MUTATION_COL,
    Mutation,
    SkempiComplex,
    add_ddg,
    antibody_antigen,
    compute_burial,
    compute_confounds,
    compute_distance_to_partner,
    ddg,
    filter_complex,
    get_complex,
    map_mutations_to_indices,
    parse_mutation,
    parse_mutations,
    parse_pdb_heavy_atoms,
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
        "#Pdb": ["1VFB_AB_C", "1vfb_AB_C", "3HFM_HL_Y", "1JTG_A_B"],
        "Mutation(s)_cleaned": ["LA38G", "LA38G,SA100AG", "LH50A", "MA10G"],
        "Mutation(s)_PDB": ["LA38G", "LA38G,SA100AG", "LH50A", "MA10G"],
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
    assert c.partner1 == ("H", "L")
    assert c.partner2 == ("Y",)
    assert c.all_chains == ("H", "L", "Y")
    assert c.note == "HyHEL-10 / HEW lysozyme"


def test_get_complex_case_insensitive():
    assert get_complex("3hfm") == get_complex("3HFM")


def test_get_complex_1jtg_subset_excludes_cd():
    c = get_complex("1JTG")
    assert c.all_chains == ("A", "B")
    assert "C" not in c.all_chains
    assert "D" not in c.all_chains


def test_get_complex_unknown():
    with pytest.raises(KeyError, match="Unknown SKEMPI complex"):
        get_complex("9ZZZ")


_STAGE4_COMPLEXES = [
    "1JTG", "3HFM", "1VFB", "1JRH", "2JEL",
    "1BRS", "4G0N", "1LFD", "1AO7", "1DQJ", "1DVF", "3S9D",
]


@pytest.mark.parametrize("pdb_id", _STAGE4_COMPLEXES)
def test_stage4_complexes_resolve(pdb_id):
    cx = get_complex(pdb_id)
    assert isinstance(cx, SkempiComplex)
    assert cx.pdb_id == pdb_id


@pytest.mark.parametrize("pdb_id", list(SKEMPI_COMPLEXES))
def test_partner_chains_disjoint_and_nonempty(pdb_id):
    cx = SKEMPI_COMPLEXES[pdb_id]
    assert len(cx.partner1) > 0
    assert len(cx.partner2) > 0
    assert set(cx.partner1).isdisjoint(set(cx.partner2))


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


@pytest.mark.skipif(not _SKEMPI_CSV.exists(), reason="SKEMPI CSV not on disk")
@pytest.mark.parametrize("pdb_id", list(SKEMPI_COMPLEXES))
def test_mutation_chains_covered_by_partners(pdb_id):
    """Every chain referenced in Mutation(s)_PDB must appear in partner1 or partner2."""
    import re

    cx = SKEMPI_COMPLEXES[pdb_id]
    all_chains = set(cx.all_chains)

    df = pd.read_csv(_SKEMPI_CSV, sep=";")
    rows = df[df["#Pdb"].str.split("_").str[0].str.upper() == pdb_id]
    assert len(rows) > 0, f"No SKEMPI rows for {pdb_id}"

    mut_re = re.compile(r"^([A-Z])([A-Za-z])(.+?)([A-Z])$")
    uncovered: set[str] = set()
    for raw in rows["Mutation(s)_PDB"].dropna():
        for part in raw.split(","):
            m = mut_re.match(part.strip())
            if m:
                chain = m.group(2)
                if chain not in all_chains:
                    uncovered.add(chain)

    assert not uncovered, (
        f"{pdb_id}: mutation chain(s) {uncovered} not in partners "
        f"{cx.partner1} + {cx.partner2}"
    )


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
_PDB_1JTG = Path(__file__).resolve().parent.parent / "data" / "raw" / "1jtg.pdb"


@pytest.mark.skipif(not _PDB_3HFM.exists(), reason="3hfm.pdb not on disk")
def test_read_pdb_residue_ids_real_3hfm():
    from igv.data import read_pdb_chains
    ids, seqs = read_pdb_residue_ids(_PDB_3HFM)
    ref = read_pdb_chains(_PDB_3HFM)
    for ch in ("H", "L", "Y"):
        assert ch in ids
        assert len(ids[ch]) == len(ref[ch])
        assert seqs[ch] == ref[ch]


# ---------------------------------------------------------------------------
# Regression: Mutation(s)_PDB vs Mutation(s)_cleaned column
# ---------------------------------------------------------------------------


def test_skempi_mutation_col_is_pdb():
    """The module constant must point to the PDB-numbered column."""
    assert SKEMPI_MUTATION_COL == "Mutation(s)_PDB"


def test_single_point_uses_pdb_column():
    """single_point must work off the PDB column, not cleaned."""
    df = pd.DataFrame({
        "Mutation(s)_cleaned": ["LA38G", "LA38G,SA100AG"],
        "Mutation(s)_PDB": ["LA50G", "LA50G,SA120AG"],
    })
    result = single_point(df)
    assert len(result) == 1
    # Verify the PDB column value survived (not just cleaned)
    assert result.iloc[0]["Mutation(s)_PDB"] == "LA50G"


def test_pdb_column_diverges_from_cleaned():
    """When the two columns disagree, code must use Mutation(s)_PDB.

    Simulates a protein like TEM-1 beta-lactamase (1JTG) where Ambler
    numbering differs from sequential numbering.
    """
    residue_ids = ["104", "238"]
    sequence = "EG"

    # The PDB column uses author numbering that matches the structure.
    pdb_mut = parse_mutation("EA104K")
    mapped, mismatches = map_mutations_to_indices(
        [pdb_mut], residue_ids, sequence,
    )
    assert len(mapped) == 1
    assert len(mismatches) == 0
    assert mapped[0] == (pdb_mut, 0)

    # The cleaned column uses sequential numbering -- wrong residue number.
    cleaned_mut = parse_mutation("EA79K")
    with pytest.raises(ValueError, match="not found in the PDB"):
        map_mutations_to_indices([cleaned_mut], residue_ids, sequence)


@pytest.mark.skipif(
    not (_SKEMPI_CSV.exists() and _PDB_1JTG.exists()),
    reason="SKEMPI CSV or 1jtg.pdb not on disk",
)
def test_1jtg_pdb_column_zero_mismatches():
    """All 138 single-point 1JTG mutations from Mutation(s)_PDB must match
    the deposited PDB wild-type residues with zero mismatches.

    This test fails if mutations are read from Mutation(s)_cleaned, because
    that column uses sequential numbering which does not match the Ambler
    numbering in the deposited 1JTG structure.
    """
    from igv.skempi import add_ddg, filter_complex, single_point

    df = pd.read_csv(_SKEMPI_CSV, sep=";")
    df = filter_complex(df, "1JTG")
    df = single_point(df)
    df = add_ddg(df)

    assert len(df) == 138, f"Expected 138 single-point 1JTG mutations, got {len(df)}"

    residue_ids, sequences = read_pdb_residue_ids(_PDB_1JTG)

    total_mapped = 0
    total_mismatches = 0
    for _, row in df.iterrows():
        mut = parse_mutation(row[SKEMPI_MUTATION_COL].strip())
        ch = mut.chain
        assert ch in residue_ids, f"Chain {ch} not in PDB"
        mapped, mismatches = map_mutations_to_indices(
            [mut], residue_ids[ch], sequences[ch], allow_mismatch=True,
        )
        total_mapped += len(mapped)
        total_mismatches += len(mismatches)

    assert total_mismatches == 0, (
        f"{total_mismatches} wild-type mismatches mapping 1JTG mutations "
        f"from {SKEMPI_MUTATION_COL} -- wrong column?"
    )
    assert total_mapped == 138


@pytest.mark.skipif(
    not (_SKEMPI_CSV.exists() and _PDB_3HFM.exists()),
    reason="SKEMPI CSV or 3hfm.pdb not on disk",
)
def test_3hfm_pdb_column_zero_mismatches():
    """3HFM: 96 single-point mutations, zero mismatches from Mutation(s)_PDB."""
    from igv.skempi import add_ddg, filter_complex, single_point

    df = pd.read_csv(_SKEMPI_CSV, sep=";")
    df = filter_complex(df, "3HFM")
    df = single_point(df)
    df = add_ddg(df)

    assert len(df) == 96

    residue_ids, sequences = read_pdb_residue_ids(_PDB_3HFM)

    total_mapped = 0
    total_mismatches = 0
    for _, row in df.iterrows():
        mut = parse_mutation(row[SKEMPI_MUTATION_COL].strip())
        ch = mut.chain
        mapped, mismatches = map_mutations_to_indices(
            [mut], residue_ids[ch], sequences[ch], allow_mismatch=True,
        )
        total_mapped += len(mapped)
        total_mismatches += len(mismatches)

    assert total_mismatches == 0
    assert total_mapped == 96


# ---------------------------------------------------------------------------
# Structural confound unit tests
# ---------------------------------------------------------------------------


def test_compute_burial_known_geometry():
    """Three atoms along a line: burial counts neighbours within radius."""
    coords = np.array([[0, 0, 0], [5, 0, 0], [20, 0, 0]], dtype=np.float64)
    chains = ["A", "A", "A"]
    res_keys = ["1", "2", "3"]
    burial = compute_burial(coords, chains, res_keys, "A", ["1", "2", "3"], radius=10.0)
    assert burial[0] == 1  # atom at (5,0,0) is within 10 A
    assert burial[1] == 1  # atom at (0,0,0) is within 10 A
    assert burial[2] == 0  # both other atoms are > 10 A away


def test_compute_burial_multi_atom_residue():
    """Residue with two atoms; neighbour within range of one counts."""
    coords = np.array([
        [0, 0, 0], [1, 0, 0],   # residue "1" (two atoms)
        [8, 0, 0],               # residue "2" (one atom, 8 A from atom 2 of res 1)
    ], dtype=np.float64)
    chains = ["A", "A", "A"]
    res_keys = ["1", "1", "2"]
    burial = compute_burial(coords, chains, res_keys, "A", ["1", "2"], radius=10.0)
    assert burial[0] == 1  # residue "2"'s atom is within 10 A of (1,0,0)
    assert burial[1] == 2  # both atoms of residue "1" are within 10 A of (8,0,0)


def test_compute_distance_to_partner_two_chains():
    """Min distance from chain A residue to chain B atoms."""
    coords = np.array([
        [0, 0, 0],    # chain A, res 1
        [3, 0, 0],    # chain A, res 2
        [10, 0, 0],   # chain B, res 1
        [12, 0, 0],   # chain B, res 2
    ], dtype=np.float64)
    chains = ["A", "A", "B", "B"]
    res_keys = ["1", "2", "1", "2"]
    dist = compute_distance_to_partner(
        coords, chains, res_keys, "A", ("B",), ["1", "2"],
    )
    assert dist[0] == pytest.approx(10.0)
    assert dist[1] == pytest.approx(7.0)


def test_compute_distance_to_partner_closest_atom():
    """Distance picks the minimum across all atom pairs."""
    coords = np.array([
        [0, 0, 0], [2, 0, 0],   # chain A, res 1 (two atoms)
        [5, 0, 0],               # chain B, res 1
    ], dtype=np.float64)
    chains = ["A", "A", "B"]
    res_keys = ["1", "1", "1"]
    dist = compute_distance_to_partner(
        coords, chains, res_keys, "A", ("B",), ["1"],
    )
    assert dist[0] == pytest.approx(3.0)  # min(5, 3) = 3


def test_amino_acid_scales_complete():
    """Both scales cover all 20 standard amino acids."""
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    assert set(HYDROPHOBICITY_KD.keys()) == standard
    assert set(RESIDUE_VOLUME.keys()) == standard


def test_parse_pdb_heavy_atoms(mini_pdb_path):
    """parse_pdb_heavy_atoms returns correct shape and chain assignments."""
    coords, chains, res_keys = parse_pdb_heavy_atoms(mini_pdb_path)
    assert coords.shape[1] == 3
    assert len(chains) == coords.shape[0]
    assert len(res_keys) == coords.shape[0]
    assert set(chains) == {"A", "B"}


# ---------------------------------------------------------------------------
# Integration test: confound panel on 1JTG chain B
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_SKEMPI_CSV.exists() and _PDB_1JTG.exists()),
    reason="SKEMPI CSV or 1jtg.pdb not on disk",
)
def test_1jtg_confound_panel():
    """Run confound panel on 1JTG chain B with synthetic random gradient.

    Reports each confound's Spearman correlation with |ddG| -- these
    numbers set the bar the gradient must clear.
    """
    from igv.metrics import auroc, auprc, partial_spearman, spearman

    df = pd.read_csv(_SKEMPI_CSV, sep=";")
    df = filter_complex(df, "1JTG")
    df = single_point(df)
    df = add_ddg(df)

    chain = "B"
    residue_ids, sequences = read_pdb_residue_ids(_PDB_1JTG)
    chain_ids = residue_ids[chain]
    chain_seq = sequences[chain]

    mutations = [parse_mutation(m.strip()) for m in df[SKEMPI_MUTATION_COL]]
    chain_mutations = [(mut, i) for i, mut in enumerate(mutations) if mut.chain == chain]
    chain_muts = [m for m, _ in chain_mutations]
    chain_ddgs = [float(df.iloc[i]["ddg_kcal_mol"]) for _, i in chain_mutations]

    mapped, _ = map_mutations_to_indices(chain_muts, chain_ids, chain_seq)

    pos_ddgs: dict[int, list[float]] = {}
    for mut, seq_idx in mapped:
        orig_pos = chain_muts.index(mut)
        pos_ddgs.setdefault(seq_idx, []).append(chain_ddgs[orig_pos])

    positions = sorted(pos_ddgs)
    ddg_max = np.array([max(pos_ddgs[p]) for p in positions])
    ddg_abs_max = np.abs(ddg_max)
    hot_labels = (ddg_max >= 2.0).astype(float)

    complex_info = get_complex("1JTG")
    partner_chains = complex_info.partner1

    confounds = compute_confounds(
        _PDB_1JTG, chain, chain_ids, chain_seq, partner_chains, positions,
    )

    assert len(positions) >= 25
    assert "burial" in confounds
    assert "distance_to_partner" in confounds
    assert "hydrophobicity" in confounds
    assert "residue_volume" in confounds
    assert "norm_position" in confounds

    print("\n--- 1JTG chain B confound correlations with |ddG| ---")
    print(f"Positions: {len(positions)}, hot spots: {int(hot_labels.sum())}")
    for name in sorted(confounds):
        vals = confounds[name]
        rho = spearman(vals, ddg_abs_max)
        print(f"  {name:25s}  rho = {rho:+.4f}")

    rng = np.random.default_rng(42)
    fake_grad = rng.standard_normal(len(positions))

    confound_matrix = np.column_stack(
        [confounds[k] for k in sorted(confounds)]
    )
    partial_rho = partial_spearman(fake_grad, ddg_abs_max, confound_matrix)
    rho_grad = spearman(fake_grad, ddg_abs_max)
    print(f"\n  random gradient vs |ddG|:  simple {rho_grad:+.4f}  partial {partial_rho:+.4f}")
    print(f"  AUROC (random grad):      {auroc(fake_grad, hot_labels):.4f}")
    print(f"  AUPRC (random grad):      {auprc(fake_grad, hot_labels):.4f}")
