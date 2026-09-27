"""Tests for igv.dms — CPU-only unit tests, no network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from igv.dms import (
    AA_ORDER,
    DDG_PER_LOG10,
    DMS_COMPLEXES,
    DmsComplex,
    STARR2020_URL,
    binding_ddg,
    download_starr2020,
    enumerate_mutants,
    get_complex,
    interface_positions,
    map_sites_to_indices,
    resolve_pdb_complex,
    singles,
    substitution_matrix,
    within_position_correlation,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_singles_df(**kwargs) -> pd.DataFrame:
    """Small synthetic DMS-style frame."""
    base = {
        "site_SARS2": [417, 417, 417, 501, 501],
        "wildtype":   ["K",  "K",  "K",  "N",  "N"],
        "mutant":     ["A",  "K",  "V",  "Y",  "N"],   # "K"→"K" and "N"→"N" are synonymous
        "bind_avg":   [0.5, -0.2, np.nan, 1.1, 0.3],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# binding_ddg — sign convention
# ---------------------------------------------------------------------------

def test_binding_ddg_negative_input_gives_positive_ddg():
    """Weaker binding in Starr (negative bind_avg) must be positive ddG."""
    result = binding_ddg(-1.0)
    assert result > 0


def test_binding_ddg_negative_input_approx_value():
    """binding_ddg(-1.0) should equal +DDG_PER_LOG10 ≈ 1.3638."""
    assert binding_ddg(-1.0) == pytest.approx(DDG_PER_LOG10, rel=1e-6)
    assert binding_ddg(-1.0) == pytest.approx(1.3638, abs=1e-3)


def test_binding_ddg_positive_input_gives_negative_ddg():
    """Tighter binding in Starr (positive bind_avg) must be negative ddG."""
    assert binding_ddg(+1.0) < 0


def test_binding_ddg_positive_input_approx_value():
    assert binding_ddg(+1.0) == pytest.approx(-DDG_PER_LOG10, rel=1e-6)


def test_binding_ddg_zero_is_zero():
    """No change in affinity must map to zero ddG."""
    assert binding_ddg(0.0) == 0.0


def test_binding_ddg_round_trip():
    """Applying twice should recover the original value."""
    original = -2.5
    assert binding_ddg(binding_ddg(original) / DDG_PER_LOG10 * (-1)) == pytest.approx(
        binding_ddg(original), rel=1e-9
    )


def test_binding_ddg_negates_symmetrically():
    """binding_ddg(-x) == -binding_ddg(x) for any x."""
    x = 3.14
    assert binding_ddg(-x) == pytest.approx(-binding_ddg(x), rel=1e-12)


def test_binding_ddg_vectorises_over_array():
    """Must accept a numpy array and return an array of equal length."""
    arr = np.array([-1.0, 0.0, 1.0])
    result = binding_ddg(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == arr.shape
    assert result[0] == pytest.approx(DDG_PER_LOG10, rel=1e-6)
    assert result[1] == 0.0
    assert result[2] == pytest.approx(-DDG_PER_LOG10, rel=1e-6)


def test_binding_ddg_is_linear():
    """Doubling the input doubles the output (linearity check)."""
    assert binding_ddg(2.0) == pytest.approx(2 * binding_ddg(1.0), rel=1e-12)


# ---------------------------------------------------------------------------
# get_complex
# ---------------------------------------------------------------------------

def test_get_complex_spike_rbd_pdb_id():
    c = get_complex("spike_rbd")
    assert c.pdb_id == "6M0J"


def test_get_complex_spike_rbd_mutated_chain():
    c = get_complex("spike_rbd")
    assert c.mutated_chain == "E"


def test_get_complex_spike_rbd_partner_chains():
    c = get_complex("spike_rbd")
    assert c.partner_chains == ("A",)


def test_get_complex_spike_rbd_all_chains():
    c = get_complex("spike_rbd")
    assert "E" in c.all_chains
    assert "A" in c.all_chains
    assert len(c.all_chains) == 2


def test_get_complex_all_chains_mutated_first():
    """mutated_chain must appear before partner_chains in all_chains."""
    c = get_complex("spike_rbd")
    assert c.all_chains[0] == c.mutated_chain


def test_get_complex_unknown_key_raises():
    with pytest.raises(KeyError, match="Unknown DMS complex"):
        get_complex("nonexistent_key_zzz")


def test_get_complex_unknown_key_lists_valid_keys():
    """Error message should name at least one registered key."""
    with pytest.raises(KeyError, match="spike_rbd"):
        get_complex("nonexistent_key_zzz")


def test_get_complex_returns_dms_complex_instance():
    assert isinstance(get_complex("spike_rbd"), DmsComplex)


# ---------------------------------------------------------------------------
# download_starr2020 — caching behaviour (no network)
# ---------------------------------------------------------------------------

def test_download_starr2020_uses_starr2020_url(tmp_path, monkeypatch):
    """When the file is absent, must call _atomic_download with STARR2020_URL."""
    called_with: list[tuple] = []

    def fake_download(url: str, dest) -> None:
        called_with.append((url, dest))
        # Simulate the download by writing a placeholder file.
        dest.write_text("fake")

    monkeypatch.setattr("igv.dms._atomic_download", fake_download)
    download_starr2020(tmp_path)

    assert len(called_with) == 1
    url_used, dest_used = called_with[0]
    assert url_used == STARR2020_URL


def test_download_starr2020_destination_name(tmp_path, monkeypatch):
    """Destination filename must be single_mut_effects.csv."""
    seen_dest: list = []

    def fake_download(url, dest) -> None:
        seen_dest.append(dest)
        dest.write_text("fake")

    monkeypatch.setattr("igv.dms._atomic_download", fake_download)
    result = download_starr2020(tmp_path)
    assert result.name == "single_mut_effects.csv"


def test_download_starr2020_caches_skips_network(tmp_path, monkeypatch):
    """When the destination file already exists, no download should occur."""
    dest = tmp_path / "single_mut_effects.csv"
    dest.write_text("already cached")

    def should_not_be_called(url, dest) -> None:  # pragma: no cover
        raise AssertionError("_atomic_download was called despite cached file")

    monkeypatch.setattr("igv.dms._atomic_download", should_not_be_called)
    result = download_starr2020(tmp_path)
    assert result == dest


def test_download_starr2020_returns_path_object(tmp_path, monkeypatch):
    dest = tmp_path / "single_mut_effects.csv"
    dest.write_text("x")
    monkeypatch.setattr("igv.dms._atomic_download", lambda u, d: None)
    result = download_starr2020(tmp_path)
    from pathlib import Path
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# singles — filtering
# ---------------------------------------------------------------------------

def test_singles_drops_synonymous_rows():
    """Rows where mutant == wildtype must be removed."""
    df = _make_singles_df()
    result = singles(df)
    assert (result["mutant"] != result["wildtype"]).all()


def test_singles_drops_null_bind_avg():
    """Rows with NaN bind_avg must be removed."""
    df = _make_singles_df()
    result = singles(df)
    assert result["bind_avg"].notna().all()


def test_singles_keeps_genuine_mutations():
    """Non-synonymous rows with a valid bind_avg must survive."""
    df = _make_singles_df()
    result = singles(df)
    assert len(result) == 2  # K→A (bind_avg 0.5) and N→Y (bind_avg 1.1)


def test_singles_does_not_mutate_input():
    """The original DataFrame must be unchanged after calling singles()."""
    df = _make_singles_df()
    original_len = len(df)
    singles(df)
    assert len(df) == original_len


def test_singles_resets_index():
    """Output index must start at 0."""
    df = _make_singles_df()
    result = singles(df)
    assert list(result.index) == list(range(len(result)))


def test_singles_empty_input_returns_empty():
    df = pd.DataFrame({"site_SARS2": [], "wildtype": [], "mutant": [], "bind_avg": []})
    result = singles(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# map_sites_to_indices — site mapping and wild-type guard
# ---------------------------------------------------------------------------

def _make_map_df(sites, wildtypes, mutants=None):
    """Build a minimal DataFrame for map_sites_to_indices."""
    if mutants is None:
        mutants = ["A"] * len(sites)
    return pd.DataFrame({
        "site_SARS2": sites,
        "wildtype": wildtypes,
        "mutant": mutants,
    })


def test_map_sites_correct_mapping():
    """Correct sites with matching wild-type map to 0-based indices."""
    residue_ids = ["10", "11", "11A", "12"]
    sequence    = "AGKL"
    df = _make_map_df([10, 11, 12], ["A", "G", "L"])
    mapped, mismatches = map_sites_to_indices(df, residue_ids, sequence)
    assert mismatches == []
    assert mapped[10] == 0
    assert mapped[11] == 1
    assert mapped[12] == 3   # "12" is at index 3 (after insertion code "11A")


def test_map_sites_non_sequential_numbering_correct_index():
    """Insertion code "11A" between 11 and 12 must not shift the index of site 12."""
    residue_ids = ["10", "11", "11A", "12"]
    sequence    = "AGKL"
    df = _make_map_df([12], ["L"])
    mapped, _ = map_sites_to_indices(df, residue_ids, sequence)
    assert mapped[12] == 3


def test_map_sites_wt_mismatch_raises_by_default():
    """A wildtype letter that disagrees with the sequence raises ValueError."""
    residue_ids = ["10", "11"]
    sequence    = "AG"
    df = _make_map_df([10], ["X"])   # dataset says X but PDB has A
    with pytest.raises(ValueError, match="mismatch"):
        map_sites_to_indices(df, residue_ids, sequence)


def test_map_sites_wt_mismatch_allow_returns_mismatch():
    """allow_mismatch=True returns mismatches without raising."""
    residue_ids = ["10", "11"]
    sequence    = "AG"
    df = _make_map_df([10], ["X"])
    mapped, mismatches = map_sites_to_indices(df, residue_ids, sequence, allow_mismatch=True)
    assert len(mismatches) == 1
    site, expected_aa, found_aa = mismatches[0]
    assert site == 10
    assert expected_aa == "X"
    assert found_aa == "A"


def test_map_sites_wt_mismatch_not_in_mapped():
    """A mismatched site must not appear in the mapped dict, even with allow_mismatch."""
    residue_ids = ["10", "11"]
    sequence    = "AG"
    df = _make_map_df([10, 11], ["X", "G"])  # site 10 mismatches, site 11 is fine
    mapped, mismatches = map_sites_to_indices(df, residue_ids, sequence, allow_mismatch=True)
    assert 10 not in mapped
    assert 11 in mapped


def test_map_sites_absent_site_is_skipped():
    """A site not in residue_ids must be silently skipped, not raise."""
    residue_ids = ["10", "11"]
    sequence    = "AG"
    df = _make_map_df([10, 999], ["A", "A"])   # site 999 is absent
    mapped, mismatches = map_sites_to_indices(df, residue_ids, sequence)
    assert 999 not in mapped
    assert 10 in mapped
    assert mismatches == []


def test_map_sites_empty_df_returns_empty():
    residue_ids = ["10"]
    sequence    = "A"
    df = _make_map_df([], [])
    mapped, mismatches = map_sites_to_indices(df, residue_ids, sequence)
    assert mapped == {}
    assert mismatches == []


def test_map_sites_multiple_mismatches_raises_with_count():
    """Error message must mention the number of mismatches when > 1."""
    residue_ids = ["10", "11", "12"]
    sequence    = "AGK"
    df = _make_map_df([10, 11, 12], ["X", "Y", "K"])  # sites 10 and 11 mismatch
    with pytest.raises(ValueError, match="2"):
        map_sites_to_indices(df, residue_ids, sequence)


# ---------------------------------------------------------------------------
# substitution_matrix
# ---------------------------------------------------------------------------

def _make_sub_df():
    """Synthetic DMS frame with two positions and a handful of substitutions."""
    return pd.DataFrame({
        "site_SARS2": [417, 417, 417, 501, 501],
        "wildtype":   ["K",  "K",  "K",  "N",  "N"],
        "mutant":     ["A",  "C",  "D",  "Y",  "W"],
        "bind_avg":   [1.0,  2.0,  3.0, -1.0,  0.5],
    })


def test_substitution_matrix_shape():
    """Matrix must have shape (n_positions, 20)."""
    df = _make_sub_df()
    mat, positions, aa_order = substitution_matrix(df, value_col="bind_avg")
    assert mat.shape == (len(positions), 20)
    assert mat.shape[1] == 20


def test_substitution_matrix_row_order():
    """Rows must be in ascending sorted order of site_SARS2."""
    df = _make_sub_df()
    _, positions, _ = substitution_matrix(df, value_col="bind_avg")
    assert positions == sorted(positions)


def test_substitution_matrix_columns_follow_aa_order():
    """Column order must equal AA_ORDER."""
    _, _, aa_order = substitution_matrix(_make_sub_df(), value_col="bind_avg")
    assert aa_order == AA_ORDER


def test_substitution_matrix_measured_cell_value():
    """A measured substitution must hold exactly the correct value."""
    df = _make_sub_df()
    mat, positions, aa_order = substitution_matrix(df, value_col="bind_avg")
    row_417 = positions.index(417)
    col_A   = aa_order.index("A")
    assert mat[row_417, col_A] == pytest.approx(1.0)


def test_substitution_matrix_unmeasured_cell_is_nan():
    """An unmeasured (site, mutant) cell must be NaN, not 0.0."""
    df = _make_sub_df()
    mat, positions, aa_order = substitution_matrix(df, value_col="bind_avg")
    row_417 = positions.index(417)
    col_G   = aa_order.index("G")   # G not measured at site 417
    assert np.isnan(mat[row_417, col_G])


def test_substitution_matrix_wt_cell_is_nan():
    """The wild-type → wild-type cell must be NaN (no measurement in singles df)."""
    # singles() would have removed K→K; simulate that by not including it.
    df = _make_sub_df()
    mat, positions, aa_order = substitution_matrix(df, value_col="bind_avg")
    row_417 = positions.index(417)
    col_K   = aa_order.index("K")   # K is wildtype at 417, not in df
    assert np.isnan(mat[row_417, col_K])


def test_substitution_matrix_duplicate_site_mutant_last_value_wins():
    """Duplicate (site, mutant) rows: last row in DataFrame order takes priority.

    The code iterates rows in order and writes directly; later writes overwrite earlier ones.
    This test documents the actual deterministic behaviour rather than asserting a raise.
    """
    df = pd.DataFrame({
        "site_SARS2": [417, 417],
        "wildtype":   ["K",  "K"],
        "mutant":     ["A",  "A"],
        "bind_avg":   [1.0,  99.0],
    })
    mat, positions, aa_order = substitution_matrix(df, value_col="bind_avg")
    row_417 = positions.index(417)
    col_A   = aa_order.index("A")
    # Last row wins: value should be 99.0
    assert mat[row_417, col_A] == pytest.approx(99.0)


def test_substitution_matrix_positions_list_length():
    """Returned positions list length must equal number of matrix rows."""
    df = _make_sub_df()
    mat, positions, _ = substitution_matrix(df, value_col="bind_avg")
    assert len(positions) == mat.shape[0]


# ---------------------------------------------------------------------------
# within_position_correlation
# ---------------------------------------------------------------------------

def _make_property_and_matrix():
    """Build a 4-row x 20-col matrix and matching property dict for testing.

    Row 0: values strictly increasing with property (rank correlation = +1.0)
    Row 1: values strictly decreasing with property (rank correlation = -1.0)
    Row 2: constant values (std = 0 → skip → NaN)
    Row 3: only 4 non-NaN values (< 5 threshold → skip → NaN)
    """
    # Use first 6 AAs in AA_ORDER: A C D E F G (indices 0–5)
    # Property: A=1, C=2, D=3, E=4, F=5, G=6; rest missing.
    aa_vals = {aa: float(i + 1) for i, aa in enumerate(AA_ORDER[:6])}

    mat = np.full((4, 20), np.nan)
    # Row 0: increasing — A=10, C=20, D=30, E=40, F=50, G=60
    for i, aa in enumerate(AA_ORDER[:6]):
        mat[0, i] = (i + 1) * 10.0
    # Row 1: decreasing — A=60, C=50, D=40, E=30, F=20, G=10
    for i, aa in enumerate(AA_ORDER[:6]):
        mat[1, i] = (6 - i) * 10.0
    # Row 2: constant — all 6 AAs have same value
    for i in range(6):
        mat[2, i] = 5.0
    # Row 3: only 4 non-NaN values (A, C, D, E)
    for i in range(4):
        mat[3, i] = float(i)

    return mat, aa_vals


def test_within_position_correlation_perfect_positive():
    """A perfectly rank-ordered row must yield Spearman correlation +1.0."""
    mat, props = _make_property_and_matrix()
    result = within_position_correlation(mat, props)
    assert result[0] == pytest.approx(1.0, abs=1e-9)


def test_within_position_correlation_perfect_negative():
    """A perfectly reversed row must yield Spearman correlation -1.0."""
    mat, props = _make_property_and_matrix()
    result = within_position_correlation(mat, props)
    assert result[1] == pytest.approx(-1.0, abs=1e-9)


def test_within_position_correlation_constant_row_is_nan():
    """A constant-value row has zero variance; result must be NaN, not 0.0."""
    mat, props = _make_property_and_matrix()
    result = within_position_correlation(mat, props)
    assert np.isnan(result[2])


def test_within_position_correlation_too_few_values_is_nan():
    """A row with fewer than 5 non-NaN values must yield NaN."""
    mat, props = _make_property_and_matrix()
    result = within_position_correlation(mat, props)
    assert np.isnan(result[3])


def test_within_position_correlation_length_matches_row_count():
    """Output length must equal the number of matrix rows, including NaN rows."""
    mat, props = _make_property_and_matrix()
    result = within_position_correlation(mat, props)
    assert len(result) == mat.shape[0]
    assert len(result) == 4


def test_within_position_correlation_nans_do_not_propagate():
    """NaN rows must not affect the correlation of valid rows."""
    mat, props = _make_property_and_matrix()
    result = within_position_correlation(mat, props)
    # Rows 0 and 1 are valid; check they are finite even though rows 2 and 3 are NaN
    assert np.isfinite(result[0])
    assert np.isfinite(result[1])


def test_within_position_correlation_returns_array():
    """Return type must be a numpy ndarray."""
    mat, props = _make_property_and_matrix()
    result = within_position_correlation(mat, props)
    assert isinstance(result, np.ndarray)


def test_within_position_correlation_empty_props_all_nan():
    """If no amino acid has a property value, all positions should be NaN."""
    mat, _ = _make_property_and_matrix()
    result = within_position_correlation(mat, {})
    assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

def test_aa_order_has_20_amino_acids():
    assert len(AA_ORDER) == 20


def test_aa_order_no_duplicates():
    assert len(set(AA_ORDER)) == 20


def test_aa_order_standard_amino_acids():
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    assert set(AA_ORDER) == standard


def test_ddg_per_log10_approx():
    assert DDG_PER_LOG10 == pytest.approx(1.3638, abs=1e-3)


def test_starr2020_url_is_string():
    assert isinstance(STARR2020_URL, str)
    assert STARR2020_URL.startswith("https://")


def test_dms_complexes_registry_not_empty():
    assert len(DMS_COMPLEXES) >= 1
    assert "spike_rbd" in DMS_COMPLEXES


# ---------------------------------------------------------------------------
# interface_positions
# ---------------------------------------------------------------------------

# Synthetic two-chain PDB with controlled geometry.
#
# Chain E:
#   Residue "1": CA at (1, 0, 0)           -> dist to chain A = 1.0  (near)
#   Residue "2": CA at (50, 0, 0)          -> dist to chain A = 50.0 (far)
#   Residue "3": CA at (20, 0, 0),
#                OG at ( 2, 0, 0)          -> min dist = 2.0 via side-chain (near)
# Chain A:
#   Residue "1": CA at ( 0, 0, 0)
#
# residue_ids = ["1", "2", "3"] (for chain E)
# With cutoff=5.0:  interface = [0, 2]
# With cutoff=1.5:  interface = [0]   (superset monotonicity holds)

def _pdb_atom_line(serial, name, chain, resseq, x, y, z, element="C"):
    """Build one PDB ATOM record with correct column layout."""
    return (
        f"ATOM  {serial:5d} {name:4s} ALA {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


_INTERFACE_PDB = (
    _pdb_atom_line(1, " CA ", "E", 1,  1.0, 0.0, 0.0)
    + _pdb_atom_line(2, " CA ", "E", 2, 50.0, 0.0, 0.0)
    + _pdb_atom_line(3, " CA ", "E", 3, 20.0, 0.0, 0.0)
    + _pdb_atom_line(4, " OG ", "E", 3,  2.0, 0.0, 0.0, "O")
    + _pdb_atom_line(5, " CA ", "A", 1,  0.0, 0.0, 0.0)
    + "END\n"
)

# Minimal PDB with chain E and A for resolve_pdb_complex (DMS path).
_DMS_PDB = (
    "ATOM      1  CA  ALA E   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      2  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)

# Minimal PDB with chains A and B for resolve_pdb_complex (SKEMPI path).
_SKEMPI_PDB = (
    "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      2  CA  GLY B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)


@pytest.fixture
def interface_pdb(tmp_path):
    p = tmp_path / "interface.pdb"
    p.write_text(_INTERFACE_PDB)
    return p


def test_interface_positions_near_residues_returned(interface_pdb):
    """Residues within cutoff of any partner atom must appear in the result."""
    result = interface_positions(
        interface_pdb,
        chain="E",
        partner_chains=("A",),
        residue_ids=["1", "2", "3"],
        cutoff=5.0,
    )
    assert 0 in result  # residue "1" (index 0) at distance 1.0
    assert 2 in result  # residue "3" (index 2) at distance 2.0 via side-chain OG


def test_interface_positions_far_residue_excluded(interface_pdb):
    """Residues beyond the cutoff must not appear in the result."""
    result = interface_positions(
        interface_pdb,
        chain="E",
        partner_chains=("A",),
        residue_ids=["1", "2", "3"],
        cutoff=5.0,
    )
    assert 1 not in result  # residue "2" (index 1) at distance 50.0


def test_interface_positions_exact_result_at_5a(interface_pdb):
    """Result must be exactly [0, 2] at cutoff=5.0 Å."""
    result = interface_positions(
        interface_pdb,
        chain="E",
        partner_chains=("A",),
        residue_ids=["1", "2", "3"],
        cutoff=5.0,
    )
    assert result == [0, 2]


def test_interface_positions_larger_cutoff_is_superset(interface_pdb):
    """A larger cutoff must return a superset of any smaller cutoff result."""
    small = set(
        interface_positions(
            interface_pdb,
            chain="E",
            partner_chains=("A",),
            residue_ids=["1", "2", "3"],
            cutoff=1.5,
        )
    )
    large = set(
        interface_positions(
            interface_pdb,
            chain="E",
            partner_chains=("A",),
            residue_ids=["1", "2", "3"],
            cutoff=5.0,
        )
    )
    assert small.issubset(large)
    assert len(large) >= len(small)


def test_interface_positions_side_chain_atom_counts(interface_pdb):
    """A residue near via a side-chain atom (not just CA) must be included.

    Residue 3 (index 2): CA is at 20 Å, OG is at 2 Å from chain A.
    The function is heavy-atom based, so OG should bring it within cutoff=5.0.
    """
    result = interface_positions(
        interface_pdb,
        chain="E",
        partner_chains=("A",),
        residue_ids=["1", "2", "3"],
        cutoff=5.0,
    )
    assert 2 in result


def test_interface_positions_empty_partner_chain(interface_pdb):
    """Empty partner_chains must return no interface positions."""
    result = interface_positions(
        interface_pdb,
        chain="E",
        partner_chains=(),
        residue_ids=["1", "2", "3"],
        cutoff=5.0,
    )
    assert result == []


def test_interface_positions_cutoff_zero_returns_empty(interface_pdb):
    """A cutoff of 0 must return no positions (no atom is at exactly distance 0)."""
    result = interface_positions(
        interface_pdb,
        chain="E",
        partner_chains=("A",),
        residue_ids=["1", "2", "3"],
        cutoff=0.0,
    )
    assert result == []


# ---------------------------------------------------------------------------
# enumerate_mutants
# ---------------------------------------------------------------------------

def _make_enum_df():
    """Synthetic DMS frame for enumerate_mutants tests.

    sequence = "KNV"
    site_to_idx = {417: 0, 501: 1, 600: 2}
    positions = [0, 2]

    Expected non-synonymous pairs at positions [0, 2]:
      (0, "A") from site 417, mutant "A" (K -> A is non-synonymous)
      (2, "A") from site 600, mutant "A" (V -> A is non-synonymous)
    """
    return pd.DataFrame({
        "site_SARS2": [417, 417, 501, 600, 999],
        "mutant":     ["A",  "K",  "Y",  "A",  "T"],
    })


_ENUM_SEQUENCE = "KNV"
_ENUM_SITE_TO_IDX = {417: 0, 501: 1, 600: 2}


def test_enumerate_mutants_synonymous_rows_excluded():
    """Rows where the mutant equals the wildtype at that position must be excluded."""
    df = _make_enum_df()
    # site 417 -> idx 0 -> sequence[0] = "K"; mutant "K" is synonymous
    result = enumerate_mutants(
        df, [0, 1, 2], sequence=_ENUM_SEQUENCE, site_to_idx=_ENUM_SITE_TO_IDX
    )
    assert (0, "K") not in result


def test_enumerate_mutants_positions_not_in_list_excluded():
    """Pairs at positions not in the requested list must be excluded."""
    df = _make_enum_df()
    # position 1 (site 501) is NOT in [0, 2]; N->Y should be excluded
    result = enumerate_mutants(
        df, [0, 2], sequence=_ENUM_SEQUENCE, site_to_idx=_ENUM_SITE_TO_IDX
    )
    assert (1, "Y") not in result


def test_enumerate_mutants_absent_site_skipped():
    """Sites absent from site_to_idx must be silently skipped, not raise."""
    df = pd.DataFrame({"site_SARS2": [999], "mutant": ["A"]})
    # site 999 is not in site_to_idx, so it should be skipped without crashing
    result = enumerate_mutants(
        df, [0, 1], sequence="KN", site_to_idx={417: 0, 501: 1}
    )
    assert result == []


def test_enumerate_mutants_deterministic_ordering():
    """Result must be sorted regardless of DataFrame row order."""
    df_forward = pd.DataFrame({
        "site_SARS2": [417, 600],
        "mutant":     ["A",  "A"],
    })
    df_reversed = pd.DataFrame({
        "site_SARS2": [600, 417],
        "mutant":     ["A",  "A"],
    })
    r_fwd = enumerate_mutants(
        df_forward, [0, 2], sequence=_ENUM_SEQUENCE, site_to_idx=_ENUM_SITE_TO_IDX
    )
    r_rev = enumerate_mutants(
        df_reversed, [0, 2], sequence=_ENUM_SEQUENCE, site_to_idx=_ENUM_SITE_TO_IDX
    )
    assert r_fwd == r_rev
    assert r_fwd == [(0, "A"), (2, "A")]


def test_enumerate_mutants_exact_result():
    """Full expected list is precisely [(0, 'A'), (2, 'A')] for the standard frame."""
    df = _make_enum_df()
    result = enumerate_mutants(
        df, [0, 2], sequence=_ENUM_SEQUENCE, site_to_idx=_ENUM_SITE_TO_IDX
    )
    assert result == [(0, "A"), (2, "A")]


def test_enumerate_mutants_empty_positions():
    """An empty positions list must return an empty list."""
    df = _make_enum_df()
    result = enumerate_mutants(
        df, [], sequence=_ENUM_SEQUENCE, site_to_idx=_ENUM_SITE_TO_IDX
    )
    assert result == []


# ---------------------------------------------------------------------------
# resolve_pdb_complex
# ---------------------------------------------------------------------------

def test_resolve_pdb_complex_dms_data_source(tmp_path, monkeypatch):
    """A registered DMS name must return data_source == 'dms'."""
    pdb_file = tmp_path / "6m0j.pdb"
    pdb_file.write_text(_DMS_PDB)
    monkeypatch.setattr("igv.data.download_rcsb", lambda pdb_id, cache_dir: pdb_file)
    result = resolve_pdb_complex("spike_rbd", "E", tmp_path)
    assert result.data_source == "dms"


def test_resolve_pdb_complex_dms_pdb_path_returned(tmp_path, monkeypatch):
    """Resolved complex must carry the path returned by download_rcsb."""
    pdb_file = tmp_path / "6m0j.pdb"
    pdb_file.write_text(_DMS_PDB)
    monkeypatch.setattr("igv.data.download_rcsb", lambda pdb_id, cache_dir: pdb_file)
    result = resolve_pdb_complex("spike_rbd", "E", tmp_path)
    assert result.pdb_path == pdb_file


def test_resolve_pdb_complex_skempi_data_source(tmp_path, monkeypatch):
    """A registered SKEMPI name must return data_source == 'skempi'."""
    pdb_file = tmp_path / "1jtg.pdb"
    pdb_file.write_text(_SKEMPI_PDB)
    monkeypatch.setattr("igv.data.download_rcsb", lambda pdb_id, cache_dir: pdb_file)
    result = resolve_pdb_complex("1JTG", "A", tmp_path)
    assert result.data_source == "skempi"


def test_resolve_pdb_complex_unknown_raises_key_error(tmp_path):
    """A dataset in neither registry must raise KeyError."""
    with pytest.raises(KeyError):
        resolve_pdb_complex("completely_unknown_dataset_xyz", "A", tmp_path)


def test_resolve_pdb_complex_unknown_error_mentions_both_registries(tmp_path):
    """The KeyError message must name both the DMS and SKEMPI registries."""
    with pytest.raises(KeyError, match="(?s)DMS.*SKEMPI|SKEMPI.*DMS"):
        resolve_pdb_complex("completely_unknown_dataset_xyz", "A", tmp_path)


def test_resolve_pdb_complex_chain_not_in_subset_raises(tmp_path, monkeypatch):
    """A chain not in the registered subset must raise ValueError, not succeed silently."""
    pdb_file = tmp_path / "6m0j.pdb"
    pdb_file.write_text(_DMS_PDB)
    monkeypatch.setattr("igv.data.download_rcsb", lambda pdb_id, cache_dir: pdb_file)
    with pytest.raises(ValueError, match="not in registered subset"):
        resolve_pdb_complex("spike_rbd", "X", tmp_path)  # "X" not in {"E", "A"}
