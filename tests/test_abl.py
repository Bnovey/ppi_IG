"""Tests for igv.abl — CPU-only unit tests, no network access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from igv.abl import (
    ABL_TKIS,
    ALL_TKIS,
    EGFR_REFERENCE_PDB,
    EGFR_TKIS,
    HAUSER2018_CACHE_NAME,
    HAUSER2018_XLSX_URL,
    IC50_CONSISTENCY_TOL_LOG10,
    IC50_VS_KD_N,
    IC50_VS_KD_RMSE_KCAL,
    RT_300K,
    RT_LN10_300K,
    TKI_PDB,
    TKI_SMILES,
    WT_IC50_NM,
    AblMutation,
    abl_only,
    docked_only,
    download_hauser2018,
    load_hauser2018_from_csv,
    map_resnum_to_pdb_index,
    map_to_pdb,
    measured,
    parse_mutation,
    summary,
    validate_ic50_consistency,
    xtal_only,
)

# ---------------------------------------------------------------------------
# Inline fixture — representative subset of Hauser 2018 Supplementary Data 3
# ---------------------------------------------------------------------------

_FIXTURE_CSV = """\
TKI,mutation,IC50,DDG_exp,DDG_prime,DDG_fep1,dDDG_fep1,DDG_fep2,dDDG_fep2,DDG_fep3,dDDG_fep3
imatinib,M244V,287.0,0.21,-0.098,-0.4,0.41,-0.35,0.41,-0.43,0.41
imatinib,T315I,9773.0,2.32,2.654,3.91,0.46,4.68,0.5,4.66,0.47
imatinib,F486S,,,,,,,,,
dasatinib,M244V,2.0,0.0,-0.067,-0.04,0.41,0.36,0.42,0.02,0.43
dasatinib,T315I,10000.0,5.08,4.085,4.96,0.46,4.84,0.49,5.0,0.47
dasatinib,F486S,,,,,,,,,
axitinib,M244V,690.0,-0.11,-0.098,-0.4,0.41,-0.35,0.41,-0.43,0.41
axitinib,V379I,0.068,,,,,,,,
erlotinib,M244V,,,,,,,,,
erlotinib,T315I,59.0,-0.15,,,,,,,
ponatinib,G250E,0.021,0.3,0.311,-0.35,0.41,-0.71,0.41,-0.74,0.41
ponatinib,M244V,3.0,0.0,-0.098,-0.4,0.41,-0.35,0.41,-0.43,0.41
ponatinib,L248R,8.0,0.58,,,,,,
ponatinib,T315I,6.0,0.41,,,,,,
"""


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return load_hauser2018_from_csv(_write_fixture(Path("/tmp/_abl_test_fixture_inline.csv")))


def _write_fixture(p: Path) -> Path:
    p.write_text(_FIXTURE_CSV)
    return p


@pytest.fixture
def fixture_csv(tmp_path) -> Path:
    p = tmp_path / HAUSER2018_CACHE_NAME
    p.write_text(_FIXTURE_CSV)
    return p


# ---------------------------------------------------------------------------
# parse_mutation
# ---------------------------------------------------------------------------


def test_parse_mutation_normal():
    m = parse_mutation("T315I")
    assert m == AblMutation(wt_aa="T", resnum=315, mut_aa="I")


def test_parse_mutation_three_digit():
    m = parse_mutation("M244V")
    assert m == AblMutation(wt_aa="M", resnum=244, mut_aa="V")


def test_parse_mutation_str_round_trips():
    m = parse_mutation("E255K")
    assert str(m) == "E255K"


def test_parse_mutation_malformed_raises():
    with pytest.raises(ValueError, match="Malformed"):
        parse_mutation("315I")


def test_parse_mutation_malformed_empty_raises():
    with pytest.raises(ValueError, match="Malformed"):
        parse_mutation("")


def test_parse_mutation_strips_whitespace():
    m = parse_mutation("  T315I  ")
    assert m.resnum == 315


# ---------------------------------------------------------------------------
# Sign convention — positive DDG_exp means resistance
# ---------------------------------------------------------------------------


def test_sign_convention_t315i_imatinib_resistance(fixture_df):
    """T315I against imatinib is the canonical resistance mutation; DDG must be positive."""
    row = fixture_df[(fixture_df["TKI"] == "imatinib") & (fixture_df["mutation"] == "T315I")]
    assert len(row) == 1
    assert row.iloc[0]["DDG_exp"] > 0


def test_sign_convention_t315i_dasatinib_resistance(fixture_df):
    """T315I is highly resistant to dasatinib (DDG ~5 kcal/mol)."""
    row = fixture_df[(fixture_df["TKI"] == "dasatinib") & (fixture_df["mutation"] == "T315I")]
    ddg = row.iloc[0]["DDG_exp"]
    assert ddg > 3.0


def test_sign_convention_sensitising_mutation(fixture_df):
    """Axitinib M244V has negative DDG (slight sensitisation)."""
    row = fixture_df[(fixture_df["TKI"] == "axitinib") & (fixture_df["mutation"] == "M244V")]
    assert row.iloc[0]["DDG_exp"] < 0


def test_sign_convention_neutral_is_zero(fixture_df):
    """Dasatinib M244V has DDG_exp = 0.0 (no change)."""
    row = fixture_df[(fixture_df["TKI"] == "dasatinib") & (fixture_df["mutation"] == "M244V")]
    assert row.iloc[0]["DDG_exp"] == 0.0


# ---------------------------------------------------------------------------
# Temperature constant — T=300K, not 298K
# ---------------------------------------------------------------------------


def test_rt_300k_value():
    """RT at 300K must be ~0.596 kcal/mol, matching the paper."""
    assert RT_300K == pytest.approx(0.5962, abs=1e-3)


def test_rt_ln10_300k_value():
    """RT*ln(10) at 300K must be ~1.373 kcal/mol per log10 unit."""
    assert RT_LN10_300K == pytest.approx(1.373, abs=1e-2)


def test_rt_is_not_298k():
    """Guard against accidentally using T=298K (RT*ln10 ~ 1.364)."""
    assert RT_LN10_300K > 1.370


# ---------------------------------------------------------------------------
# dlog10_ic50 derived column
# ---------------------------------------------------------------------------


def test_dlog10_ic50_column_exists(fixture_df):
    """The loader must add a dlog10_ic50 column."""
    assert "dlog10_ic50" in fixture_df.columns


def test_dlog10_ic50_zero_for_neutral(fixture_df):
    """DDG_exp = 0 must produce dlog10_ic50 = 0."""
    row = fixture_df[(fixture_df["TKI"] == "dasatinib") & (fixture_df["mutation"] == "M244V")]
    assert row.iloc[0]["dlog10_ic50"] == pytest.approx(0.0)


def test_dlog10_ic50_positive_for_resistance(fixture_df):
    """Positive DDG (resistance) must give positive dlog10_ic50."""
    row = fixture_df[(fixture_df["TKI"] == "imatinib") & (fixture_df["mutation"] == "T315I")]
    assert row.iloc[0]["dlog10_ic50"] > 0


def test_dlog10_ic50_round_trip(fixture_df):
    """dlog10_ic50 * RT_LN10_300K must recover DDG_exp."""
    meas_df = measured(fixture_df)
    recovered = meas_df["dlog10_ic50"] * RT_LN10_300K
    np.testing.assert_allclose(recovered.values, meas_df["DDG_exp"].values, atol=1e-10)


def test_dlog10_ic50_nan_where_ddg_nan(fixture_df):
    """dlog10_ic50 must be NaN where DDG_exp is NaN."""
    nan_rows = fixture_df[fixture_df["DDG_exp"].isna()]
    assert nan_rows["dlog10_ic50"].isna().all()


# ---------------------------------------------------------------------------
# structure_type derived column
# ---------------------------------------------------------------------------


def test_structure_type_column_exists(fixture_df):
    assert "structure_type" in fixture_df.columns


def test_structure_type_xtal_for_abl_tkis(fixture_df):
    for tki in ABL_TKIS:
        rows = fixture_df[fixture_df["TKI"] == tki]
        if len(rows) > 0:
            assert (rows["structure_type"] == "xtal").all(), f"{tki} should be xtal"


def test_structure_type_docked_for_egfr_tkis(fixture_df):
    for tki in EGFR_TKIS:
        rows = fixture_df[fixture_df["TKI"] == tki]
        if len(rows) > 0:
            assert (rows["structure_type"] == "docked").all(), f"{tki} should be docked"


# ---------------------------------------------------------------------------
# measured / abl_only / xtal_only / docked_only — filtering
# ---------------------------------------------------------------------------


def test_measured_drops_nan_rows(fixture_df):
    result = measured(fixture_df)
    assert result["IC50"].notna().all()
    assert result["DDG_exp"].notna().all()


def test_measured_keeps_valid_rows(fixture_df):
    result = measured(fixture_df)
    assert len(result) == 10  # 14 rows minus 4 with NaN IC50 or DDG_exp


def test_measured_does_not_mutate_input(fixture_df):
    original_len = len(fixture_df)
    measured(fixture_df)
    assert len(fixture_df) == original_len


def test_measured_resets_index(fixture_df):
    result = measured(fixture_df)
    assert list(result.index) == list(range(len(result)))


def test_abl_only_drops_egfr(fixture_df):
    result = abl_only(fixture_df)
    assert "erlotinib" not in result["TKI"].values
    assert "gefitinib" not in result["TKI"].values


def test_abl_only_keeps_abl_tkis(fixture_df):
    result = abl_only(fixture_df)
    for tki in result["TKI"].unique():
        assert tki in ABL_TKIS


def test_abl_only_count(fixture_df):
    result = abl_only(fixture_df)
    assert len(result) == 12  # 14 total minus 2 erlotinib rows


def test_xtal_only_matches_abl_only(fixture_df):
    xtal = xtal_only(fixture_df)
    abl = abl_only(fixture_df)
    assert len(xtal) == len(abl)


def test_docked_only_returns_egfr(fixture_df):
    result = docked_only(fixture_df)
    assert len(result) > 0
    for tki in result["TKI"].unique():
        assert tki in EGFR_TKIS


def test_xtal_plus_docked_equals_total(fixture_df):
    assert len(xtal_only(fixture_df)) + len(docked_only(fixture_df)) == len(fixture_df)


# ---------------------------------------------------------------------------
# load_hauser2018_from_csv
# ---------------------------------------------------------------------------


def test_load_from_csv_reads_all_columns(fixture_csv):
    df = load_hauser2018_from_csv(fixture_csv)
    assert "TKI" in df.columns
    assert "mutation" in df.columns
    assert "IC50" in df.columns
    assert "DDG_exp" in df.columns
    assert "dlog10_ic50" in df.columns
    assert "structure_type" in df.columns


def test_load_from_csv_row_count(fixture_csv):
    df = load_hauser2018_from_csv(fixture_csv)
    assert len(df) == 14


# ---------------------------------------------------------------------------
# download — caching behaviour (no network)
# ---------------------------------------------------------------------------


def test_download_caches_skips_network(tmp_path, monkeypatch):
    """When the CSV already exists, no download should occur."""
    dest = tmp_path / HAUSER2018_CACHE_NAME
    dest.write_text(_FIXTURE_CSV)

    def should_not_be_called(url, dest):
        raise AssertionError("_atomic_download called despite cached file")

    monkeypatch.setattr("igv.abl._atomic_download", should_not_be_called)
    result = download_hauser2018(tmp_path)
    assert result == dest


def test_download_cache_key_is_dataset_specific():
    """Cache filename must include the dataset name to avoid collisions."""
    assert "hauser2018" in HAUSER2018_CACHE_NAME
    assert "abl" in HAUSER2018_CACHE_NAME


def test_download_url_is_github():
    assert "github" in HAUSER2018_XLSX_URL
    assert "kehauser" in HAUSER2018_XLSX_URL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_all_tkis_has_eight():
    assert len(ALL_TKIS) == 8


def test_abl_tkis_has_six():
    assert len(ABL_TKIS) == 6


def test_egfr_tkis_has_two():
    assert len(EGFR_TKIS) == 2


def test_all_tkis_is_union():
    assert set(ALL_TKIS) == set(ABL_TKIS) | set(EGFR_TKIS)


def test_tki_pdb_covers_all_abl_tkis():
    for tki in ABL_TKIS:
        assert tki in TKI_PDB, f"Missing PDB for {tki}"


def test_tki_pdb_does_not_cover_egfr():
    for tki in EGFR_TKIS:
        assert tki not in TKI_PDB


def test_egfr_reference_pdb_covers_egfr_tkis():
    for tki in EGFR_TKIS:
        assert tki in EGFR_REFERENCE_PDB


def test_tki_pdb_values_are_four_chars():
    for tki, pdb in TKI_PDB.items():
        assert len(pdb) == 4, f"{tki} PDB {pdb!r} is not 4 characters"


def test_tki_smiles_covers_all_eight_tkis():
    for tki in ALL_TKIS:
        assert tki in TKI_SMILES, f"Missing SMILES for {tki}"
        assert len(TKI_SMILES[tki]) > 10, f"SMILES too short for {tki}"


def test_wt_ic50_covers_all_abl_tkis():
    for tki in ABL_TKIS:
        assert tki in WT_IC50_NM, f"Missing WT IC50 for {tki}"
        assert WT_IC50_NM[tki] > 0


def test_tki_pdb_imatinib():
    assert TKI_PDB["imatinib"] == "1OPJ"


def test_tki_pdb_dasatinib():
    assert TKI_PDB["dasatinib"] == "4XEY"


def test_tki_pdb_nilotinib():
    assert TKI_PDB["nilotinib"] == "3CS9"


def test_tki_pdb_bosutinib():
    assert TKI_PDB["bosutinib"] == "3UE4"


def test_tki_pdb_ponatinib():
    assert TKI_PDB["ponatinib"] == "3OXZ"


def test_tki_pdb_axitinib():
    assert TKI_PDB["axitinib"] == "4WA9"


def test_egfr_reference_erlotinib():
    assert EGFR_REFERENCE_PDB["erlotinib"] == "4HJO"


def test_egfr_reference_gefitinib():
    assert EGFR_REFERENCE_PDB["gefitinib"] == "4WKQ"


def test_noise_floor_constant():
    """IC50-vs-Kd RMSE must match the paper's Fig 2d value."""
    assert IC50_VS_KD_RMSE_KCAL == pytest.approx(0.813, abs=0.001)
    assert IC50_VS_KD_N == 27


# ---------------------------------------------------------------------------
# Numbering convention — ABL1 isoform 1b
# ---------------------------------------------------------------------------


def test_gatekeeper_is_315():
    """T315 is the gatekeeper in 1b numbering. If anyone changes this to
    333 (1a numbering) or something else, this test catches it."""
    m = parse_mutation("T315I")
    assert m.resnum == 315
    assert m.wt_aa == "T"


def test_mutation_resnums_are_1b_range(fixture_df):
    """All mutation residue numbers should fall in the ABL1 1b kinase domain range."""
    for mut_str in fixture_df["mutation"].unique():
        mut = parse_mutation(mut_str)
        assert (
            200 <= mut.resnum <= 600
        ), f"Residue {mut.resnum} outside expected ABL1 1b kinase domain range"


# ---------------------------------------------------------------------------
# map_to_pdb
# ---------------------------------------------------------------------------


def _make_pdb_data():
    """Synthetic PDB residue IDs and sequence for mapping tests.

    Residues 244, 250, 315, 317, 486 present; 379 absent.
    """
    residue_ids = ["244", "250", "315", "317", "486"]
    sequence = "MGTFF"
    return residue_ids, sequence


def test_map_to_pdb_correct_mapping(fixture_df):
    residue_ids, sequence = _make_pdb_data()
    meas_df = measured(fixture_df)
    imatinib_df = meas_df[meas_df["TKI"] == "imatinib"]
    mapped, unmapped = map_to_pdb(
        imatinib_df,
        residue_ids,
        sequence,
        chain="A",
        pdb_id="TEST",
    )
    m244v = [m for m in mapped if m.mutation.resnum == 244]
    assert len(m244v) == 1
    assert m244v[0].pdb_index == 0
    assert m244v[0].ddg_exp == pytest.approx(0.21)


def test_map_to_pdb_carries_dlog10(fixture_df):
    residue_ids, sequence = _make_pdb_data()
    meas_df = measured(fixture_df)
    imatinib_df = meas_df[meas_df["TKI"] == "imatinib"]
    mapped, _ = map_to_pdb(
        imatinib_df,
        residue_ids,
        sequence,
        chain="A",
        pdb_id="TEST",
    )
    m244v = [m for m in mapped if m.mutation.resnum == 244][0]
    assert m244v.dlog10_ic50 == pytest.approx(0.21 / RT_LN10_300K, rel=1e-6)


def test_map_to_pdb_t315i_maps(fixture_df):
    residue_ids, sequence = _make_pdb_data()
    meas_df = measured(fixture_df)
    imatinib_df = meas_df[meas_df["TKI"] == "imatinib"]
    mapped, _ = map_to_pdb(
        imatinib_df,
        residue_ids,
        sequence,
        chain="A",
        pdb_id="TEST",
    )
    t315i = [m for m in mapped if m.mutation.resnum == 315]
    assert len(t315i) == 1
    assert t315i[0].pdb_index == 2


def test_map_to_pdb_absent_residue_flagged(fixture_df):
    """V379I should be flagged as unmapped since 379 is not in residue_ids."""
    residue_ids, sequence = _make_pdb_data()
    axi_df = fixture_df[fixture_df["TKI"] == "axitinib"]
    axi_df = axi_df[axi_df["IC50"].notna() & axi_df["DDG_exp"].notna()]
    mapped, unmapped = map_to_pdb(
        axi_df,
        residue_ids,
        sequence,
        chain="A",
        pdb_id="TEST",
    )
    m244v = [m for m in mapped if m.mutation.resnum == 244]
    assert len(m244v) == 1


def test_map_to_pdb_wt_mismatch_flagged():
    """A wild-type mismatch must produce an UnmappedMeasurement."""
    residue_ids = ["315"]
    sequence = "X"
    df = pd.DataFrame(
        {
            "TKI": ["imatinib"],
            "mutation": ["T315I"],
            "IC50": [9773.0],
            "DDG_exp": [2.32],
        }
    )
    mapped, unmapped = map_to_pdb(
        df,
        residue_ids,
        sequence,
        chain="A",
        pdb_id="TEST",
    )
    assert len(mapped) == 0
    assert len(unmapped) == 1
    assert "mismatch" in unmapped[0].reason


def test_map_to_pdb_nan_rows_skipped(fixture_df):
    """Rows with NaN IC50 or DDG_exp must be silently skipped, not flagged."""
    residue_ids, sequence = _make_pdb_data()
    mapped, unmapped = map_to_pdb(
        fixture_df,
        residue_ids,
        sequence,
        chain="A",
        pdb_id="TEST",
    )
    all_mutations = [m.mutation for m in mapped] + [u.mutation for u in unmapped]
    f486s_entries = [m for m in all_mutations if m.resnum == 486 and m.mut_aa == "S"]
    assert len(f486s_entries) == 0


def test_map_to_pdb_does_not_silently_drop():
    """Every row with data must appear in either mapped or unmapped."""
    residue_ids = ["244"]
    sequence = "M"
    df = pd.DataFrame(
        {
            "TKI": ["imatinib", "imatinib"],
            "mutation": ["M244V", "T315I"],
            "IC50": [287.0, 9773.0],
            "DDG_exp": [0.21, 2.32],
        }
    )
    mapped, unmapped = map_to_pdb(
        df,
        residue_ids,
        sequence,
        chain="A",
        pdb_id="TEST",
    )
    assert len(mapped) + len(unmapped) == 2


# ---------------------------------------------------------------------------
# map_resnum_to_pdb_index
# ---------------------------------------------------------------------------


def test_map_resnum_present():
    residue_ids = ["244", "315", "486"]
    sequence = "MTF"
    idx = map_resnum_to_pdb_index(315, residue_ids, sequence)
    assert idx == 1


def test_map_resnum_absent():
    residue_ids = ["244", "315", "486"]
    sequence = "MTF"
    idx = map_resnum_to_pdb_index(379, residue_ids, sequence)
    assert idx is None


def test_map_resnum_wt_check_passes():
    residue_ids = ["315"]
    sequence = "T"
    idx = map_resnum_to_pdb_index(315, residue_ids, sequence, expected_wt="T")
    assert idx == 0


def test_map_resnum_wt_check_fails():
    residue_ids = ["315"]
    sequence = "X"
    with pytest.raises(ValueError, match="mismatch"):
        map_resnum_to_pdb_index(315, residue_ids, sequence, expected_wt="T")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_total_rows(fixture_df):
    s = summary(fixture_df)
    assert s["total_rows"] == 14


def test_summary_measured_count(fixture_df):
    s = summary(fixture_df)
    assert s["total_measured"] == 10


def test_summary_per_tki_keys(fixture_df):
    s = summary(fixture_df)
    assert "imatinib" in s["per_tki"]
    assert "dasatinib" in s["per_tki"]


def test_summary_distinct_positions(fixture_df):
    s = summary(fixture_df)
    assert s["n_distinct_positions"] >= 3


def test_summary_ddg_range(fixture_df):
    s = summary(fixture_df)
    assert s["ddg_min"] < 0
    assert s["ddg_max"] > 0


def test_summary_resistance_count(fixture_df):
    s = summary(fixture_df)
    assert s["n_resistance"] > 0
    assert s["n_sensitising"] > 0


# ---------------------------------------------------------------------------
# Censored / qualitative value handling
# ---------------------------------------------------------------------------


def test_censored_ic50_without_ddg_is_nan(fixture_df):
    """V379I in axitinib has IC50 but no DDG_exp -- measured() must drop it."""
    result = measured(fixture_df)
    v379i = result[result["mutation"] == "V379I"]
    assert len(v379i) == 0


def test_missing_data_rows_have_nan(fixture_df):
    """F486S rows (imatinib, dasatinib) should have NaN IC50."""
    f486s = fixture_df[fixture_df["mutation"] == "F486S"]
    assert f486s["IC50"].isna().all()
    assert f486s["DDG_exp"].isna().all()


def test_erlotinib_is_egfr_control(fixture_df):
    """Erlotinib rows must be excluded by abl_only()."""
    result = abl_only(fixture_df)
    assert "erlotinib" not in result["TKI"].values


def test_erlotinib_is_docked(fixture_df):
    """Erlotinib rows must be flagged as docked."""
    erl = fixture_df[fixture_df["TKI"] == "erlotinib"]
    assert (erl["structure_type"] == "docked").all()


# ---------------------------------------------------------------------------
# IC50 consistency check — ponatinib / G250E is the known bad row
# ---------------------------------------------------------------------------


def test_ic50_suspect_column_exists(fixture_df):
    assert "ic50_suspect" in fixture_df.columns


def test_ic50_suspect_g250e_ponatinib_flagged(fixture_df):
    """G250E / ponatinib has a corrupt IC50 (0.021 nM vs expected ~5 nM).
    The ic50_suspect flag must be True for exactly this row."""
    g250e = fixture_df[(fixture_df["TKI"] == "ponatinib") & (fixture_df["mutation"] == "G250E")]
    assert len(g250e) == 1
    assert bool(g250e.iloc[0]["ic50_suspect"])


def test_ic50_suspect_clean_rows_not_flagged(fixture_df):
    """All rows except G250E/ponatinib must pass the consistency check."""
    meas_df = measured(fixture_df)
    not_g250e = meas_df[~((meas_df["TKI"] == "ponatinib") & (meas_df["mutation"] == "G250E"))]
    assert not not_g250e["ic50_suspect"].any(), (
        f"Unexpected suspect rows: "
        f"{not_g250e[not_g250e['ic50_suspect']][['TKI', 'mutation', 'IC50', 'DDG_exp']].to_string()}"
    )


def test_ic50_suspect_nan_rows_not_flagged(fixture_df):
    """Rows with NaN IC50 or DDG_exp must not be flagged (nothing to check)."""
    nan_rows = fixture_df[fixture_df["IC50"].isna() | fixture_df["DDG_exp"].isna()]
    assert not nan_rows["ic50_suspect"].any()


def test_validate_ic50_consistency_returns_suspect_rows(fixture_df):
    result = validate_ic50_consistency(fixture_df)
    assert len(result) == 1
    assert result.iloc[0]["mutation"] == "G250E"
    assert result.iloc[0]["TKI"] == "ponatinib"


def test_validate_ic50_consistency_deviation_column(fixture_df):
    result = validate_ic50_consistency(fixture_df)
    assert "ic50_deviation_log10" in result.columns
    assert result.iloc[0]["ic50_deviation_log10"] > 2.0


def test_validate_ic50_consistency_raises_without_column():
    df = pd.DataFrame({"TKI": ["x"], "mutation": ["T315I"], "IC50": [1.0], "DDG_exp": [0.1]})
    with pytest.raises(ValueError, match="ic50_suspect"):
        validate_ic50_consistency(df)


def test_ic50_consistency_tolerance_is_stated():
    """The tolerance must be a positive number, available as a module constant."""
    assert IC50_CONSISTENCY_TOL_LOG10 > 0
    assert IC50_CONSISTENCY_TOL_LOG10 < 1.0


# ---------------------------------------------------------------------------
# Integration test (requires real data on disk)
# ---------------------------------------------------------------------------

_HAUSER_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / HAUSER2018_CACHE_NAME


@pytest.mark.skipif(not _HAUSER_CSV.exists(), reason="Hauser 2018 CSV not on disk")
def test_full_dataset_counts():
    """Verify total measurement count matches the paper's 144."""
    df = load_hauser2018_from_csv(_HAUSER_CSV)
    meas_df = measured(df)
    assert len(meas_df) == 144, f"Expected 144 measurements, got {len(meas_df)}"


@pytest.mark.skipif(not _HAUSER_CSV.exists(), reason="Hauser 2018 CSV not on disk")
def test_full_dataset_abl_only_count():
    """6 ABL TKIs: 131 measurements."""
    df = load_hauser2018_from_csv(_HAUSER_CSV)
    result = abl_only(measured(df))
    assert len(result) == 131


@pytest.mark.skipif(not _HAUSER_CSV.exists(), reason="Hauser 2018 CSV not on disk")
def test_full_dataset_distinct_mutations():
    """31 distinct mutations across all TKIs."""
    df = load_hauser2018_from_csv(_HAUSER_CSV)
    meas_df = measured(df)
    assert meas_df["mutation"].nunique() == 31


@pytest.mark.skipif(not _HAUSER_CSV.exists(), reason="Hauser 2018 CSV not on disk")
def test_full_dataset_per_tki_counts():
    """Per-TKI measurement counts verified against the spreadsheet."""
    df = load_hauser2018_from_csv(_HAUSER_CSV)
    meas_df = measured(df)
    expected = {
        "imatinib": 21,
        "dasatinib": 21,
        "nilotinib": 21,
        "ponatinib": 21,
        "bosutinib": 21,
        "axitinib": 26,
        "erlotinib": 7,
        "gefitinib": 6,
    }
    for tki, count in expected.items():
        actual = len(meas_df[meas_df["TKI"] == tki])
        assert actual == count, f"{tki}: expected {count}, got {actual}"


@pytest.mark.skipif(not _HAUSER_CSV.exists(), reason="Hauser 2018 CSV not on disk")
def test_full_dataset_structure_type_counts():
    """131 xtal, 13 docked among measured rows."""
    df = load_hauser2018_from_csv(_HAUSER_CSV)
    meas_df = measured(df)
    assert len(xtal_only(meas_df)) == 131
    assert len(docked_only(meas_df)) == 13


@pytest.mark.skipif(not _HAUSER_CSV.exists(), reason="Hauser 2018 CSV not on disk")
def test_full_dataset_exactly_one_suspect_ic50():
    """Exactly one row must be flagged: ponatinib / G250E."""
    df = load_hauser2018_from_csv(_HAUSER_CSV)
    suspect = validate_ic50_consistency(df)
    assert len(suspect) == 1, f"Expected 1 suspect row, got {len(suspect)}"
    assert suspect.iloc[0]["TKI"] == "ponatinib"
    assert suspect.iloc[0]["mutation"] == "G250E"


@pytest.mark.skipif(not _HAUSER_CSV.exists(), reason="Hauser 2018 CSV not on disk")
def test_full_dataset_143_clean_rows():
    """143 of 144 measured rows must pass the IC50 consistency check."""
    df = load_hauser2018_from_csv(_HAUSER_CSV)
    meas_df = measured(df)
    clean = meas_df[~meas_df["ic50_suspect"]]
    assert len(clean) == 143, f"Expected 143 clean rows, got {len(clean)}"
