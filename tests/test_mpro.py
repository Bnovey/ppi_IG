"""Tests for igv.mpro -- CPU-only, no network.

Uses fixture snippets of real pdftotext -layout output to test
parsing of Table S1 from the Hu et al. 2023 supplement.
"""

from __future__ import annotations

import textwrap

import pytest

from igv.mpro import (
    ENZYMATICALLY_INACTIVE,
    MPRO_PDF_SHA256,
    WT_NIRMATRELVIR_IC50,
    WT_NIRMATRELVIR_KI,
    _classify_compound,
    _extract_drug_values,
    _extract_kinetics,
    parse_table_s1,
)


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------


class TestClassifyCompound:
    def test_gc376(self):
        assert _classify_compound(42) == "gc376"
        assert _classify_compound(53) == "gc376"

    def test_pf00835231(self):
        assert _classify_compound(65) == "pf00835231"
        assert _classify_compound(80) == "pf00835231"

    def test_nirmatrelvir(self):
        assert _classify_compound(90) == "nirmatrelvir"
        assert _classify_compound(105) == "nirmatrelvir"


# ---------------------------------------------------------------------------
# Fixture snippets from the real extracted text
# ---------------------------------------------------------------------------

# A full row with Ki for all three compounds (T135I, from page S12)
SNIPPET_T135I = textwrap.dedent("""\
                         Resistant mutants identified from the GISAID SARS-CoV-2 sequence analysis
                           Km = 27.08 ± 2.76 µM
                             Vmax = 29.28 ± 0.97
                                                     IC50 = 42.12 ± 3.41 nM   IC50 = 15.63 ± 1.05 nM   IC50 = 21.86 ± 0.85 nM
 T135I        1,342                   nM/s
                                                      Ki = 15.38 ± 1.13 nM      Ki = 2.47 ± 0.19 nM      Ki = 2.26 ± 0.12 nM
                               kcat = 0.2928 S-1
                           kcat/Km = 10,812 S-1M-1
aThe occurrence""")

# A row with IC50 but no Ki (M49I, from page S12)
SNIPPET_M49I = textwrap.dedent("""\
                         Resistant mutants identified from the GISAID SARS-CoV-2 sequence analysis
                           Km = 21.43 ± 3.29 µM
                             Vmax = 19.90 ± 0.94
 M49I         2,080                   nM/s           IC50 = 41.55 ± 0.75 nM   IC50 = 17.84 ± 0.64 nM   IC50 = 26.54 ± 1.05 nM
                               kcat = 0.3980 S-1
                           kcat/Km = 18,572 S-1M-1
aThe occurrence""")

# An enzymatically inactive row (H41M)
SNIPPET_H41M = textwrap.dedent("""\
                         Resistant mutants identified from the GISAID SARS-CoV-2 sequence analysis
H41M           84          Enzymatically inactive
aThe occurrence""")

# A deletion row (M49 deletion) with wrapped name
SNIPPET_M49_DEL = textwrap.dedent("""\
                         Resistant mutants identified from the GISAID SARS-CoV-2 sequence analysis
                           Km = 30.87 ± 3.16 µM
                            Vmax = 17.45 ± 0.60
 M49                                                                                                   IC50 = 32.99 ± 1.94 nM
               29                    nM/s
deletion                                                                                                 Ki = 8.02 ± 0.82 nM
                              kcat = 0.1745 S-1
                           kcat/Km = 5,653 S-1M-1
aThe occurrence""")

# WT row
SNIPPET_WT = textwrap.dedent("""\
                         Resistant mutants identified from the GISAID SARS-CoV-2 sequence analysis
                           Km = 35.36 ± 2.41 µM
                             Vmax = 38.97 ± 0.91
                                                     IC50 = 40.25 ± 1.61 nM   IC50 = 15.09 ± 0.80 nM   IC50 = 26.03 ± 1.65 nM
  WT                                 nM/s
                                                      Ki = 15.50 ± 1.28 nM     Ki = 0.84 ± 0.061 nM      Ki =1.88 ± 0.33 nM
                                kcat = 0.39 S-1
                           kcat/Km = 11,000 S-1M-1
aThe occurrence""")


# ---------------------------------------------------------------------------
# Full row with Ki (T135I)
# ---------------------------------------------------------------------------


class TestFullRowWithKi:
    @pytest.fixture()
    def entry(self):
        entries = parse_table_s1(SNIPPET_T135I)
        return next(e for e in entries if e["mutation_id"] == "T135I")

    def test_mutation_id(self, entry):
        assert entry["mutation_id"] == "T135I"

    def test_position(self, entry):
        assert entry["position"] == 135

    def test_wt_mut_aa(self, entry):
        assert entry["wt_aa"] == "T"
        assert entry["mut_aa"] == "I"

    def test_variant_type(self, entry):
        assert entry["variant_type"] == "substitution"

    def test_status(self, entry):
        assert entry["status"] == "active"

    def test_occurrence(self, entry):
        assert entry["occurrence"] == 1342

    def test_kinetics(self, entry):
        assert entry["km"] == pytest.approx(27.08)
        assert entry["km_err"] == pytest.approx(2.76)
        assert entry["kcat"] == pytest.approx(0.2928)
        assert entry["kcat_over_km"] == pytest.approx(10812)

    def test_nirmatrelvir_ki(self, entry):
        assert entry["nirmatrelvir_ki"] == pytest.approx(2.26)
        assert entry["nirmatrelvir_ki_err"] == pytest.approx(0.12)

    def test_nirmatrelvir_ic50(self, entry):
        assert entry["nirmatrelvir_ic50"] == pytest.approx(21.86)

    def test_gc376_ki(self, entry):
        assert entry["gc376_ki"] == pytest.approx(15.38)

    def test_pf_ki(self, entry):
        assert entry["pf00835231_ki"] == pytest.approx(2.47)


# ---------------------------------------------------------------------------
# Row with IC50 only, no Ki (M49I)
# ---------------------------------------------------------------------------


class TestRowIC50Only:
    @pytest.fixture()
    def entry(self):
        entries = parse_table_s1(SNIPPET_M49I)
        return next(e for e in entries if e["mutation_id"] == "M49I")

    def test_nirmatrelvir_ic50(self, entry):
        assert entry["nirmatrelvir_ic50"] == pytest.approx(26.54)

    def test_nirmatrelvir_ki_absent(self, entry):
        assert entry.get("nirmatrelvir_ki") is None

    def test_gc376_ic50(self, entry):
        assert entry["gc376_ic50"] == pytest.approx(41.55)

    def test_pf_ic50(self, entry):
        assert entry["pf00835231_ic50"] == pytest.approx(17.84)

    def test_occurrence(self, entry):
        assert entry["occurrence"] == 2080


# ---------------------------------------------------------------------------
# Enzymatically inactive row (H41M)
# ---------------------------------------------------------------------------


class TestInactiveRow:
    @pytest.fixture()
    def entry(self):
        entries = parse_table_s1(SNIPPET_H41M)
        return next(e for e in entries if e["mutation_id"] == "H41M")

    def test_status(self, entry):
        assert entry["status"] == "inactive"

    def test_variant_type(self, entry):
        assert entry["variant_type"] == "substitution"

    def test_no_ki(self, entry):
        assert entry.get("nirmatrelvir_ki") is None

    def test_no_kinetics(self, entry):
        assert entry.get("km") is None

    def test_occurrence(self, entry):
        assert entry["occurrence"] == 84


# ---------------------------------------------------------------------------
# Deletion row with wrapped name (M49 deletion)
# ---------------------------------------------------------------------------


class TestDeletionRow:
    @pytest.fixture()
    def entry(self):
        entries = parse_table_s1(SNIPPET_M49_DEL)
        return next(
            e for e in entries if e["mutation_id"] == "M49 deletion"
        )

    def test_variant_type(self, entry):
        assert entry["variant_type"] == "deletion"

    def test_position(self, entry):
        assert entry["position"] == 49

    def test_wt_aa(self, entry):
        assert entry["wt_aa"] == "M"

    def test_mut_aa_none(self, entry):
        assert entry["mut_aa"] is None

    def test_occurrence(self, entry):
        assert entry["occurrence"] == 29

    def test_nirmatrelvir_ki(self, entry):
        assert entry["nirmatrelvir_ki"] == pytest.approx(8.02)

    def test_nirmatrelvir_ic50(self, entry):
        assert entry["nirmatrelvir_ic50"] == pytest.approx(32.99)


# ---------------------------------------------------------------------------
# WT anchor guard
# ---------------------------------------------------------------------------


class TestWTAnchor:
    @pytest.fixture()
    def wt(self):
        entries = parse_table_s1(SNIPPET_WT)
        return next(e for e in entries if e["mutation_id"] == "WT")

    def test_wt_nirmatrelvir_ki(self, wt):
        assert wt["nirmatrelvir_ki"] == pytest.approx(WT_NIRMATRELVIR_KI, abs=0.01)

    def test_wt_nirmatrelvir_ic50(self, wt):
        assert wt["nirmatrelvir_ic50"] == pytest.approx(WT_NIRMATRELVIR_IC50, abs=0.01)

    def test_wt_gc376_ki(self, wt):
        assert wt["gc376_ki"] == pytest.approx(15.50)

    def test_wt_km(self, wt):
        assert wt["km"] == pytest.approx(35.36)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_dead_mutants_known(self):
        assert ENZYMATICALLY_INACTIVE == {"H41M", "H41T", "H41Y", "H163W"}

    def test_wt_ki_value(self):
        assert WT_NIRMATRELVIR_KI == pytest.approx(1.88)

    def test_pdf_sha256_is_set(self):
        assert len(MPRO_PDF_SHA256) == 64


# ---------------------------------------------------------------------------
# Drug-value extraction (unit tests)
# ---------------------------------------------------------------------------


class TestExtractDrugValues:
    def test_three_ic50_on_one_line(self):
        line = "                                                     IC50 = 40.25 ± 1.61 nM   IC50 = 15.09 ± 0.80 nM   IC50 = 26.03 ± 1.65 nM"
        vals = _extract_drug_values([line])
        assert vals["gc376_ic50"] == pytest.approx(40.25)
        assert vals["pf00835231_ic50"] == pytest.approx(15.09)
        assert vals["nirmatrelvir_ic50"] == pytest.approx(26.03)

    def test_censored_ic50(self):
        line = "                                                                                                            IC50 > 10 µM"
        vals = _extract_drug_values([line])
        assert vals["nirmatrelvir_ic50"] == pytest.approx(10000.0)
        assert vals["nirmatrelvir_ic50_censored"] is True

    def test_censored_ki(self):
        line = "                                                                                                             Ki > 10 µM"
        vals = _extract_drug_values([line])
        assert vals["nirmatrelvir_ki"] == pytest.approx(10000.0)
        assert vals["nirmatrelvir_ki_censored"] is True

    def test_single_nirmatrelvir_ki(self):
        line = "                                                                                                         Ki = 2.00 ± 0.08 nM"
        vals = _extract_drug_values([line])
        assert vals["nirmatrelvir_ki"] == pytest.approx(2.00)
        assert vals["nirmatrelvir_ki_err"] == pytest.approx(0.08)

    def test_no_space_after_equals(self):
        line = "                                                                                                          Ki =1.88 ± 0.33 nM"
        vals = _extract_drug_values([line])
        assert vals["nirmatrelvir_ki"] == pytest.approx(1.88)


# ---------------------------------------------------------------------------
# Kinetics extraction (unit tests)
# ---------------------------------------------------------------------------


class TestExtractKinetics:
    def test_full_kinetics(self):
        lines = [
            "Km = 35.36 ± 2.41 µM",
            "Vmax = 38.97 ± 0.91",
            "kcat = 0.39 S-1",
            "kcat/Km = 11,000 S-1M-1",
        ]
        k = _extract_kinetics(lines)
        assert k["km"] == pytest.approx(35.36)
        assert k["km_err"] == pytest.approx(2.41)
        assert k["vmax"] == pytest.approx(38.97)
        assert k["kcat"] == pytest.approx(0.39)
        assert k["kcat_over_km"] == pytest.approx(11000)

    def test_kcat_not_confused_with_kcat_km(self):
        lines = [
            "kcat = 0.0795 S-1",
            "kcat/Km = 2,588 S-1M-1",
        ]
        k = _extract_kinetics(lines)
        assert k["kcat"] == pytest.approx(0.0795)
        assert k["kcat_over_km"] == pytest.approx(2588)
