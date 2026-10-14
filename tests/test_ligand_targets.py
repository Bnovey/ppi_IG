"""Tests for igv.ligand_targets -- CPU-only, no network."""

from __future__ import annotations

import numpy as np
import pytest

from igv.ligand_targets import (
    TARGETS,
    LigandTarget,
    compute_ligand_distances,
    get_target,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestTargetRegistry:
    def test_has_six_targets(self):
        expected = {
            "ABL1-imatinib", "EGFR-erlotinib", "EGFR-gefitinib",
            "PimA-GDD", "PimA-GDP", "Mpro-nirmatrelvir",
        }
        assert set(TARGETS.keys()) == expected

    def test_all_have_required_fields(self):
        for name, t in TARGETS.items():
            assert isinstance(t, LigandTarget), name
            assert t.pdb_id, name
            assert t.chains, name
            assert t.ligand_ccd, name
            assert t.sign_convention == "positive=resistance", name
            assert t.ground_truth_path, name

    def test_get_target_known(self):
        t = get_target("ABL1-imatinib")
        assert t.pdb_id == "2HYY"

    def test_get_target_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown target"):
            get_target("nonexistent")

    def test_egfr_offset(self):
        t = get_target("EGFR-erlotinib")
        assert t.resnum_offset == -24

    def test_mpro_is_dimer(self):
        t = get_target("Mpro-nirmatrelvir")
        assert len(t.chains) == 2
        assert "A" in t.chains
        assert "B" in t.chains

    def test_mpro_ligand_is_4wi(self):
        t = get_target("Mpro-nirmatrelvir")
        assert t.ligand_ccd == "4WI"

    def test_pima_gdd_pdb(self):
        t = get_target("PimA-GDD")
        assert t.pdb_id == "3OKP"

    def test_pima_gdp_pdb(self):
        t = get_target("PimA-GDP")
        assert t.pdb_id == "3OKC"


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------


class TestComputeLigandDistances:
    @staticmethod
    def _atom(serial, name, resname, chain, resseq, x, y, z, elem, rec="ATOM  "):
        return (
            f"{rec}{serial:5d} {name:<4s} {resname:>3s} {chain:1s}{resseq:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
        )

    def test_basic_distance(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        a = self._atom
        pdb.write_text(
            a(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C")
            + a(2, "CB", "ALA", "A", 10, 1.0, 0.0, 0.0, "C")
            + a(3, "CA", "GLY", "A", 20, 10.0, 0.0, 0.0, "C")
            + a(4, "C1", "LIG", "A", 1, 3.0, 0.0, 0.0, "C", "HETATM")
            + a(5, "C2", "LIG", "A", 1, 4.0, 0.0, 0.0, "C", "HETATM")
        )
        dists = compute_ligand_distances(pdb, "A", "LIG", [10, 20])
        assert dists[10] == pytest.approx(2.0, abs=0.01)
        assert dists[20] == pytest.approx(6.0, abs=0.01)

    def test_missing_position_nan(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        a = self._atom
        pdb.write_text(
            a(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C")
            + a(2, "C1", "LIG", "A", 1, 3.0, 0.0, 0.0, "C", "HETATM")
        )
        dists = compute_ligand_distances(pdb, "A", "LIG", [10, 99])
        assert np.isnan(dists[99])

    def test_no_ligand_raises(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        a = self._atom
        pdb.write_text(a(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"))
        with pytest.raises(ValueError, match="No HETATM"):
            compute_ligand_distances(pdb, "A", "LIG", [10])
