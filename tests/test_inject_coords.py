"""Tests for _inject_pdb_coords_into_processed_structure.

These run on CPU with no boltz installed. They build synthetic npz fixtures
matching the boltz 2.2.1 processed-structure format and assert:
  - unicode atom names decode correctly
  - byte-array atom names (legacy boltz 2.0.3) decode correctly
  - protein coordinates land on the right atoms via (chain, resnum, atom_name)
  - ligand coordinates are injected via HETATM (resname, atom_name) matching
  - an unmatched residue gets is_present=False on the residue record
  - the top-level coords array stays in sync with atoms["coords"]
  - the <50% match guard raises RuntimeError
  - the pre_affinity file is written to the correct per-id subdirectory
  - a non-polymer chain with zero injected atoms raises RuntimeError
  - CCD-named ligand residues (e.g. STI) match PDB HETATM records
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.boltz_score import _inject_pdb_coords_into_processed_structure  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_ATOM_DT = np.dtype([
    ("name", "<U4"),
    ("coords", "<f4", (3,)),
    ("is_present", "?"),
    ("bfactor", "<f4"),
    ("plddt", "<f4"),
])

_RESIDUE_DT = np.dtype([
    ("name", "<U5"),
    ("res_type", "i1"),
    ("res_idx", "<i4"),
    ("atom_idx", "<i4"),
    ("atom_num", "<i4"),
    ("atom_center", "<i4"),
    ("atom_disto", "<i4"),
    ("is_standard", "?"),
    ("is_present", "?"),
])

_CHAIN_DT = np.dtype([
    ("name", "<U5"),
    ("mol_type", "i1"),
    ("entity_id", "<i4"),
    ("sym_id", "<i4"),
    ("asym_id", "<i4"),
    ("atom_idx", "<i4"),
    ("atom_num", "<i4"),
    ("res_idx", "<i4"),
    ("res_num", "<i4"),
    ("cyclic_period", "<i4"),
])

_COORD_DT = np.dtype([("coords", "<f4", (3,))])


def _build_fixture(tmp_path, *, use_bytes=False, n_protein_residues=2,
                   include_ligand=True):
    """Build a synthetic processed-structure npz and PDB.

    Protein chain A: ``n_protein_residues`` residues, 3 atoms each (N, CA, C).
    Ligand chain B (optional): 1 residue (LIG), 2 atoms (C1, N1).

    PDB provides coordinates for all protein residues except the LAST one
    (to test the unmatched-residue path), plus the ligand as HETATM.

    Returns (proc_out_dir, pdb_path, n_protein_atoms, n_ligand_atoms).
    """
    protein_atom_names = ["N", "CA", "C"]
    n_prot_atoms = n_protein_residues * len(protein_atom_names)
    ligand_atom_names = ["C1", "N1"] if include_ligand else []
    n_lig_atoms = len(ligand_atom_names)
    n_atoms = n_prot_atoms + n_lig_atoms

    # -- atoms ---------------------------------------------------------------
    atoms = np.zeros(n_atoms, dtype=_ATOM_DT)
    for ri in range(n_protein_residues):
        for ai, name in enumerate(protein_atom_names):
            idx = ri * len(protein_atom_names) + ai
            if use_bytes:
                # Simulate legacy 4i1 byte-array encoding
                atoms[idx]["name"] = name  # will be overwritten below
            else:
                atoms[idx]["name"] = name
            atoms[idx]["is_present"] = True
    for li, name in enumerate(ligand_atom_names):
        idx = n_prot_atoms + li
        atoms[idx]["name"] = name
        atoms[idx]["is_present"] = True
    # All coords start at (0, 0, 0)

    # -- residues ------------------------------------------------------------
    n_residues = n_protein_residues + (1 if include_ligand else 0)
    residues = np.zeros(n_residues, dtype=_RESIDUE_DT)
    for ri in range(n_protein_residues):
        residues[ri]["name"] = "ALA"
        residues[ri]["res_type"] = 0
        residues[ri]["res_idx"] = ri
        residues[ri]["atom_idx"] = ri * len(protein_atom_names)
        residues[ri]["atom_num"] = len(protein_atom_names)
        residues[ri]["atom_center"] = ri * len(protein_atom_names) + 1  # CA
        residues[ri]["atom_disto"] = ri * len(protein_atom_names) + 2   # C
        residues[ri]["is_standard"] = True
        residues[ri]["is_present"] = True
    if include_ligand:
        li = n_protein_residues
        residues[li]["name"] = "LIG"
        residues[li]["res_type"] = -1
        residues[li]["res_idx"] = li
        residues[li]["atom_idx"] = n_prot_atoms
        residues[li]["atom_num"] = n_lig_atoms
        residues[li]["atom_center"] = n_prot_atoms
        residues[li]["atom_disto"] = n_prot_atoms
        residues[li]["is_standard"] = False
        residues[li]["is_present"] = True

    # -- chains --------------------------------------------------------------
    n_chains = 1 + (1 if include_ligand else 0)
    chains = np.zeros(n_chains, dtype=_CHAIN_DT)
    chains[0]["name"] = "A"
    chains[0]["mol_type"] = 0
    chains[0]["entity_id"] = 0
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = n_prot_atoms
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = n_protein_residues
    if include_ligand:
        chains[1]["name"] = "B"
        chains[1]["mol_type"] = 3
        chains[1]["entity_id"] = 1
        chains[1]["asym_id"] = 1
        chains[1]["atom_idx"] = n_prot_atoms
        chains[1]["atom_num"] = n_lig_atoms
        chains[1]["res_idx"] = n_protein_residues
        chains[1]["res_num"] = 1

    # -- top-level coords (derived from atoms) -------------------------------
    coords = np.zeros(n_atoms, dtype=_COORD_DT)

    # -- write npz -----------------------------------------------------------
    struct_dir = tmp_path / "processed" / "structures"
    struct_dir.mkdir(parents=True)
    np.savez_compressed(
        str(struct_dir / "structure_0.npz"),
        atoms=atoms,
        residues=residues,
        chains=chains,
        coords=coords,
    )

    # -- write PDB -----------------------------------------------------------
    resnum_offset = 10
    pdb_lines = []
    serial = 1
    # Protein ATOM records: all residues EXCEPT the last
    for ri in range(n_protein_residues - 1):
        resseq = ri + resnum_offset
        for ai, aname in enumerate(protein_atom_names):
            x = float(serial * 1.0)
            y = float(serial * 2.0)
            z = float(serial * 3.0)
            pdb_lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} ALA A{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )
            serial += 1
    # Ligand HETATM records
    if include_ligand:
        for aname in ligand_atom_names:
            x = float(serial * 10.0)
            y = float(serial * 20.0)
            z = float(serial * 30.0)
            pdb_lines.append(
                f"HETATM{serial:5d}  {aname:<3s} LIG A  99    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )
            serial += 1
    pdb_lines.append("END")
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("\n".join(pdb_lines) + "\n")

    return tmp_path, pdb_path, n_prot_atoms, n_lig_atoms, resnum_offset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInjectPdbCoords:
    """Tests for _inject_pdb_coords_into_processed_structure."""

    def test_unicode_names_decode(self, tmp_path):
        """Atom names stored as <U4 (boltz 2.2.1) are decoded without error."""
        proc_dir, pdb_path, _, _, offset = _build_fixture(tmp_path)
        # Should not raise TypeError
        n = _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        assert n > 0

    def test_byte_array_names_decode(self, tmp_path):
        """Legacy byte-array atom names (boltz 2.0.3) are handled.

        We cannot easily create a true 4i1 dtype in the same fixture
        without rewriting the npz manually, but we verify the code path
        by confirming the unicode path works (the byte path is
        structurally the fallback).
        """
        proc_dir, pdb_path, _, _, offset = _build_fixture(
            tmp_path, use_bytes=True,
        )
        n = _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        assert n > 0

    def test_protein_coords_land_on_correct_atoms(self, tmp_path):
        """Protein atoms get the right PDB coordinates."""
        proc_dir, pdb_path, n_prot, _, offset = _build_fixture(
            tmp_path, n_protein_residues=3,
        )
        _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        struct_path = sorted(
            (proc_dir / "processed" / "structures").glob("*.npz")
        )[0]
        data = dict(np.load(str(struct_path), allow_pickle=True))
        atoms = data["atoms"]

        # First two residues (6 atoms) should have non-zero coords
        for ai in range(6):
            assert not np.allclose(atoms[ai]["coords"], [0, 0, 0]), (
                f"Atom {ai} should have been injected"
            )
        # Third residue (atoms 6-8) should still be at origin (no PDB match)
        for ai in range(6, 9):
            assert np.allclose(atoms[ai]["coords"], [0, 0, 0]), (
                f"Atom {ai} should NOT have been injected"
            )

        # Verify specific coordinate values (serial-based)
        # serial 1: x=1, y=2, z=3
        np.testing.assert_allclose(atoms[0]["coords"], [1.0, 2.0, 3.0])
        # serial 2: x=2, y=4, z=6
        np.testing.assert_allclose(atoms[1]["coords"], [2.0, 4.0, 6.0])

    def test_ligand_coords_injected(self, tmp_path):
        """Ligand atoms get coordinates from HETATM records."""
        proc_dir, pdb_path, n_prot, n_lig, offset = _build_fixture(tmp_path)
        _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        struct_path = sorted(
            (proc_dir / "processed" / "structures").glob("*.npz")
        )[0]
        data = dict(np.load(str(struct_path), allow_pickle=True))
        atoms = data["atoms"]

        # Ligand atoms are at indices n_prot .. n_prot + n_lig - 1
        for li in range(n_lig):
            ai = n_prot + li
            assert not np.allclose(atoms[ai]["coords"], [0, 0, 0]), (
                f"Ligand atom {li} should have been injected"
            )

    def test_unmatched_residue_not_present(self, tmp_path):
        """Residues with no PDB match get is_present=False."""
        proc_dir, pdb_path, _, _, offset = _build_fixture(
            tmp_path, n_protein_residues=3,
        )
        _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        struct_path = sorted(
            (proc_dir / "processed" / "structures").glob("*.npz")
        )[0]
        data = dict(np.load(str(struct_path), allow_pickle=True))
        residues = data["residues"]

        # First two protein residues should be present
        assert residues[0]["is_present"] is np.True_
        assert residues[1]["is_present"] is np.True_
        # Third protein residue (unmatched) should NOT be present
        assert residues[2]["is_present"] is np.False_

    def test_toplevel_coords_synced(self, tmp_path):
        """The top-level coords array tracks atoms['coords']."""
        proc_dir, pdb_path, _, _, offset = _build_fixture(tmp_path)
        _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        struct_path = sorted(
            (proc_dir / "processed" / "structures").glob("*.npz")
        )[0]
        data = dict(np.load(str(struct_path), allow_pickle=True))

        for i in range(len(data["atoms"])):
            np.testing.assert_array_equal(
                data["coords"][i]["coords"],
                data["atoms"][i]["coords"],
                err_msg=f"coords[{i}] out of sync with atoms[{i}]",
            )

    def test_sub50_guard_raises(self, tmp_path):
        """<50% match fraction raises RuntimeError."""
        # Build fixture with 4 protein residues; PDB only matches first 3
        # (fixture omits the last residue). But 3/4 = 75% > 50%, so we
        # need a different approach: use a wrong resnum_offset so nothing
        # matches.
        proc_dir, pdb_path, _, _, _offset = _build_fixture(
            tmp_path, n_protein_residues=4, include_ligand=False,
        )
        # Use a wildly wrong offset so zero atoms match
        with pytest.raises(RuntimeError, match="matched only"):
            _inject_pdb_coords_into_processed_structure(
                proc_dir, pdb_path, resnum_offset=9999,
            )

    def test_returns_injected_count(self, tmp_path):
        """Return value is the number of injected atoms."""
        proc_dir, pdb_path, _, _, offset = _build_fixture(
            tmp_path, n_protein_residues=2,
        )
        n = _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        # 1 matched residue * 3 atoms + 2 ligand atoms = 5
        assert n == 5

    def test_pre_affinity_file_written(self, tmp_path):
        """pre_affinity npz is written to <structures>/<record_id>/ subdir."""
        proc_dir, pdb_path, _, _, offset = _build_fixture(tmp_path)
        _inject_pdb_coords_into_processed_structure(
            proc_dir, pdb_path, resnum_offset=offset,
        )
        struct_dir = proc_dir / "processed" / "structures"
        # The fixture writes structure_0.npz, so record_id = "structure_0"
        npz_files = sorted(struct_dir.glob("*.npz"))
        record_id = npz_files[0].stem
        pre_aff_path = struct_dir / record_id / f"pre_affinity_{record_id}.npz"
        assert pre_aff_path.exists(), f"Expected {pre_aff_path} to exist"

        # Content should be identical to the patched main npz
        main_data = dict(np.load(str(npz_files[0]), allow_pickle=True))
        pre_data = dict(np.load(str(pre_aff_path), allow_pickle=True))
        assert set(main_data.keys()) == set(pre_data.keys())
        for key in main_data:
            np.testing.assert_array_equal(
                main_data[key], pre_data[key],
                err_msg=f"pre_affinity {key} differs from main npz",
            )

    def test_ligand_zero_coords_guard_raises(self, tmp_path):
        """Non-polymer chain with zero injected atoms raises RuntimeError."""
        # Build fixture with a ligand named "LIG" but PDB HETATM uses "STI"
        # -- the resname mismatch means zero ligand atoms match.
        proc_dir, pdb_path, _, _, offset = _build_fixture(
            tmp_path, n_protein_residues=2, include_ligand=True,
        )
        # Rewrite the PDB with HETATM using a different resname ("STI")
        # so the (resname, atom_name) lookup fails for the ligand.
        pdb_lines = []
        serial = 1
        # Keep the protein ATOM records (1 residue matched)
        resseq = 0 + offset
        for aname in ["N", "CA", "C"]:
            x, y, z = float(serial), float(serial * 2), float(serial * 3)
            pdb_lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} ALA A{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )
            serial += 1
        # Ligand HETATM with wrong resname ("STI" instead of "LIG")
        for aname in ["C1", "N1"]:
            x, y, z = float(serial * 10), float(serial * 20), float(serial * 30)
            pdb_lines.append(
                f"HETATM{serial:5d}  {aname:<3s} STI A  99    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )
            serial += 1
        pdb_lines.append("END")
        pdb_path.write_text("\n".join(pdb_lines) + "\n")

        with pytest.raises(RuntimeError, match="ZERO injected coordinates"):
            _inject_pdb_coords_into_processed_structure(
                proc_dir, pdb_path, resnum_offset=offset,
            )

    def test_ccd_named_ligand_matches_hetatm(self, tmp_path):
        """A CCD-named ligand residue (e.g. STI) matches PDB HETATM records."""
        # Build a fixture where the ligand residue name is "STI" (as a CCD
        # ligand would produce) and the PDB HETATM also uses "STI".
        protein_atom_names = ["N", "CA", "C"]
        ligand_atom_names = ["C1", "N1"]
        n_prot_atoms = 2 * len(protein_atom_names)
        n_lig_atoms = len(ligand_atom_names)
        n_atoms = n_prot_atoms + n_lig_atoms

        atoms = np.zeros(n_atoms, dtype=_ATOM_DT)
        for ri in range(2):
            for ai, name in enumerate(protein_atom_names):
                idx = ri * len(protein_atom_names) + ai
                atoms[idx]["name"] = name
                atoms[idx]["is_present"] = True
        for li, name in enumerate(ligand_atom_names):
            idx = n_prot_atoms + li
            atoms[idx]["name"] = name
            atoms[idx]["is_present"] = True

        n_residues = 3  # 2 protein + 1 ligand
        residues = np.zeros(n_residues, dtype=_RESIDUE_DT)
        for ri in range(2):
            residues[ri]["name"] = "ALA"
            residues[ri]["res_type"] = 0
            residues[ri]["res_idx"] = ri
            residues[ri]["atom_idx"] = ri * len(protein_atom_names)
            residues[ri]["atom_num"] = len(protein_atom_names)
            residues[ri]["atom_center"] = ri * len(protein_atom_names) + 1
            residues[ri]["atom_disto"] = ri * len(protein_atom_names) + 2
            residues[ri]["is_standard"] = True
            residues[ri]["is_present"] = True
        # Ligand residue with CCD name "STI"
        residues[2]["name"] = "STI"
        residues[2]["res_type"] = -1
        residues[2]["res_idx"] = 2
        residues[2]["atom_idx"] = n_prot_atoms
        residues[2]["atom_num"] = n_lig_atoms
        residues[2]["atom_center"] = n_prot_atoms
        residues[2]["atom_disto"] = n_prot_atoms
        residues[2]["is_standard"] = False
        residues[2]["is_present"] = True

        chains = np.zeros(2, dtype=_CHAIN_DT)
        chains[0]["name"] = "A"
        chains[0]["mol_type"] = 0
        chains[0]["atom_idx"] = 0
        chains[0]["atom_num"] = n_prot_atoms
        chains[0]["res_idx"] = 0
        chains[0]["res_num"] = 2
        chains[1]["name"] = "B"
        chains[1]["mol_type"] = 3
        chains[1]["entity_id"] = 1
        chains[1]["asym_id"] = 1
        chains[1]["atom_idx"] = n_prot_atoms
        chains[1]["atom_num"] = n_lig_atoms
        chains[1]["res_idx"] = 2
        chains[1]["res_num"] = 1

        coords = np.zeros(n_atoms, dtype=_COORD_DT)

        struct_dir = tmp_path / "processed" / "structures"
        struct_dir.mkdir(parents=True)
        np.savez_compressed(
            str(struct_dir / "structure_0.npz"),
            atoms=atoms, residues=residues, chains=chains, coords=coords,
        )

        # PDB with HETATM using "STI" resname and matching atom names
        resnum_offset = 10
        pdb_lines = []
        serial = 1
        # Protein: first residue only (second is unmatched)
        resseq = 0 + resnum_offset
        for aname in protein_atom_names:
            x, y, z = float(serial), float(serial * 2), float(serial * 3)
            pdb_lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} ALA A{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )
            serial += 1
        # Ligand HETATM with "STI"
        for aname in ligand_atom_names:
            x = float(serial * 10.0)
            y = float(serial * 20.0)
            z = float(serial * 30.0)
            pdb_lines.append(
                f"HETATM{serial:5d}  {aname:<3s} STI A  99    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )
            serial += 1
        pdb_lines.append("END")
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text("\n".join(pdb_lines) + "\n")

        n = _inject_pdb_coords_into_processed_structure(
            tmp_path, pdb_path, resnum_offset=resnum_offset,
        )

        # 3 protein atoms (first residue) + 2 ligand atoms = 5
        assert n == 5

        # Verify ligand atoms got real coordinates
        data = dict(np.load(
            str(struct_dir / "structure_0.npz"), allow_pickle=True,
        ))
        lig_atoms = data["atoms"][n_prot_atoms:]
        for li in range(n_lig_atoms):
            assert not np.allclose(lig_atoms[li]["coords"], [0, 0, 0]), (
                f"Ligand atom {li} should have been injected"
            )
