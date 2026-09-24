"""Tests for igv.deterministic -- the seeding mechanism for boltz featurisation.

These tests run without boltz (not installed on this machine) and without
GPU.  They verify:
  1. The context manager installs and restores RNG state correctly.
  2. The RDKit EmbedMolecule patch injects the seed and is scoped.
  3. Same seed -> identical conformers; different seeds -> different conformers.
  4. Different molecules under the same seed -> DIFFERENT conformers
     (the critical correctness requirement).
  5. The env-var resolution follows the IGV_* convention.
  6. A reference and a mutant differing by one residue share identical
     augmented ref_pos for their common residues (the embedding-delta
     correctness requirement).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.deterministic import (  # noqa: E402
    _DEFAULT_SEED,
    _ENV_VAR,
    deterministic_featurisation,
    resolve_feat_seed,
)


# ---------------------------------------------------------------------------
# resolve_feat_seed
# ---------------------------------------------------------------------------


class TestResolveFeatSeed:
    def test_explicit_seed_wins(self):
        assert resolve_feat_seed(seed=99, env={"IGV_FEAT_SEED": "7"}) == 99

    def test_env_var(self):
        assert resolve_feat_seed(seed=None, env={"IGV_FEAT_SEED": "123"}) == 123

    def test_default(self):
        assert resolve_feat_seed(seed=None, env={}) == _DEFAULT_SEED

    def test_env_var_name(self):
        assert _ENV_VAR == "IGV_FEAT_SEED"


# ---------------------------------------------------------------------------
# Context manager: RNG state save/restore
# ---------------------------------------------------------------------------


class TestContextManagerStateRestore:
    """Verify that the context manager restores all RNG state on exit."""

    def test_python_random_restored(self):
        import random

        random.seed(999)
        state_before = random.getstate()
        with deterministic_featurisation(seed=42):
            # Inside, the state is different
            state_inside = random.getstate()
            assert state_inside != state_before
        state_after = random.getstate()
        assert state_after == state_before

    def test_numpy_random_restored(self):
        np.random.seed(999)
        state_before = np.random.get_state()
        with deterministic_featurisation(seed=42):
            pass
        state_after = np.random.get_state()
        # Compare the key array in the state tuple
        assert np.array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]

    def test_yields_resolved_seed(self):
        with deterministic_featurisation(seed=77) as s:
            assert s == 77

    def test_yields_default_seed_when_none(self):
        with deterministic_featurisation() as s:
            assert s == _DEFAULT_SEED


# ---------------------------------------------------------------------------
# RDKit EmbedMolecule patch
# ---------------------------------------------------------------------------


class TestRDKitPatch:
    """Verify the EmbedMolecule/EmbedMultipleConfs monkeypatch."""

    def test_patch_is_scoped(self):
        from rdkit.Chem import AllChem

        original = AllChem.EmbedMolecule
        with deterministic_featurisation(seed=42):
            assert AllChem.EmbedMolecule is not original
        assert AllChem.EmbedMolecule is original

    def test_patch_restores_on_exception(self):
        from rdkit.Chem import AllChem

        original = AllChem.EmbedMolecule
        with pytest.raises(RuntimeError):
            with deterministic_featurisation(seed=42):
                assert AllChem.EmbedMolecule is not original
                raise RuntimeError("deliberate")
        assert AllChem.EmbedMolecule is original

    def test_patched_embed_injects_seed(self):
        """EmbedMolecule called with default randomSeed=-1 should get the seed."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC(=O)O")
        mol = Chem.AddHs(mol)
        opts = AllChem.ETKDGv3()
        assert opts.randomSeed == -1  # default

        with deterministic_featurisation(seed=42):
            AllChem.EmbedMolecule(mol, opts)

        # After the patched call, the options object should have had
        # randomSeed set to 42
        assert opts.randomSeed == 42

    def test_patched_embed_preserves_explicit_seed(self):
        """If the caller already set randomSeed != -1, do not overwrite."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC(=O)O")
        mol = Chem.AddHs(mol)
        opts = AllChem.ETKDGv3()
        opts.randomSeed = 999

        with deterministic_featurisation(seed=42):
            AllChem.EmbedMolecule(mol, opts)

        assert opts.randomSeed == 999


# ---------------------------------------------------------------------------
# Conformer determinism: same molecule, same seed -> identical positions
# ---------------------------------------------------------------------------


def _embed_and_get_positions(smiles, seed):
    """Embed a molecule under the context manager and return atom positions."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    opts = AllChem.ETKDGv3()
    # Leave randomSeed at -1 so the patch kicks in
    with deterministic_featurisation(seed=seed):
        AllChem.EmbedMolecule(mol, opts)
    conf = mol.GetConformer(0)
    return np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])


class TestConformerDeterminism:
    """The core contract: same seed -> identical; different molecules -> different."""

    def test_same_seed_same_molecule_identical(self):
        pos1 = _embed_and_get_positions("CC(=O)O", seed=42)
        pos2 = _embed_and_get_positions("CC(=O)O", seed=42)
        np.testing.assert_array_equal(pos1, pos2)

    def test_different_seed_same_molecule_differs(self):
        pos1 = _embed_and_get_positions("CC(=O)O", seed=42)
        pos2 = _embed_and_get_positions("CC(=O)O", seed=99)
        assert not np.array_equal(pos1, pos2)

    def test_different_molecules_same_seed_differ(self):
        """Critical correctness requirement: the fix must not make all residues
        share one conformer.  ALA and ARG must still get distinct 3D coordinates."""
        # Use amino acid SMILES that are structurally very different
        ala_smiles = "C[C@@H](N)C(=O)O"  # alanine
        arg_smiles = "N=C(N)NCCC[C@@H](N)C(=O)O"  # arginine
        pos_ala = _embed_and_get_positions(ala_smiles, seed=42)
        pos_arg = _embed_and_get_positions(arg_smiles, seed=42)
        # Different number of atoms already proves they differ
        assert pos_ala.shape != pos_arg.shape

    def test_different_similar_molecules_same_seed_differ(self):
        """Even molecules with the same atom count must get different conformers."""
        # Leucine and isoleucine: same formula, different structure
        leu_smiles = "CC(C)C[C@@H](N)C(=O)O"
        ile_smiles = "CC[C@H](C)[C@@H](N)C(=O)O"
        pos_leu = _embed_and_get_positions(leu_smiles, seed=42)
        pos_ile = _embed_and_get_positions(ile_smiles, seed=42)
        # Same number of heavy+H atoms after AddHs, but positions must differ
        # because the molecular graphs are different
        assert not np.allclose(pos_leu, pos_ile, atol=1e-6)

    def test_repeated_embeds_in_one_context_are_deterministic(self):
        """Multiple EmbedMolecule calls inside a single context should be
        reproducible when the whole context is repeated."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        smiles_list = ["CC(=O)O", "CCO", "c1ccccc1"]

        def embed_batch(seed):
            results = []
            with deterministic_featurisation(seed=seed):
                for smi in smiles_list:
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    opts = AllChem.ETKDGv3()
                    AllChem.EmbedMolecule(mol, opts)
                    conf = mol.GetConformer(0)
                    pos = np.array([
                        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())
                    ])
                    results.append(pos)
            return results

        batch1 = embed_batch(42)
        batch2 = embed_batch(42)
        for p1, p2 in zip(batch1, batch2):
            np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# EmbedMultipleConfs patch
# ---------------------------------------------------------------------------


class TestEmbedMultipleConfsPatch:
    def test_multiple_confs_patched(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        def get_confs(seed):
            mol = Chem.MolFromSmiles("CCCC")
            mol = Chem.AddHs(mol)
            opts = AllChem.ETKDGv3()
            with deterministic_featurisation(seed=seed):
                AllChem.EmbedMultipleConfs(mol, numConfs=3, params=opts)
            positions = []
            for conf in mol.GetConformers():
                pos = np.array([
                    [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())
                ])
                positions.append(pos)
            return positions

        confs1 = get_confs(42)
        confs2 = get_confs(42)
        assert len(confs1) == len(confs2)
        for c1, c2 in zip(confs1, confs2):
            np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# Reference vs mutant: shared residues must agree
# ---------------------------------------------------------------------------


def _copysign(a, b):
    """Reproduce boltz/model/modules/utils.py:_copysign."""
    import torch

    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _center_random_augmentation(atom_coords, atom_mask, augmentation=True, s_trans=1.0):
    """Reproduce the exact RNG-consuming logic of boltz's function.

    boltz/model/modules/utils.py:67 -- center_random_augmentation.
    """
    import torch

    # centering (always applied in the featuriser call)
    denom = torch.sum(atom_mask[:, :, None], dim=1, keepdim=True).clamp(min=1e-8)
    atom_mean = torch.sum(
        atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
    ) / denom
    atom_coords = atom_coords - atom_mean

    if augmentation:
        # randomly_rotate -> random_rotations(1) -> random_quaternions(1)
        o = torch.randn((1, 4), dtype=atom_coords.dtype, device=atom_coords.device)
        s = (o * o).sum(1)
        o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
        r, i, j, k = torch.unbind(o, -1)
        two_s = 2.0 / (o * o).sum(-1)
        R = torch.stack([
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ], -1).reshape(o.shape[:-1] + (3, 3))
        atom_coords = torch.einsum("bmd,bds->bms", atom_coords, R)
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

    return atom_coords


def _run_featuriser_loop(residue_conformers, seed, augmentation=True):
    """Simulate boltz featurizerv2.py:1467-1473 per-residue augmentation loop.

    Each entry in residue_conformers is a (n_atoms, 3) tensor.
    Returns a list of augmented (n_atoms, 3) tensors.
    """
    import torch

    torch.manual_seed(seed)
    results = []
    for atoms in residue_conformers:
        mask = torch.ones(1, atoms.shape[0])
        out = _center_random_augmentation(
            atoms.unsqueeze(0), mask, augmentation=augmentation
        )
        results.append(out.squeeze(0))
    return results


class TestReferenceVsMutant:
    """The embedding-delta correctness requirement.

    A substitution at position k changes one residue's conformer while all
    others stay the same.  Under the deterministic_featurisation fix, the
    shared residues' ref_pos must be IDENTICAL between reference and mutant.
    """

    def test_shared_residues_identical_with_augmentation_disabled(self):
        """With augmentation=False (the fix), shared residues are trivially
        identical because the only operation is centering, which is
        deterministic and depends only on the residue's own atoms."""
        import torch

        # Create conformers: 3 residues, middle one substituted
        shared_0 = torch.randn(5, 3)   # e.g. ALA (5 heavy atoms)
        shared_2 = torch.randn(10, 3)  # e.g. TRP
        ref_mid = torch.randn(7, 3)    # reference residue
        mut_mid = torch.randn(12, 3)   # mutant residue (different atom count)

        ref_results = _run_featuriser_loop(
            [shared_0.clone(), ref_mid, shared_2.clone()],
            seed=42, augmentation=False,
        )
        mut_results = _run_featuriser_loop(
            [shared_0.clone(), mut_mid, shared_2.clone()],
            seed=42, augmentation=False,
        )

        # Shared residues must be byte-identical
        assert torch.equal(ref_results[0], mut_results[0]), (
            f"Residue 0 differs: max_abs="
            f"{(ref_results[0] - mut_results[0]).abs().max()}"
        )
        assert torch.equal(ref_results[2], mut_results[2]), (
            f"Residue 2 differs: max_abs="
            f"{(ref_results[2] - mut_results[2]).abs().max()}"
        )

    def test_shared_residues_identical_even_with_augmentation(self):
        """Even with augmentation=True under the same seed, shared residues
        agree because center_random_augmentation consumes exactly 7 RNG
        values per call regardless of atom count.  This test documents the
        invariant that makes the seed-only approach sufficient IF boltz does
        not change the per-call consumption.  The augmentation=False patch
        does not depend on this invariant."""
        import torch

        shared_0 = torch.randn(5, 3)
        shared_2 = torch.randn(10, 3)
        ref_mid = torch.randn(7, 3)
        mut_mid = torch.randn(12, 3)

        ref_results = _run_featuriser_loop(
            [shared_0.clone(), ref_mid, shared_2.clone()],
            seed=42, augmentation=True,
        )
        mut_results = _run_featuriser_loop(
            [shared_0.clone(), mut_mid, shared_2.clone()],
            seed=42, augmentation=True,
        )

        assert torch.equal(ref_results[0], mut_results[0])
        assert torch.equal(ref_results[2], mut_results[2])

    def test_mutated_residue_still_differs(self):
        """The mutated residue must still have different ref_pos, because
        it is a different molecule with different conformer geometry."""
        import torch

        ref_mid = torch.randn(7, 3)
        mut_mid = torch.randn(12, 3)

        ref_results = _run_featuriser_loop([ref_mid], seed=42, augmentation=False)
        mut_results = _run_featuriser_loop([mut_mid], seed=42, augmentation=False)

        # Different shapes -> different (different amino acids have different
        # numbers of atoms)
        assert ref_results[0].shape != mut_results[0].shape

    def test_rng_consumption_constant_per_call(self):
        """Each center_random_augmentation call with augmentation=True
        consumes exactly 7 torch RNG values (4 quaternion + 3 translation),
        regardless of atom count."""
        import torch

        states = []
        for n_atoms in [3, 5, 7, 10, 14, 20, 50]:
            torch.manual_seed(42)
            atoms = torch.zeros(n_atoms, 3)  # static, no RNG
            mask = torch.ones(1, n_atoms)
            _center_random_augmentation(atoms.unsqueeze(0), mask, augmentation=True)
            states.append(torch.random.get_rng_state().numpy().tobytes())

        # All states after consuming 7 values must be identical
        for i in range(1, len(states)):
            assert states[i] == states[0], (
                f"n_atoms variant {i} produced different RNG state"
            )
