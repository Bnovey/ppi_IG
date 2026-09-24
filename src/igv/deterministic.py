"""Deterministic featurisation: eliminate all RNG from boltz's feature pipeline.

Boltz's featurisation pipeline draws from three independent RNG sources,
none of which are seeded in the inference path:

1. **torch global RNG** -- ``center_random_augmentation`` in
   ``boltz/data/feature/featurizerv2.py:1470`` applies a random
   roto-translation to ``ref_pos`` via ``torch.randn_like``.  This is
   the **dominant source**: it moves atom positions by up to 10+ Angstrom
   between runs.  It is called **per residue group** in a loop over
   ``ref_space_uid``, consuming exactly 7 torch RNG values per group
   (4 for quaternion via ``random_quaternions``, 3 for translation via
   ``torch.randn_like(atom_coords[:, 0:1, :])``) -- constant regardless
   of atom count.  Under a fixed seed the shared residues between
   reference and mutant get identical augmented ``ref_pos``, because the
   per-call consumption is constant, so a mutation at position k does
   **not** shift the RNG state for positions after k.

   However, the rotation itself is **training-time data augmentation**
   that has no business running at inference.  Rather than relying on
   the RNG-consumption invariant (fragile against future boltz changes),
   we **disable augmentation entirely** by patching
   ``center_random_augmentation`` to centre without rotating.  Centering
   is preserved.

2. **RDKit conformer embedding** -- ``boltz/data/parse/schema.py:227``
   calls ``AllChem.EmbedMolecule(mol, options)`` without setting
   ``options.randomSeed``.  Hit during ``process_inputs`` for ligands and
   non-standard residues.  Not hit for canonical amino acids (loaded from
   pickle), but patched for generality.

3. **Python/numpy RNG** -- boltz's inference ``PredictionDataset`` seeds
   numpy at 42 (``inferencev2.py:270``), but Python's ``random`` module
   is not seeded.

:func:`deterministic_featurisation` is a context manager that:
- Patches ``center_random_augmentation`` to centre-only (no rotation/translation)
- Seeds torch, numpy, and Python ``random``
- Patches RDKit's ``EmbedMolecule`` / ``EmbedMultipleConfs`` to inject a
  fixed ``randomSeed``
- Restores everything on exit

This makes repeated featurisation of one input produce byte-identical
``ref_pos`` tensors, AND ensures that a reference and a point mutant
share identical ``ref_pos`` for their common residues.

The critical invariant: **different molecules still get different
conformers**.  The conformer geometry comes from the CCD molecule's
pre-computed 3D structure, not from the augmentation.  The augmentation
only rotated and translated it randomly; disabling it leaves each
amino acid's canonical centred conformer, which is distinct per residue
type.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Generator

log = logging.getLogger(__name__)

# Default seed.  42 matches boltz's own inference-path numpy seed
# (inferencev2.py:270).
_DEFAULT_SEED = 42

# Environment variable, following the IGV_* convention.
_ENV_VAR = "IGV_FEAT_SEED"


def resolve_feat_seed(
    seed: int | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Resolve the featurisation seed from *seed* / ``IGV_FEAT_SEED`` / default.

    Pure function, no imports beyond stdlib.

    Parameters
    ----------
    seed : int or None
        Explicit seed.  Takes precedence over the env var.
    env : dict or None
        Environment mapping; defaults to ``os.environ``.

    Returns
    -------
    int
        The resolved seed.  Always non-negative.
    """
    if seed is not None:
        return int(seed)
    if env is None:
        env = os.environ
    raw = env.get(_ENV_VAR)
    if raw is not None:
        return int(raw)
    return _DEFAULT_SEED


@contextlib.contextmanager
def deterministic_featurisation(
    seed: int | None = None,
) -> Generator[int, None, None]:
    """Seed all RNG sources and disable augmentation so boltz featurisation
    is deterministic.

    Usage::

        with deterministic_featurisation(seed=42) as s:
            feats, token_map = build_complex_feats(...)

    On entry:

    1. Patches ``boltz.model.modules.utils.center_random_augmentation`` to
       call the original with ``augmentation=False``, keeping centering but
       removing the random roto-translation.  This is the primary fix for
       the ``ref_pos`` nondeterminism.
    2. Seeds torch, numpy, and Python ``random`` -- covers
       ``random.choice(conf_ids)`` in the featurizer and any other RNG boltz
       may draw from.
    3. Patches ``rdkit.Chem.AllChem.EmbedMolecule`` / ``EmbedMultipleConfs``
       to inject ``randomSeed`` into ETKDG options -- covers conformer
       generation for ligands / non-standard residues.

    On exit, restores all state and unpatches.

    Yields the resolved seed so callers can record it in provenance.

    Parameters
    ----------
    seed : int or None
        The seed to use.  ``None`` reads ``IGV_FEAT_SEED`` from the
        environment, falling back to 42.
    """
    import random as _random

    import numpy as np

    resolved = resolve_feat_seed(seed)
    log.info("deterministic_featurisation: seed=%d", resolved)

    # -- Save state --------------------------------------------------------
    py_state = _random.getstate()
    np_state = np.random.get_state()

    # torch is optional (not installed on laptops without GPU).
    _torch_state = None
    try:
        import torch

        _torch_state = torch.random.get_rng_state()
    except ImportError:
        pass

    # -- Set seeds ---------------------------------------------------------
    _random.seed(resolved)
    np.random.seed(resolved)
    if _torch_state is not None:
        torch.manual_seed(resolved)

    # -- Patch center_random_augmentation ----------------------------------
    _cra_patches: list[tuple] = []
    try:
        _cra_patches = _patch_center_random_augmentation()
    except (ImportError, AttributeError):
        log.debug(
            "boltz not importable; skipping center_random_augmentation patch "
            "(will be covered by the torch seed alone)"
        )

    # -- Patch RDKit -------------------------------------------------------
    _rdkit_patches: list[tuple] = []
    try:
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdDistGeom

        _rdkit_patches = _patch_rdkit(AllChem, rdDistGeom, resolved)
    except ImportError:
        log.debug("rdkit not available; skipping EmbedMolecule patch")

    try:
        yield resolved
    finally:
        # -- Restore patches -----------------------------------------------
        for module, attr, original in _cra_patches:
            setattr(module, attr, original)
        for module, attr, original in _rdkit_patches:
            setattr(module, attr, original)

        # -- Restore RNG state ---------------------------------------------
        if _torch_state is not None:
            torch.random.set_rng_state(_torch_state)
        np.random.set_state(np_state)
        _random.setstate(py_state)


# ---------------------------------------------------------------------------
# Patch: center_random_augmentation -> centre-only, no augmentation
# ---------------------------------------------------------------------------


def _patch_center_random_augmentation() -> list[tuple]:
    """Replace ``center_random_augmentation`` with a centre-only version.

    The original function (``boltz/model/modules/utils.py:67``) applies
    centering, then (if ``augmentation=True``, the default) a random
    rotation and translation.  The featuriser calls it with the default
    (``featurizerv2.py:1470``), so every featurisation draws from torch's
    global RNG.

    The patch calls the original with ``augmentation=False``, preserving
    centering while eliminating the random rotation and translation.

    Returns ``[(module, attr, original)]`` for cleanup.
    """
    from boltz.model.modules import utils as _boltz_utils

    original = _boltz_utils.center_random_augmentation

    def _center_only(*args, augmentation=True, **kwargs):  # noqa: ARG001
        # Always forward augmentation=False, regardless of what the caller
        # passed.  The `augmentation` parameter is captured and discarded.
        return original(*args, augmentation=False, **kwargs)

    _boltz_utils.center_random_augmentation = _center_only

    # The featuriser imports from the module, so we must also patch the
    # attribute in any module that has already imported the name.  The
    # featuriser does `from boltz.model.modules.utils import
    # center_random_augmentation`, so its module-level name is already
    # bound.  Patch that too if it exists.
    patches = [(_boltz_utils, "center_random_augmentation", original)]
    try:
        from boltz.data.feature import featurizerv2 as _fv2

        if hasattr(_fv2, "center_random_augmentation"):
            patches.append((_fv2, "center_random_augmentation", original))
            _fv2.center_random_augmentation = _center_only
    except ImportError:
        pass
    try:
        from boltz.data.feature import featurizer as _fv1

        if hasattr(_fv1, "center_random_augmentation"):
            patches.append((_fv1, "center_random_augmentation", original))
            _fv1.center_random_augmentation = _center_only
    except ImportError:
        pass

    log.debug("Patched center_random_augmentation -> centre-only (no augmentation)")
    return patches


# ---------------------------------------------------------------------------
# Patch: RDKit EmbedMolecule / EmbedMultipleConfs
# ---------------------------------------------------------------------------


def _patch_rdkit(
    allchem_mod,
    distgeom_mod,
    seed: int,
) -> list[tuple]:
    """Monkeypatch EmbedMolecule / EmbedMultipleConfs to inject randomSeed.

    Returns a list of ``(module, attr_name, original_fn)`` for cleanup.
    """
    patches: list[tuple] = []

    orig_embed = allchem_mod.EmbedMolecule
    orig_embed_multi = allchem_mod.EmbedMultipleConfs

    def _seeded_embed(mol, params=None, *args, **kwargs):
        if params is not None and hasattr(params, "randomSeed"):
            if params.randomSeed == -1:
                params.randomSeed = seed
        if params is not None:
            return orig_embed(mol, params, *args, **kwargs)
        return orig_embed(mol, *args, **kwargs)

    def _seeded_embed_multi(mol, numConfs=10, params=None, *args, **kwargs):
        if params is not None and hasattr(params, "randomSeed"):
            if params.randomSeed == -1:
                params.randomSeed = seed
        if params is not None:
            return orig_embed_multi(mol, numConfs, params, *args, **kwargs)
        return orig_embed_multi(mol, numConfs, *args, **kwargs)

    # Patch on AllChem (primary import path in boltz).
    allchem_mod.EmbedMolecule = _seeded_embed
    patches.append((allchem_mod, "EmbedMolecule", orig_embed))

    allchem_mod.EmbedMultipleConfs = _seeded_embed_multi
    patches.append((allchem_mod, "EmbedMultipleConfs", orig_embed_multi))

    # rdDistGeom is the canonical module; AllChem re-exports from it.
    # If they are the same object, patching AllChem already covers it.
    # If not (future rdkit refactor), patch both.
    if distgeom_mod.EmbedMolecule is not _seeded_embed:
        distgeom_mod.EmbedMolecule = _seeded_embed
        patches.append((distgeom_mod, "EmbedMolecule", orig_embed))
    if distgeom_mod.EmbedMultipleConfs is not _seeded_embed_multi:
        distgeom_mod.EmbedMultipleConfs = _seeded_embed_multi
        patches.append((distgeom_mod, "EmbedMultipleConfs", orig_embed_multi))

    log.debug("Patched EmbedMolecule/EmbedMultipleConfs with randomSeed=%d", seed)
    return patches
