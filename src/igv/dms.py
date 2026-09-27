"""Starr et al. 2020 SARS-CoV-2 RBD deep mutational scan loader."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from igv.data import _atomic_download
from igv.metrics import spearman
from igv.skempi import _R_KCAL

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARR2020_URL = (
    "https://media.githubusercontent.com/media/jbloomlab/"
    "SARS-CoV-2-RBD_DMS/master/results/single_mut_effects/single_mut_effects.csv"
)

# R·T·ln(10) at 298 K in kcal/mol — converts Δlog10(KD) to ΔΔG.
DDG_PER_LOG10 = 2.303 * _R_KCAL * 298  # ≈ 1.3636

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Complex registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DmsComplex:
    """A protein-protein complex for a deep mutational scan.

    ``mutated_chain`` is the chain that was scanned.
    ``partner_chains`` are the binding partner(s).
    """

    pdb_id: str
    mutated_chain: str
    partner_chains: tuple[str, ...]

    @property
    def all_chains(self) -> tuple[str, ...]:
        """Chains to featurise (mutated + partners)."""
        return (self.mutated_chain,) + self.partner_chains


DMS_COMPLEXES: dict[str, DmsComplex] = {
    "spike_rbd": DmsComplex(
        pdb_id="6M0J",
        mutated_chain="E",
        partner_chains=("A",),
    ),
}


def get_complex(key: str) -> DmsComplex:
    """Look up a registered DMS complex by key."""
    try:
        return DMS_COMPLEXES[key]
    except KeyError:
        raise KeyError(
            f"Unknown DMS complex {key!r}. "
            f"Registered keys: {sorted(DMS_COMPLEXES)}"
        ) from None


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------

def download_starr2020(cache_dir: Path) -> Path:
    """Download the Starr et al. 2020 single-mutant-effects CSV, caching to disk.

    Writes atomically via a temp file + os.replace so interrupted downloads
    never leave a corrupt cache entry.
    """
    cache_dir = Path(cache_dir)
    dest = cache_dir / "single_mut_effects.csv"

    if dest.exists():
        return dest

    _atomic_download(STARR2020_URL, dest)
    return dest


def load_starr2020(cache_dir: Path) -> pd.DataFrame:
    """Download (if needed) and read the Starr 2020 single-mutant-effects CSV."""
    path = download_starr2020(cache_dir)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def singles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop synonymous rows and rows missing the binding readout."""
    mask = (df["mutant"] != df["wildtype"]) & df["bind_avg"].notna()
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def binding_ddg(delta_log10_kd: float | np.ndarray) -> float | np.ndarray:
    """Convert Starr Δlog10(KD,app) to ΔΔG in kcal/mol on SKEMPI's sign convention.

    SKEMPI: positive ΔΔG = destabilising (weaker binding).
    Starr:  positive bind_avg = tighter binding (lower KD).

    The conversion negates: ΔΔG = −Δlog10(KD) × R·T·ln(10).
    """
    return -delta_log10_kd * DDG_PER_LOG10


# ---------------------------------------------------------------------------
# Site mapping
# ---------------------------------------------------------------------------

def map_sites_to_indices(
    df: pd.DataFrame,
    residue_ids: list[str],
    sequence: str,
    *,
    allow_mismatch: bool = False,
) -> tuple[dict[int, int], list[tuple[int, str, str]]]:
    """Map dataset Spike-numbering sites to 0-based indices into chain E.

    For each unique ``site_SARS2`` in *df* that appears in *residue_ids*,
    verifies the dataset's ``wildtype`` amino acid against the PDB sequence.

    Returns ``(mapped, mismatches)`` where *mapped* is a dict of
    ``{site_SARS2: 0_based_index}`` and *mismatches* is a list of
    ``(site, expected_aa, found_aa)`` triples.

    Raises ``ValueError`` if any mismatch is found and *allow_mismatch* is
    False.  Sites not present in the PDB are silently skipped.
    """
    id_to_idx = {rid: i for i, rid in enumerate(residue_ids)}

    mapped: dict[int, int] = {}
    mismatches: list[tuple[int, str, str]] = []

    sites = df.drop_duplicates("site_SARS2")[["site_SARS2", "wildtype"]]
    for _, row in sites.iterrows():
        site = int(row["site_SARS2"])
        site_str = str(site)
        wt = row["wildtype"]

        if site_str not in id_to_idx:
            continue

        idx = id_to_idx[site_str]
        pdb_aa = sequence[idx]

        if pdb_aa != wt:
            mismatches.append((site, wt, pdb_aa))
            if allow_mismatch:
                log.warning(
                    "Wild-type mismatch at site %d: dataset says %s, PDB has %s (index %d)",
                    site, wt, pdb_aa, idx,
                )
            continue

        mapped[site] = idx

    if mismatches and not allow_mismatch:
        raise ValueError(
            f"{len(mismatches)} wild-type mismatch(es). "
            f"First: site {mismatches[0][0]} expected {mismatches[0][1]}, "
            f"found {mismatches[0][2]}. Pass allow_mismatch=True to skip."
        )

    return mapped, mismatches


# ---------------------------------------------------------------------------
# Substitution matrix
# ---------------------------------------------------------------------------

def substitution_matrix(
    df: pd.DataFrame,
    *,
    value_col: str,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Build a position × 20 matrix of mutation effects.

    Rows = positions present in *df* (sorted by ``site_SARS2``).
    Columns = the 20 standard amino acids in ``AA_ORDER``.
    Cells = the value from *value_col*; ``np.nan`` where unmeasured.

    Returns ``(matrix, positions, aa_order)``.
    """
    aa_to_col = {aa: i for i, aa in enumerate(AA_ORDER)}
    positions = sorted(df["site_SARS2"].unique())
    pos_to_row = {p: i for i, p in enumerate(positions)}

    mat = np.full((len(positions), 20), np.nan)
    for _, row in df.iterrows():
        r = pos_to_row[row["site_SARS2"]]
        c = aa_to_col.get(row["mutant"])
        if c is not None and pd.notna(row[value_col]):
            mat[r, c] = row[value_col]

    return mat, positions, AA_ORDER


# ---------------------------------------------------------------------------
# Within-position correlation
# ---------------------------------------------------------------------------

def within_position_correlation(
    matrix: np.ndarray,
    values_per_aa: dict[str, float],
) -> np.ndarray:
    """Per-position Spearman correlation between a matrix row and an amino-acid property.

    Skips positions with fewer than 5 non-NaN cells or zero variance in either
    vector.  Uses :func:`igv.metrics.spearman` which handles ties correctly.

    Returns a 1-D array aligned with *matrix* rows; skipped positions are NaN.
    """
    prop_vec = np.array([values_per_aa.get(aa, np.nan) for aa in AA_ORDER])
    n_pos = matrix.shape[0]
    result = np.full(n_pos, np.nan)

    for i in range(n_pos):
        row = matrix[i]
        valid = np.isfinite(row) & np.isfinite(prop_vec)
        if valid.sum() < 5:
            continue
        r_vals = row[valid]
        p_vals = prop_vec[valid]
        if np.std(r_vals) == 0 or np.std(p_vals) == 0:
            continue
        result[i] = spearman(r_vals, p_vals)

    return result
