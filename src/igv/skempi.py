"""SKEMPI 2.0 data loader: download, parse mutations, and compute ΔΔG."""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKEMPI_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"

_R_KCAL = 1.987204259e-3  # kcal/(mol·K)

_MUTATION_RE = re.compile(r"^([A-Z])([A-Za-z])(.+?)([A-Z])$")


# ---------------------------------------------------------------------------
# Complex registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkempiComplex:
    pdb_id: str
    ab_chains: tuple[str, ...]
    ag_chains: tuple[str, ...]


SKEMPI_COMPLEXES: dict[str, SkempiComplex] = {
    "3HFM": SkempiComplex(pdb_id="3HFM", ab_chains=("H", "L"), ag_chains=("Y",)),
    "1VFB": SkempiComplex(pdb_id="1VFB", ab_chains=("A", "B"), ag_chains=("C",)),
    "1MHP": SkempiComplex(pdb_id="1MHP", ab_chains=("H", "L"), ag_chains=("A",)),
}


def get_complex(key: str) -> SkempiComplex:
    """Look up a registered SKEMPI complex by its PDB key (case-insensitive)."""
    try:
        return SKEMPI_COMPLEXES[key.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown SKEMPI complex {key!r}. "
            f"Registered keys: {sorted(SKEMPI_COMPLEXES)}"
        ) from None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_skempi(cache_dir: Path) -> Path:
    """Download the SKEMPI 2.0 CSV, caching to disk.

    Writes atomically via a temp file + os.replace so interrupted downloads
    never leave a corrupt cache entry.
    """
    cache_dir = Path(cache_dir)
    dest = cache_dir / "skempi_v2.csv"

    if dest.exists():
        return dest

    cache_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SKEMPI_URL, timeout=120)
    resp.raise_for_status()

    fd, tmp = tempfile.mkstemp(dir=cache_dir)
    try:
        os.write(fd, resp.content)
        os.close(fd)
        os.replace(tmp, dest)
    except BaseException:
        os.close(fd)
        os.unlink(tmp)
        raise

    return dest


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_skempi(cache_dir: Path) -> pd.DataFrame:
    """Download (if needed) and read the SKEMPI 2.0 semicolon-delimited CSV."""
    path = download_skempi(cache_dir)
    return pd.read_csv(path, sep=";")


# ---------------------------------------------------------------------------
# Mutation parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mutation:
    wt_aa: str
    chain: str
    resnum: str
    mut_aa: str


def parse_mutation(s: str) -> Mutation:
    """Parse a single mutation code like ``LI38G``.

    Format: <wt_aa><chain><resnum><mut_aa> where resnum may contain
    insertion codes (e.g. ``38A``).
    """
    m = _MUTATION_RE.match(s)
    if m is None:
        raise ValueError(f"Malformed mutation code: {s!r}")
    return Mutation(wt_aa=m.group(1), chain=m.group(2), resnum=m.group(3), mut_aa=m.group(4))


def parse_mutations(s: str) -> list[Mutation]:
    """Parse a comma-separated string of mutation codes."""
    return [parse_mutation(part.strip()) for part in s.split(",")]


# ---------------------------------------------------------------------------
# ΔΔG
# ---------------------------------------------------------------------------

def ddg(kd_mut: float, kd_wt: float, temperature_k: float = 298.0) -> float:
    """Return ΔΔG in kcal/mol: ``R * T * ln(Kd_mut / Kd_wt)``.

    Positive ΔΔG means the mutation weakens binding (Kd goes up).
    """
    if kd_mut <= 0 or kd_wt <= 0:
        raise ValueError(
            f"Kd values must be positive, got kd_mut={kd_mut}, kd_wt={kd_wt}"
        )
    return _R_KCAL * temperature_k * math.log(kd_mut / kd_wt)


def _parse_temperature(raw: object) -> float:
    """Extract a numeric temperature from messy free text. Default 298.0 K."""
    if pd.isna(raw):
        return 298.0
    s = str(raw).strip()
    m = re.match(r"(\d+\.?\d*)", s)
    if m is None:
        return 298.0
    return float(m.group(1))


def add_ddg(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ddg_kcal_mol`` and ``temperature_k`` columns; drop rows lacking affinity."""
    df = df.copy()
    df["temperature_k"] = df["Temperature"].apply(_parse_temperature)

    mask = df["Affinity_mut_parsed"].notna() & df["Affinity_wt_parsed"].notna()
    df = df[mask].reset_index(drop=True)

    positive = (df["Affinity_mut_parsed"] > 0) & (df["Affinity_wt_parsed"] > 0)
    df = df[positive].reset_index(drop=True)

    df["ddg_kcal_mol"] = np.vectorize(ddg)(
        df["Affinity_mut_parsed"].values,
        df["Affinity_wt_parsed"].values,
        df["temperature_k"].values,
    )
    return df


# ---------------------------------------------------------------------------
# DataFrame filters
# ---------------------------------------------------------------------------

def single_point(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with exactly one mutation."""
    counts = df["Mutation(s)_cleaned"].str.count(",") + 1
    return df[counts == 1].reset_index(drop=True)


def filter_complex(df: pd.DataFrame, pdb: str) -> pd.DataFrame:
    """Case-insensitive match on the PDB id portion of ``#Pdb``."""
    pdb_lower = pdb.lower()
    mask = df["#Pdb"].str.split("_").str[0].str.lower() == pdb_lower
    return df[mask].reset_index(drop=True)


def antibody_antigen(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose ``Hold_out_type`` contains ``AB/AG``."""
    mask = df["Hold_out_type"].fillna("").str.contains("AB/AG", regex=False)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# PDB residue-ID reader (mirrors read_pdb_chains ordering)
# ---------------------------------------------------------------------------

_THREE_TO_ONE_SKEMPI = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_pdb_residue_ids(
    pdb_path: Path,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return per-chain residue identifiers and sequences from ATOM records.

    Uses the same deduplication and ordering logic as
    ``igv.data.read_pdb_chains`` so that ``residue_ids[ch][i]`` corresponds to
    the *i*-th character of the sequence for chain *ch*.

    Returns ``(residue_ids, sequences)`` where *residue_ids* maps chain to a
    list of author residue IDs (resSeq + iCode, stripped) and *sequences* maps
    chain to its one-letter sequence.
    """
    ids: dict[str, list[str]] = {}
    seqs: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            res_key = line[22:27].strip()
            resname = line[17:20].strip()

            if chain not in seen:
                seen[chain] = set()
                ids[chain] = []
                seqs[chain] = []

            if res_key not in seen[chain]:
                seen[chain].add(res_key)
                ids[chain].append(res_key)
                seqs[chain].append(_THREE_TO_ONE_SKEMPI.get(resname, "X"))

    from igv.data import read_pdb_chains

    ref_chains = read_pdb_chains(Path(pdb_path))
    for ch in ref_chains:
        ref_seq = ref_chains[ch]
        my_seq = "".join(seqs.get(ch, []))
        if len(ids.get(ch, [])) != len(ref_seq):
            raise AssertionError(
                f"Chain {ch}: read_pdb_residue_ids produced "
                f"{len(ids.get(ch, []))} residues but read_pdb_chains "
                f"produced {len(ref_seq)}"
            )
        if my_seq != ref_seq:
            raise AssertionError(
                f"Chain {ch}: sequence mismatch between read_pdb_residue_ids "
                f"and read_pdb_chains"
            )

    sequences = {ch: "".join(aa_list) for ch, aa_list in seqs.items()}
    return ids, sequences


# ---------------------------------------------------------------------------
# Map SKEMPI mutations to sequence indices with wild-type guard
# ---------------------------------------------------------------------------

def map_mutations_to_indices(
    mutations: list[Mutation],
    residue_ids: list[str],
    sequence: str,
    *,
    allow_mismatch: bool = False,
) -> tuple[list[tuple[Mutation, int]], list[tuple[Mutation, str, str]]]:
    """Map parsed mutations to 0-based sequence indices.

    For each mutation, locates ``mutation.resnum`` in *residue_ids* and verifies
    that the wild-type amino acid matches the PDB sequence at that position.

    Returns ``(mapped, mismatches)`` where *mapped* is a list of
    ``(mutation, index)`` pairs and *mismatches* is a list of
    ``(mutation, expected_aa, found_aa)`` triples.

    Raises ``ValueError`` if any mismatch is found and *allow_mismatch* is
    False.
    """
    id_to_idx = {rid: i for i, rid in enumerate(residue_ids)}

    mapped: list[tuple[Mutation, int]] = []
    mismatches: list[tuple[Mutation, str, str]] = []
    not_found: list[Mutation] = []

    for mut in mutations:
        if mut.resnum not in id_to_idx:
            not_found.append(mut)
            continue
        idx = id_to_idx[mut.resnum]
        pdb_aa = sequence[idx]
        if pdb_aa != mut.wt_aa:
            mismatches.append((mut, mut.wt_aa, pdb_aa))
            if allow_mismatch:
                log.warning(
                    "Wild-type mismatch for %s%s%s%s: expected %s, "
                    "found %s at index %d",
                    mut.wt_aa, mut.chain, mut.resnum, mut.mut_aa,
                    mut.wt_aa, pdb_aa, idx,
                )
            continue
        mapped.append((mut, idx))

    if not_found:
        raise ValueError(
            f"{len(not_found)} mutation(s) reference residue numbers not "
            f"found in the PDB: {not_found}"
        )

    if mismatches and not allow_mismatch:
        raise ValueError(
            f"{len(mismatches)} wild-type mismatch(es) — the residue "
            f"numbering is wrong or the PDB does not match SKEMPI. "
            f"First: {mismatches[0][0]} expected {mismatches[0][1]}, "
            f"found {mismatches[0][2]}. Use --allow-mismatch to skip."
        )

    return mapped, mismatches
