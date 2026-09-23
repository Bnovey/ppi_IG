"""AbBiBench data loader: download, parse, and build mutant libraries."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AFFINITY_NAMES = [
    "1mhp_LC",
    "1mlc",
    "1n8z",
    "2fjg",
    "3gbn_h1",
    "3gbn_h9",
    "5a12_ang2",
    "5a12_vegf",
    "aayl49",
    "aayl49_ML",
    "aayl50_LC",
    "aayl51",
    "aayl52_LC",
    "g6_LC",
    "4fqi_h1",
    "4fqi_h3",
    "4d5_her2",
]

_STRUCTURE_NAMES = [
    "1mhp",
    "1mhp_hla",
    "1mlc_bae",
    "1n8z_bac",
    "2fjg_hlv",
    "3gbn_hlab",
    "4fqi_hlab",
    "4zff_hld",
    "4zfg_hla",
    "AAYL49_bca",
    "AAYL50_bca",
    "AAYL51_bca",
    "AAYL52_bca",
]

_HF_BASE = (
    "https://huggingface.co/datasets/AbBibench/"
    "Antibody_Binding_Benchmark_Dataset/resolve/main"
)

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_EXPECTED_COLUMNS = {"heavy_chain_seq", "binding_score", "light_chain_seq"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _affinity_url(name: str) -> str:
    if name == "4d5_her2":
        return f"{_HF_BASE}/binding_affinity/4d5_her2_benchmarking_data_trimmed.csv"
    return f"{_HF_BASE}/binding_affinity/{name}_benchmarking_data.csv"


def _structure_url(name: str) -> str:
    return f"{_HF_BASE}/complex_structure/{name}.pdb"


_RCSB_BASE = "https://files.rcsb.org/download"


def _rcsb_url(pdb_id: str) -> str:
    return f"{_RCSB_BASE}/{pdb_id.upper()}.pdb"


def _atomic_download(url: str, dest: Path) -> None:
    """Fetch *url* and write to *dest* atomically via temp file + os.replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    fd, tmp = tempfile.mkstemp(dir=dest.parent)
    try:
        os.write(fd, resp.content)
        os.close(fd)
        os.replace(tmp, dest)
    except BaseException:
        os.close(fd)
        os.unlink(tmp)
        raise


def download(name: str, kind: str, cache_dir: Path) -> Path:
    """Download an affinity CSV or structure PDB, caching to disk.

    Writes atomically via a temp file + os.replace so interrupted downloads
    never leave a corrupt cache entry.
    """
    cache_dir = Path(cache_dir)
    if kind == "affinity":
        if name == "4d5_her2":
            dest = cache_dir / "4d5_her2_benchmarking_data_trimmed.csv"
        else:
            dest = cache_dir / f"{name}_benchmarking_data.csv"
        url = _affinity_url(name)
    elif kind == "structure":
        dest = cache_dir / f"{name}.pdb"
        url = _structure_url(name)
    else:
        raise ValueError(f"kind must be 'affinity' or 'structure', got {kind!r}")

    if dest.exists():
        return dest

    _atomic_download(url, dest)
    return dest


def download_rcsb(pdb_id: str, cache_dir: Path) -> Path:
    """Download a PDB file from RCSB by its 4-character id, caching to disk.

    Writes atomically via a temp file + os.replace so interrupted downloads
    never leave a corrupt cache entry.  Cached as ``{pdb_id_lower}.pdb``.
    """
    cache_dir = Path(cache_dir)
    dest = cache_dir / f"{pdb_id.lower()}.pdb"

    if dest.exists():
        return dest

    _atomic_download(_rcsb_url(pdb_id), dest)
    return dest


# ---------------------------------------------------------------------------
# Affinity loading
# ---------------------------------------------------------------------------

def load_affinity(name: str, cache_dir: Path) -> pd.DataFrame:
    """Download (if needed) and read an affinity CSV. Asserts expected columns."""
    path = download(name, "affinity", cache_dir)
    df = pd.read_csv(path)
    missing = _EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns in {name}: {missing}"
    return df


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def read_pdb_chains(pdb_path: Path) -> dict[str, str]:
    """Parse ATOM records and return {chain_id: one-letter sequence}.

    Deduplicates on (chain, resseq+icode) using columns 22:27.
    Unknown residues become 'X'.
    """
    chains: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[str]] = {}

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line.startswith("HETATM"):
                continue

            chain = line[21]
            res_key = line[22:27].strip()
            resname = line[17:20].strip()

            if chain not in seen:
                seen[chain] = set()
                chains[chain] = []

            if res_key not in seen[chain]:
                seen[chain].add(res_key)
                aa = _THREE_TO_ONE.get(resname, "X")
                chains[chain].append((res_key, aa))

    return {ch: "".join(aa for _, aa in residues) for ch, residues in chains.items()}


# ---------------------------------------------------------------------------
# Variable-position detection
# ---------------------------------------------------------------------------

def find_variable_positions(seqs: Sequence[str]) -> list[int]:
    """Return sorted 0-based indices where more than one character occurs.

    All sequences must have the same length.
    """
    if not seqs:
        return []
    length = len(seqs[0])
    for i, s in enumerate(seqs):
        if len(s) != length:
            raise ValueError(
                f"Sequence {i} has length {len(s)}, expected {length}. "
                "All sequences must share one length."
            )
    variable = []
    for pos in range(length):
        chars = {s[pos] for s in seqs}
        if len(chars) > 1:
            variable.append(pos)
    return variable


# ---------------------------------------------------------------------------
# Substitution derivation (pure logic, no I/O)
# ---------------------------------------------------------------------------

def _derive_substitutions(
    seqs: Sequence[str],
    reference: str,
    variable_positions: list[int],
) -> list[tuple[tuple[int, str], ...]]:
    """For each sequence, return the (position, mutant_aa) pairs where it differs
    from the reference at variable positions."""
    result = []
    for seq in seqs:
        subs = []
        for pos in variable_positions:
            if seq[pos] != reference[pos]:
                subs.append((pos, seq[pos]))
        result.append(tuple(subs))
    return result


def _build_consensus(seqs: Sequence[str], variable_positions: list[int]) -> str:
    """Build a consensus sequence: most common residue at each variable position,
    invariant positions held at their fixed value."""
    if not seqs:
        raise ValueError("No sequences to build consensus from")
    ref = list(seqs[0])
    for pos in variable_positions:
        counts = Counter(s[pos] for s in seqs)
        ref[pos] = counts.most_common(1)[0][0]
    return "".join(ref)


# ---------------------------------------------------------------------------
# MutantLibrary
# ---------------------------------------------------------------------------

@dataclass
class MutantLibrary:
    name: str
    reference_seq: str
    variable_positions: list[int]
    alphabet_at: dict[int, list[str]]
    chain: str
    frame: pd.DataFrame
    substitutions: list[tuple[tuple[int, str], ...]]


def build_library(
    name: str,
    cache_dir: Path,
    chain: str = "H",
    reference: str | None = None,
) -> MutantLibrary:
    """Load an affinity CSV and build a MutantLibrary.

    Parameters
    ----------
    name : str
        One of the 17 affinity dataset names.
    cache_dir : Path
        Directory to cache downloaded files.
    chain : str
        Which chain varies: "H" (heavy_chain_seq) or "L" (light_chain_seq).
    reference : str or None
        If None, the consensus sequence is used as reference.
    """
    cache_dir = Path(cache_dir)
    df = load_affinity(name, cache_dir)

    if chain == "H":
        col = "heavy_chain_seq"
    elif chain == "L":
        col = "light_chain_seq"
    else:
        raise ValueError(
            f"build_library chain must be 'H' or 'L', got {chain!r}. "
            f"For non-antibody chains, use read_pdb_chains() directly."
        )
    seqs = df[col].tolist()

    var_pos = find_variable_positions(seqs)

    if reference is None:
        reference = _build_consensus(seqs, var_pos)

    alphabet_at: dict[int, list[str]] = {}
    for pos in var_pos:
        alphabet_at[pos] = sorted({s[pos] for s in seqs})

    subs = _derive_substitutions(seqs, reference, var_pos)

    df = df.copy()
    df["n_mut"] = [len(s) for s in subs]

    return MutantLibrary(
        name=name,
        reference_seq=reference,
        variable_positions=var_pos,
        alphabet_at=alphabet_at,
        chain=chain,
        frame=df,
        substitutions=subs,
    )


# ---------------------------------------------------------------------------
# Structure-reference alignment
# ---------------------------------------------------------------------------

def resolve_chain_subset(
    struct_chains: dict[str, str],
    subset_arg: str | None,
    mutated_chain: str,
) -> tuple[dict[str, str], str]:
    """Filter *struct_chains* to the requested subset.

    Returns ``(filtered_chains, subset_label)`` where *subset_label* is the
    concatenation of chain IDs (e.g. ``"HLA"``) or ``"all"`` when no subset was
    requested.  The returned dict preserves the PDB's original chain order.
    """
    if subset_arg is None:
        return dict(struct_chains), "all"

    requested = [c.strip() for c in subset_arg.split(",") if c.strip()]
    if not requested:
        raise SystemExit("--chain-subset is empty after parsing.")

    if len(requested) != len(set(requested)):
        raise SystemExit(
            f"--chain-subset contains duplicates: {requested}"
        )

    available = list(struct_chains)
    unknown = [c for c in requested if c not in struct_chains]
    if unknown:
        raise SystemExit(
            f"Unknown chain(s) {unknown} in --chain-subset. "
            f"Available chains in PDB: {available}"
        )

    if mutated_chain not in requested:
        raise SystemExit(
            f"--chain {mutated_chain} (the mutated chain) must be included "
            f"in --chain-subset {requested}."
        )

    requested_set = set(requested)
    filtered = {c: s for c, s in struct_chains.items() if c in requested_set}
    label = "".join(filtered)
    return filtered, label


def align_reference_to_structure(
    reference_seq: str, chain_seq: str
) -> dict:
    """Compare a library reference sequence against a PDB chain sequence.

    Returns {"same_length": bool, "offset": int|None,
             "mismatches": [(index, structure_aa, reference_aa), ...]}.

    If lengths differ, tries sliding offsets looking for >95% identity.
    Raises NotImplementedError if no good offset is found.
    """
    if len(reference_seq) == len(chain_seq):
        mismatches = [
            (i, chain_seq[i], reference_seq[i])
            for i in range(len(reference_seq))
            if chain_seq[i] != reference_seq[i]
        ]
        return {"same_length": True, "offset": 0, "mismatches": mismatches}

    # Try sliding the shorter sequence along the longer one
    ref_len = len(reference_seq)
    chain_len = len(chain_seq)

    best_offset = None
    best_identity = 0.0
    best_mismatches = []

    if ref_len <= chain_len:
        for offset in range(chain_len - ref_len + 1):
            matches = sum(
                reference_seq[i] == chain_seq[offset + i] for i in range(ref_len)
            )
            identity = matches / ref_len
            if identity > best_identity:
                best_identity = identity
                best_offset = offset
                best_mismatches = [
                    (i, chain_seq[offset + i], reference_seq[i])
                    for i in range(ref_len)
                    if chain_seq[offset + i] != reference_seq[i]
                ]
    else:
        for offset in range(ref_len - chain_len + 1):
            matches = sum(
                chain_seq[i] == reference_seq[offset + i] for i in range(chain_len)
            )
            identity = matches / chain_len
            if identity > best_identity:
                best_identity = identity
                best_offset = -offset
                best_mismatches = [
                    (i, chain_seq[i], reference_seq[offset + i])
                    for i in range(chain_len)
                    if chain_seq[i] != reference_seq[offset + i]
                ]

    if best_identity < 0.95:
        raise NotImplementedError(
            f"No offset gives >95% identity (best: {best_identity:.1%} at offset {best_offset}). "
            f"Lengths: reference={ref_len}, chain={chain_len}. "
            "Full sequence alignment is not implemented."
        )

    return {"same_length": False, "offset": best_offset, "mismatches": best_mismatches}
