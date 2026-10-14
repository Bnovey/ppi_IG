"""Platinum database loader for protein-ligand affinity ground truth.

Source: Pires, Blundell & Ascher, Nucleic Acids Research 43:D387-D391, 2015.
doi:10.1093/nar/gku966, PMC4384026.

Bulk download: https://biosig.lab.uq.edu.au/platinum/static/platinum_flat_file.csv

The flat file carries 1008 data rows with per-mutation Kd/Ki/Km values,
wild-type and mutant affinities, distances to ligand, and RSA. All
measurements are biochemical (ITC, enzyme kinetics, fluorescence, SPR).

Used here for PimA mannosyltransferase (C. glutamicum, UniProt Q8NNK8)
with ligands GDD (3OKP) and GDP (3OKC).
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from igv.data import _atomic_download

log = logging.getLogger(__name__)

PLATINUM_CSV_URL = (
    "https://biosig.lab.uq.edu.au/platinum/static/platinum_flat_file.csv"
)

PLATINUM_CACHE_NAME = "platinum_flat_file.csv"

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class PlatinumMutation:
    wt_aa: str
    resnum: int
    mut_aa: str

    def __str__(self) -> str:
        return f"{self.wt_aa}{self.resnum}{self.mut_aa}"


def parse_mutation(s: str) -> PlatinumMutation:
    m = _MUTATION_RE.match(s.strip())
    if m is None:
        raise ValueError(f"Malformed mutation code: {s!r}")
    return PlatinumMutation(wt_aa=m.group(1), resnum=int(m.group(2)), mut_aa=m.group(3))


def download_platinum(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    dest = cache_dir / PLATINUM_CACHE_NAME
    if dest.exists():
        return dest
    _atomic_download(PLATINUM_CSV_URL, dest)
    return dest


def load_platinum(cache_dir: Path) -> list[dict[str, str]]:
    path = download_platinum(cache_dir)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def extract_target(
    all_rows: list[dict[str, str]],
    uniprot: str,
    lig_id: str,
) -> pd.DataFrame:
    """Filter Platinum rows to single-point mutations for a specific target/ligand.

    Returns a DataFrame with columns: mutation, k_wt, k_mt, response_log10,
    censored, unit, distance_to_ligand, position, wt_aa, mut_aa, wt_pdb.
    """
    filtered = [
        r for r in all_rows
        if r["mut.is_single_point"] == "YES"
        and r["mut.uniprot"] == uniprot
        and r["affin.lig_id"] == lig_id
    ]

    rows: list[dict] = []
    for r in filtered:
        mut_str = r["mutation"]
        m = _MUTATION_RE.match(mut_str)
        if m is None:
            log.warning("Skipping non-parseable mutation: %s", mut_str)
            continue

        k_wt_str = r["affin.k_wt"]
        k_mt_str = r["affin.k_mt"]

        censored = False
        if k_mt_str.startswith(">"):
            censored = True
            k_mt_str = k_mt_str[1:]

        try:
            k_wt = float(k_wt_str)
        except (ValueError, TypeError):
            log.warning("Non-numeric k_wt for %s: %s", mut_str, k_wt_str)
            continue

        try:
            k_mt = float(k_mt_str)
        except (ValueError, TypeError):
            log.warning("Non-numeric k_mt for %s: %s", mut_str, k_mt_str)
            continue

        if k_wt <= 0 or k_mt <= 0:
            log.warning("Non-positive affinity for %s: k_wt=%s, k_mt=%s", mut_str, k_wt, k_mt)
            continue

        response = math.log10(k_mt) - math.log10(k_wt)

        dist_str = r.get("mut.distance_to_lig", "")
        try:
            dist = float(dist_str)
        except (ValueError, TypeError):
            dist = np.nan

        rows.append({
            "mutation": mut_str,
            "k_wt": k_wt,
            "k_mt": k_mt,
            "response_log10": response,
            "censored": censored,
            "unit": r["affin.unit"],
            "distance_to_ligand": dist,
            "position": int(m.group(2)),
            "wt_aa": m.group(1),
            "mut_aa": m.group(3),
            "wt_pdb": r["mut.wt_pdb"],
        })

    return pd.DataFrame(rows)


def verify_wt_residues(
    df: pd.DataFrame,
    pdb_path: Path,
    chain: str = "A",
) -> tuple[int, int, list[str]]:
    """Verify WT residues of measured positions against a PDB structure."""
    from igv.skempi import read_pdb_residue_ids

    ids, seqs = read_pdb_residue_ids(pdb_path)
    seq = seqs[chain]
    resids = ids[chain]
    id_to_idx = {rid: i for i, rid in enumerate(resids)}

    n_match = 0
    n_total = 0
    mismatches: list[str] = []

    for _, row in df.iterrows():
        pdb_resnum = str(row["position"])
        expected_aa = row["wt_aa"]
        n_total += 1

        if pdb_resnum not in id_to_idx:
            mismatches.append(f"{row['mutation']}: resnum {pdb_resnum} absent")
            continue

        idx = id_to_idx[pdb_resnum]
        pdb_aa = seq[idx]
        if pdb_aa == expected_aa:
            n_match += 1
        else:
            mismatches.append(
                f"{row['mutation']}: expected {expected_aa}, got {pdb_aa}"
            )

    return n_match, n_total, mismatches
