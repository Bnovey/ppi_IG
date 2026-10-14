"""EGFR kinase domain resistance data -- Robichaux et al. 2021.

Source: Robichaux JP et al., "Structure-based classification predicts drug
response in EGFR-mutant NSCLC", Nature 597:732-737, 2021.
doi:10.1038/s41586-021-03898-1, PMID 34526717, PMC8481125.

Data: Supplementary Table 4, Panel A -- 77 mutation rows x 59 columns.
18 TKIs x 3 replicates per mutation-drug pair, zero missing values in the
27 single-missense subset.

Response variable: log(IC50_mutant / IC50_WT), already a ratio. Positive =
resistance. The supplement does not state the log base explicitly.
Recorded as ``log_base="unknown"`` in provenance.

Numbering: Robichaux uses precursor/clinical numbering. PDB 1M17 uses
mature-protein numbering at an offset of exactly -24. All 20 measured
positions give 20/20 WT-residue agreement at this offset.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from igv.data import _atomic_download

log = logging.getLogger(__name__)

ROBICHAUX_XLSX_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41586-021-03898-1/MediaObjects/"
    "41586_2021_3898_MOESM4_ESM.xlsx"
)

ROBICHAUX_CACHE_NAME = "robichaux2021_supp_table4.xlsx"

_SINGLE_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

EGFR_1M17_OFFSET = -24

EGFR_DRUGS = (
    "Erlotinib", "Gefitinib", "AZD3759", "Sapatinib", "Afatinib",
    "Dacomitinib", "Neratinib", "Poziotinib", "Tarlox-TKI", "CLN-081",
    "AZ5104", "Mobocertinib", "Osimertinib", "Nazartinib", "Olmutinib",
    "Rociletinib", "Naquotinib", "Lazertinib",
)

SIGN_ANCHORS: dict[tuple[str, str], tuple[str, float]] = {
    ("T790M", "Erlotinib"): ("positive", 1.56),
    ("T790M", "Osimertinib"): ("negative", -0.76),
    ("L858R", "Erlotinib"): ("negative", -1.17),
    ("L718Q", "Osimertinib"): ("positive", 1.03),
}


@dataclass(frozen=True)
class EGFRMutation:
    wt_aa: str
    resnum: int
    mut_aa: str

    def __str__(self) -> str:
        return f"{self.wt_aa}{self.resnum}{self.mut_aa}"

    @property
    def pdb_resnum_1m17(self) -> int:
        return self.resnum + EGFR_1M17_OFFSET


def parse_egfr_mutation(s: str) -> EGFRMutation:
    m = _SINGLE_MUTATION_RE.match(s.strip())
    if m is None:
        raise ValueError(f"Malformed EGFR mutation code: {s!r}")
    return EGFRMutation(wt_aa=m.group(1), resnum=int(m.group(2)), mut_aa=m.group(3))


def download_robichaux2021(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    dest = cache_dir / ROBICHAUX_CACHE_NAME
    if dest.exists():
        return dest
    _atomic_download(ROBICHAUX_XLSX_URL, dest)
    return dest


def parse_panel_a(xlsx_path: Path) -> pd.DataFrame:
    """Parse Panel A of Robichaux 2021 Supplementary Table 4.

    Returns a tidy DataFrame with columns: mutation, drug, rep1, rep2, rep3,
    mean_response, replicate_sd, position, wt_aa, mut_aa.

    Only single missense substitutions are kept (27 of 77 rows).
    """
    df_raw = pd.read_excel(xlsx_path, sheet_name="Panel A", engine="openpyxl")

    mut_col = "EGFR Mutation"
    if mut_col not in df_raw.columns:
        raise ValueError(f"Expected column '{mut_col}' not found in Panel A")

    mut_col_idx = list(df_raw.columns).index(mut_col)
    drug_columns = list(df_raw.columns[mut_col_idx + 1:])

    drugs: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(drug_columns):
        base = re.sub(r"\.\d+$", "", str(drug_columns[i]))
        rep_cols = [drug_columns[i]]
        j = i + 1
        while j < len(drug_columns) and re.sub(r"\.\d+$", "", str(drug_columns[j])) == base:
            rep_cols.append(drug_columns[j])
            j += 1
        drugs.append((base, rep_cols))
        i = j

    rows: list[dict] = []
    for _, row in df_raw.iterrows():
        mut_str = str(row[mut_col]).strip()
        m = _SINGLE_MUTATION_RE.match(mut_str)
        if m is None:
            continue
        mut = parse_egfr_mutation(mut_str)

        for drug_name, rep_cols in drugs:
            reps = []
            for rc in rep_cols[:3]:
                val = row[rc]
                try:
                    reps.append(float(val))
                except (ValueError, TypeError):
                    reps.append(np.nan)
            while len(reps) < 3:
                reps.append(np.nan)

            finite = [r for r in reps if np.isfinite(r)]
            mean_val = float(np.mean(finite)) if finite else np.nan
            sd_val = float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan

            rows.append({
                "mutation": str(mut),
                "drug": drug_name,
                "rep1": reps[0],
                "rep2": reps[1],
                "rep3": reps[2],
                "mean_response": mean_val,
                "replicate_sd": sd_val,
                "position": mut.resnum,
                "wt_aa": mut.wt_aa,
                "mut_aa": mut.mut_aa,
            })

    return pd.DataFrame(rows)


def load_egfr_ground_truth(cache_dir: Path) -> pd.DataFrame:
    xlsx_path = download_robichaux2021(cache_dir)
    return parse_panel_a(xlsx_path)


def verify_wt_residues_1m17(
    mutations: list[str],
    pdb_path: Path,
    chain: str = "A",
) -> tuple[int, int, list[str]]:
    """Verify WT residues of measured EGFR positions against 1M17.

    Applies the -24 offset. Returns (n_match, n_total, mismatches).
    """
    from igv.skempi import read_pdb_residue_ids

    ids, seqs = read_pdb_residue_ids(pdb_path)
    seq = seqs[chain]
    resids = ids[chain]
    id_to_idx = {rid: i for i, rid in enumerate(resids)}

    n_match = 0
    n_total = 0
    mismatches: list[str] = []

    for m_str in mutations:
        mut = parse_egfr_mutation(m_str)
        pdb_resnum = str(mut.pdb_resnum_1m17)
        n_total += 1

        if pdb_resnum not in id_to_idx:
            mismatches.append(f"{m_str}: resnum {pdb_resnum} absent from structure")
            continue

        idx = id_to_idx[pdb_resnum]
        pdb_aa = seq[idx]
        if pdb_aa == mut.wt_aa:
            n_match += 1
        else:
            mismatches.append(
                f"{m_str}: expected {mut.wt_aa} at PDB resnum {pdb_resnum}, "
                f"got {pdb_aa}"
            )

    return n_match, n_total, mismatches


def verify_sign_convention(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check sign anchors: T790M erlotinib positive, osimertinib negative, etc."""
    problems: list[str] = []

    for (mut_name, drug_name), (expected_sign, expected_val) in SIGN_ANCHORS.items():
        rows = df[(df["mutation"] == mut_name) & (df["drug"] == drug_name)]
        if len(rows) == 0:
            problems.append(f"Anchor {mut_name}/{drug_name} not found in data")
            continue
        val = float(rows.iloc[0]["mean_response"])
        if expected_sign == "positive" and val <= 0:
            problems.append(
                f"{mut_name}/{drug_name}: expected positive (resistance), got {val:.4f}"
            )
        elif expected_sign == "negative" and val >= 0:
            problems.append(
                f"{mut_name}/{drug_name}: expected negative (sensitivity), got {val:.4f}"
            )
        if abs(val - expected_val) > 0.5:
            problems.append(
                f"{mut_name}/{drug_name}: value {val:.4f} differs from "
                f"expected {expected_val:.2f} by more than 0.5 log units"
            )

    return len(problems) == 0, problems
