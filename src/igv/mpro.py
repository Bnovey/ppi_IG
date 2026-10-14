"""SARS-CoV-2 Mpro + nirmatrelvir ground truth -- Hu et al. 2023.

Source: Hu Y et al., ACS Central Science 2023.
doi:10.1021/acscentsci.3c00538, PMID 37637734, PMC10451032.

Parses Table S1 from the supplement PDF (hu2023_mpro_si_001.pdf) via
``pdftotext -layout``.  Table S1 contains enzymatic characterization and
drug inhibition data for SARS-CoV-2 Mpro mutants against GC-376,
PF-00835231, and nirmatrelvir (PF-07321332).

Structure: 8DZ2 (dimer, chains A+B, 302 residues each, 0 gaps).
Ligand: 4WI (nirmatrelvir, 35 heavy atoms), bound in both active sites.
Must be modelled as the dimer: E166 and H172 make direct contact with the
partner protomer at 2.6 and 3.2 A respectively.

Sign convention: higher Ki = weaker binding = resistance.
Fold-change is Ki_mut / Ki_wt (or IC50_mut / IC50_wt), so values > 1
indicate resistance.  This matches the SKEMPI/EGFR convention used
elsewhere in this repo (positive = resistance).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MPRO_PDF = "hu2023_mpro_si_001.pdf"
MPRO_CSV = "mpro_ki.csv"
MPRO_PROVENANCE = "mpro_ki.csv.prov.json"

MPRO_PDF_SHA256 = (
    "05c430b1de90877aab6de74ab2bf4d328c1d39f1f61453ebea15ceeb7813fc9c"
)

WT_NIRMATRELVIR_KI = 1.88
WT_NIRMATRELVIR_KI_ERR = 0.33
WT_NIRMATRELVIR_IC50 = 26.03
WT_NIRMATRELVIR_IC50_ERR = 1.65

PARSER_VERSION = "1.0.0"

ENZYMATICALLY_INACTIVE = {"H41M", "H41T", "H41Y", "H163W"}

# Column-position thresholds for compound classification.
# Determined from pdftotext -layout output across all pages:
#   GC-376 IC50/Ki:   positions 40--54
#   PF-00835231:      positions 65--80
#   Nirmatrelvir:     positions 90--105
_GC376_CEIL = 58
_NIRM_FLOOR = 86

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_IC50_RE = re.compile(r"IC50\s*=\s*([\d,.]+)\s*±\s*([\d,.]+)")
_IC50_CENS_RE = re.compile(r"IC50\s*>\s*=?\s*([\d,.]+)\s*(nM|µM|uM)")
_KI_RE = re.compile(r"Ki\s*=\s*([\d,.]+)\s*±\s*([\d,.]+)")
_KI_CENS_RE = re.compile(r"Ki\s*>\s*=?\s*([\d,.]+)\s*(nM|µM|uM)")

_KM_RE = re.compile(r"Km\s*=\s*([\d,.]+)\s*±\s*([\d,.]+)\s*[µu]M")
_VMAX_RE = re.compile(r"Vmax\s*=\s*([\d,.]+)\s*±\s*([\d,.]+)")
_KCAT_RE = re.compile(r"(?<![/\w])kcat\s*=\s*([\d,.]+)\s*S-1(?!M)")
_KCAT_KM_RE = re.compile(r"kcat/Km\s*=\s*([\d,.]+)\s*S-1M-1")

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# ---------------------------------------------------------------------------
# PDF verification and text extraction
# ---------------------------------------------------------------------------


def verify_pdf(pdf_path: Path) -> None:
    sha = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    if sha != MPRO_PDF_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {pdf_path}:\n"
            f"  expected: {MPRO_PDF_SHA256}\n"
            f"  got:      {sha}"
        )


def extract_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    result.check_returncode()
    return result.stdout


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _pnum(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        log.warning("Malformed number in PDF: %r", s)
        return None


def _classify_compound(pos: int) -> str:
    if pos < _GC376_CEIL:
        return "gc376"
    if pos >= _NIRM_FLOOR:
        return "nirmatrelvir"
    return "pf00835231"


def _extract_drug_values(block_lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for line in block_lines:
        for m in _IC50_RE.finditer(line):
            cpd = _classify_compound(m.start())
            k = f"{cpd}_ic50"
            if k not in out:
                out[k] = _pnum(m.group(1))
                out[f"{k}_err"] = _pnum(m.group(2))

        for m in _IC50_CENS_RE.finditer(line):
            cpd = _classify_compound(m.start())
            k = f"{cpd}_ic50"
            if k not in out:
                val = _pnum(m.group(1))
                if m.group(2) in ("µM", "uM"):
                    val *= 1000
                out[k] = val
                out[f"{k}_err"] = None
                out[f"{k}_censored"] = True

        for m in _KI_RE.finditer(line):
            cpd = _classify_compound(m.start())
            k = f"{cpd}_ki"
            if k not in out:
                out[k] = _pnum(m.group(1))
                out[f"{k}_err"] = _pnum(m.group(2))

        for m in _KI_CENS_RE.finditer(line):
            cpd = _classify_compound(m.start())
            k = f"{cpd}_ki"
            if k not in out:
                val = _pnum(m.group(1))
                if m.group(2) in ("µM", "uM"):
                    val *= 1000
                out[k] = val
                out[f"{k}_err"] = None
                out[f"{k}_censored"] = True

    return out


def _extract_kinetics(block_lines: list[str]) -> dict[str, float | None]:
    text = "\n".join(block_lines)
    out: dict[str, float | None] = {}

    m = _KM_RE.search(text)
    if m:
        out["km"] = _pnum(m.group(1))
        out["km_err"] = _pnum(m.group(2))

    m = _VMAX_RE.search(text)
    if m:
        out["vmax"] = _pnum(m.group(1))
        out["vmax_err"] = _pnum(m.group(2))

    m = _KCAT_RE.search(text)
    if m:
        out["kcat"] = _pnum(m.group(1))

    m = _KCAT_KM_RE.search(text)
    if m:
        out["kcat_over_km"] = _pnum(m.group(1))

    return out


# ---------------------------------------------------------------------------
# Anchor detection
# ---------------------------------------------------------------------------


def _find_anchors(table_lines: list[str]) -> list[tuple[int, dict[str, Any]]]:
    """Identify the start line and metadata for each mutant entry."""
    anchors: list[tuple[int, dict[str, Any]]] = []
    skip_second_part = False
    i = 0

    while i < len(table_lines):
        line = table_lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Page headers like S12, S13, ...
        if re.match(r"^S\d{1,2}$", stripped):
            i += 1
            continue

        if skip_second_part:
            if re.match(r"^[A-Z]\d+[A-Z]\b", stripped):
                skip_second_part = False
            i += 1
            continue

        # WT
        if re.match(r"^WT\b", stripped):
            anchors.append((i, {
                "mutation_id": "WT", "variant_type": "wt", "occurrence": None,
            }))
            i += 1
            continue

        # Double mutant prefix: "L50F/" or "Q189E/"
        dm = re.match(r"^([A-Z]\d+[A-Z])/", stripped)
        if dm:
            part1 = dm.group(1)
            part2 = None
            occ = None
            for j in range(i + 1, min(i + 5, len(table_lines))):
                s2 = table_lines[j].strip()
                m2 = re.match(r"^([A-Z]\d+[A-Z])\b", s2)
                if m2:
                    part2 = m2.group(1)
                    break
                om = re.match(r"^(\d[\d,]*)(?:[^.\d]|$)", s2)
                if om and occ is None:
                    occ = int(om.group(1).replace(",", ""))
            if part2:
                anchors.append((i, {
                    "mutation_id": f"{part1}/{part2}",
                    "variant_type": "double",
                    "occurrence": occ,
                }))
                skip_second_part = True
            i += 1
            continue

        # Single substitution: "H41M    84" or "H41M   84   Enzymatically inactive"
        sm = re.match(r"^([A-Z]\d+[A-Z])\s+([\d,]+)", stripped)
        if sm:
            mut_id = sm.group(1)
            occ = int(sm.group(2).replace(",", ""))
            status = "inactive" if "Enzymatically inactive" in line else "active"
            anchors.append((i, {
                "mutation_id": mut_id,
                "variant_type": "substitution",
                "occurrence": occ,
                "status": status,
            }))
            i += 1
            continue

        # Deletion prefix: "M49" not followed by a letter (position-only)
        delm = re.match(r"^([A-Z])(\d+)(?![A-Z\d])", stripped)
        if delm:
            is_del = any(
                re.search(r"\bdeletion\b", table_lines[j], re.I)
                for j in range(i, min(i + 5, len(table_lines)))
            )
            if is_del:
                pos_name = delm.group(1) + delm.group(2)
                occ = None
                for j in range(i + 1, min(i + 3, len(table_lines))):
                    om = re.match(
                        r"^(\d[\d,]*)(?:[^.\d]|$)", table_lines[j].strip()
                    )
                    if om:
                        occ = int(om.group(1).replace(",", ""))
                        break
                anchors.append((i, {
                    "mutation_id": f"{pos_name} deletion",
                    "variant_type": "deletion",
                    "occurrence": occ,
                }))
                i += 1
                continue

        i += 1

    return anchors


# ---------------------------------------------------------------------------
# Block assembly
# ---------------------------------------------------------------------------


def _assemble_blocks(
    table_lines: list[str],
    anchors: list[tuple[int, dict[str, Any]]],
) -> list[tuple[dict[str, Any], list[str]]]:
    """Split table lines into per-mutant blocks and adjust boundaries.

    Kinetic data (Km, Vmax) for mutant N appears *before* its name line
    but *after* the previous mutant's kcat/Km line.  This function moves
    trailing lines (after the logical block end) from block N to block N+1
    so each block contains exactly one mutant's complete data.
    """
    if not anchors:
        return []

    # Raw split: include lines before first anchor in the first block
    raw: list[tuple[dict[str, Any], list[str]]] = []
    for idx, (anchor_line, info) in enumerate(anchors):
        if idx + 1 < len(anchors):
            end = anchors[idx + 1][0]
        else:
            end = len(table_lines)
        block = table_lines[anchor_line:end]
        if idx == 0:
            block = table_lines[:anchor_line] + block
        raw.append((info, block))

    # Move trailing lines from block N to block N+1
    for idx in range(len(raw) - 1):
        info, lines = raw[idx]

        logical_end = len(lines)
        if info.get("status") == "inactive":
            for j, ln in enumerate(lines):
                if "Enzymatically inactive" in ln:
                    logical_end = j + 1
                    break
        else:
            for j in range(len(lines) - 1, -1, -1):
                if _KCAT_KM_RE.search(lines[j]):
                    logical_end = j + 1
                    break

        content = lines[:logical_end]
        trailing = lines[logical_end:]
        raw[idx] = (info, content)
        if trailing:
            nxt_info, nxt_lines = raw[idx + 1]
            raw[idx + 1] = (nxt_info, trailing + nxt_lines)

    return raw


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_table_s1(text: str) -> list[dict[str, Any]]:
    """Parse Table S1 from pdftotext -layout output.

    Returns a list of dicts, one per table entry (WT + all mutants).
    """
    lines = text.split("\n")

    start = end = None
    for i, ln in enumerate(lines):
        if "Resistant mutants identified" in ln and start is None:
            start = i
        if start is not None and ln.strip().startswith("aThe occurrence"):
            end = i
            break

    if start is None or end is None:
        raise ValueError("Could not locate Table S1 boundaries in text")

    table = lines[start:end]
    anchors = _find_anchors(table)
    blocks = _assemble_blocks(table, anchors)

    entries: list[dict[str, Any]] = []
    for info, block_lines in blocks:
        entry: dict[str, Any] = dict(info)

        # Parse mutation components
        mut_id = entry["mutation_id"]
        vtype = entry["variant_type"]

        if vtype == "substitution":
            m = _MUT_RE.match(mut_id)
            entry["wt_aa"] = m.group(1)
            entry["position"] = int(m.group(2))
            entry["mut_aa"] = m.group(3)
        elif vtype == "deletion":
            dm = re.match(r"^([A-Z])(\d+) deletion$", mut_id)
            entry["wt_aa"] = dm.group(1)
            entry["position"] = int(dm.group(2))
            entry["mut_aa"] = None
        else:
            entry["wt_aa"] = None
            entry["position"] = None
            entry["mut_aa"] = None

        if "status" not in entry:
            entry["status"] = "active"

        # Extract data
        if entry["status"] != "inactive":
            entry.update(_extract_kinetics(block_lines))
            entry.update(_extract_drug_values(block_lines))

        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# CSV building
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "mutation_id", "position", "wt_aa", "mut_aa",
    "variant_type", "status", "occurrence",
    "km", "km_err", "vmax", "vmax_err", "kcat", "kcat_over_km",
    "gc376_ic50", "gc376_ic50_err", "gc376_ki", "gc376_ki_err",
    "pf00835231_ic50", "pf00835231_ic50_err", "pf00835231_ki", "pf00835231_ki_err",
    "nirmatrelvir_ic50", "nirmatrelvir_ic50_err", "nirmatrelvir_ki", "nirmatrelvir_ki_err",
    "nirmatrelvir_ic50_fold_change", "nirmatrelvir_ki_fold_change",
]


def build_mpro_csv(
    pdf_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Parse the Hu 2023 supplement PDF and write mpro_ki.csv + provenance."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verify_pdf(pdf_path)
    text = extract_text(pdf_path)
    entries = parse_table_s1(text)

    # Look up WT values for fold-change computation
    wt = next(e for e in entries if e["mutation_id"] == "WT")
    wt_ki = wt.get("nirmatrelvir_ki")
    wt_ic50 = wt.get("nirmatrelvir_ic50")

    assert wt_ki is not None and abs(wt_ki - WT_NIRMATRELVIR_KI) < 0.01, (
        f"WT nirmatrelvir Ki mismatch: expected {WT_NIRMATRELVIR_KI}, "
        f"got {wt_ki}"
    )

    n_subs = sum(1 for e in entries if e["variant_type"] == "substitution")
    sub_positions = {
        e["position"]
        for e in entries
        if e["variant_type"] == "substitution" and e["position"] is not None
    }
    assert n_subs == 102, f"Expected 102 substitutions, got {n_subs}"
    assert len(sub_positions) == 12, (
        f"Expected 12 positions, got {len(sub_positions)}: {sorted(sub_positions)}"
    )

    for e in entries:
        ki = e.get("nirmatrelvir_ki")
        ic50 = e.get("nirmatrelvir_ic50")
        if ki is not None and wt_ki and not e.get("nirmatrelvir_ki_censored"):
            e["nirmatrelvir_ki_fold_change"] = ki / wt_ki
        if ic50 is not None and wt_ic50 and not e.get("nirmatrelvir_ic50_censored"):
            e["nirmatrelvir_ic50_fold_change"] = ic50 / wt_ic50

    df = pd.DataFrame(entries)
    for col in _CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_CSV_COLUMNS]

    csv_path = output_dir / MPRO_CSV
    df.to_csv(csv_path, index=False, float_format="%.4f")
    log.info("Wrote %d rows to %s", len(df), csv_path)

    _write_provenance(csv_path, pdf_path, len(df))
    return csv_path


def _write_provenance(csv_path: Path, pdf_path: Path, n_rows: int) -> None:
    prov = {
        "source_doi": "10.1021/acscentsci.3c00538",
        "source_pmid": "37637734",
        "source_pmc": "PMC10451032",
        "pdf_file": str(pdf_path),
        "pdf_sha256": MPRO_PDF_SHA256,
        "parser_version": PARSER_VERSION,
        "extraction_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "sign_convention": (
            "higher Ki = weaker binding = resistance; "
            "fold_change = Ki_mut / Ki_wt, values > 1 = resistance; "
            "matches SKEMPI/EGFR convention (positive = resistance)"
        ),
        "compounds": ["GC-376", "PF-00835231", "nirmatrelvir (PF-07321332)"],
        "structure": "8DZ2 (dimer, chains A+B)",
        "ligand": "4WI (nirmatrelvir, 35 heavy atoms)",
    }
    prov_path = csv_path.parent / MPRO_PROVENANCE
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_mpro_ground_truth(csv_path: str | Path) -> pd.DataFrame:
    """Load Mpro ground truth CSV and compute log10 fold changes.

    Filters to active substitutions with a measured (non-censored)
    nirmatrelvir Ki.  Returns a DataFrame with ``response_log10 =
    log10(Ki_mut / Ki_wt)``, positive = resistance.
    """
    df = pd.read_csv(csv_path)

    mask = (
        (df["variant_type"] == "substitution")
        & (df["status"] == "active")
        & df["nirmatrelvir_ki"].notna()
        & ~df.get("nirmatrelvir_ki_censored", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    )
    gt = df[mask].copy()

    gt["response_log10"] = gt["nirmatrelvir_ki_fold_change"].apply(
        lambda fc: math.log10(fc) if fc and fc > 0 else float("nan")
    )

    return gt.reset_index(drop=True)


# ---------------------------------------------------------------------------
# WT-residue verification against PDB
# ---------------------------------------------------------------------------


def verify_wt_residues(
    mutations: list[str],
    pdb_path: Path,
    chain: str = "A",
) -> tuple[int, int, list[str]]:
    """Verify WT residues of mutated positions against a PDB structure."""
    from igv.skempi import read_pdb_residue_ids

    ids, seqs = read_pdb_residue_ids(pdb_path)
    seq = seqs[chain]
    resids = ids[chain]
    id_to_idx = {rid: i for i, rid in enumerate(resids)}

    n_match = 0
    n_total = 0
    mismatches: list[str] = []

    for m_str in mutations:
        m = _MUT_RE.match(m_str)
        if m is None:
            continue
        wt_aa, resnum = m.group(1), m.group(2)
        n_total += 1

        if resnum not in id_to_idx:
            mismatches.append(f"{m_str}: resnum {resnum} absent from structure")
            continue

        idx = id_to_idx[resnum]
        pdb_aa = seq[idx]
        if pdb_aa == wt_aa:
            n_match += 1
        else:
            mismatches.append(
                f"{m_str}: expected {wt_aa} at resnum {resnum}, got {pdb_aa}"
            )

    return n_match, n_total, mismatches
