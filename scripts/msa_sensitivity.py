#!/usr/bin/env python3
"""Measure how much the ColabFold MSA profile changes when a single residue is mutated.

Tests whether the per-mutant MSA re-query in stage 04 introduces signal or
noise relative to pinning the WT MSA (as stage 02 does).

Profile definition (fixed across all comparisons):
    For each aligned column, compute the frequency of each of the 20 standard
    amino acids plus gap, across all non-query rows.  Result is an (L, 21)
    matrix with rows summing to 1.

Limitations:
  - Single-chain MSA only. MSA pairing across chains (which Boltz-2 performs
    when both chains are passed) is out of scope.
  - Uses mode="all" which returns only UniRef hits (72 homologs). mode="env"
    would additionally return ~137 BFD/Mgnify/MetaEuk/SMAG environmental hits,
    but the existing repo code (skempi.py fetch_colabfold_msa) uses mode="all"
    so we match that for consistency.

Known bug in existing code: src/igv/skempi.py:824 does resp.text on the
/result/download/{id} response, which returns tar.gz binary -- this corrupts
the data via UTF-8 decoding of raw gzip bytes. Not fixed here (diagnostic
only); should be filed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import read_pdb_chains
from igv.skempi import read_pdb_residue_ids, skempi_positions

log = logging.getLogger(__name__)

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY-")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

_COLABFOLD_HOST = "https://api.colabfold.com"

MUTATIONS: list[tuple[int, str]] = [
    (30, "A"),   # E30A  -- near N-terminus of SKEMPI range
    (52, "A"),   # Y52A  -- early-mid
    (88, "A"),   # N88A  -- middle
    (111, "A"),  # W111A -- late-mid
    (149, "A"),  # W149A -- near C-terminus of SKEMPI range
]


def _fetch_from_api(sequence: str) -> str:
    """Submit sequence to ColabFold, poll, download, extract a3m text."""
    resp = requests.post(
        f"{_COLABFOLD_HOST}/ticket/msa",
        data={"q": f">query\n{sequence}", "mode": "all"},
        timeout=30,
    )
    resp.raise_for_status()
    ticket = resp.json()
    ticket_id = ticket["id"]
    status = ticket.get("status")
    log.info("  ticket %s, initial status: %s", ticket_id, status)

    for poll in range(120):
        if status == "COMPLETE":
            break
        if status in ("ERROR", "MAINTENANCE"):
            raise RuntimeError(f"ColabFold returned {status} for ticket {ticket_id}")
        if status == "RATELIMIT":
            log.warning("  rate-limited, backing off 30s")
            time.sleep(30)
        else:
            time.sleep(8)

        resp = requests.get(f"{_COLABFOLD_HOST}/ticket/{ticket_id}", timeout=30)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status")
        if poll % 5 == 4:
            log.info("  poll %d: %s", poll + 1, status)

    if status != "COMPLETE":
        raise RuntimeError(f"ColabFold did not complete (last status: {status})")

    resp = requests.get(
        f"{_COLABFOLD_HOST}/result/download/{ticket_id}",
        timeout=60,
    )
    resp.raise_for_status()

    tf = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
    a3m_text = None
    for member in tf.getmembers():
        if member.name.endswith(".a3m"):
            a3m_text = tf.extractfile(member).read().decode()
            break
    tf.close()

    if a3m_text is None:
        raise RuntimeError("No .a3m file found in ColabFold tar.gz response")

    return a3m_text.replace("\x00", "")


def _write_cache(text: str, path: Path) -> None:
    msa_dir = path.parent
    msa_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=msa_dir, suffix=".a3m")
    try:
        os.write(fd, text.encode())
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        os.close(fd)
        os.unlink(tmp)
        raise


def _cache_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return False
    return True


def fetch_msa(sequence: str, cache_dir: Path) -> Path:
    """Fetch MSA, caching by SHA-256 of query sequence."""
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()
    msa_dir = Path(cache_dir) / "msa"
    cached = msa_dir / f"{seq_hash}.a3m"
    if _cache_is_valid(cached):
        return cached

    a3m_text = _fetch_from_api(sequence)
    _write_cache(a3m_text, cached)
    return cached


def fetch_msa_uncached(sequence: str, cache_dir: Path, tag: str) -> Path:
    """Fetch MSA without using the sequence-hash cache. Used for WT replicates."""
    msa_dir = Path(cache_dir) / "msa"
    dest = msa_dir / f"replicate_{tag}.a3m"
    if _cache_is_valid(dest):
        return dest

    a3m_text = _fetch_from_api(sequence)
    _write_cache(a3m_text, dest)
    return dest


def parse_a3m_rows(a3m_path: Path, query_length: int) -> tuple[str, list[str], list[str]]:
    """Parse a3m and return (query_row, aligned_rows, header_lines).

    Strips NUL bytes, lowercase insertion characters, and validates query length.
    """
    headers: list[str] = []
    seqs: list[list[str]] = []

    with open(a3m_path) as fh:
        for line in fh:
            line = line.rstrip("\n").replace("\x00", "")
            if not line:
                continue
            if line.startswith(">"):
                headers.append(line)
                seqs.append([])
            elif seqs:
                seqs[-1].append(line)
            else:
                headers.append(">query")
                seqs.append([line])

    rows: list[str] = []
    for parts in seqs:
        raw = "".join(parts)
        cleaned = re.sub(r"[a-z]", "", raw)
        rows.append(cleaned)

    if not rows or len(rows[0]) != query_length:
        raise ValueError(
            f"Query row length {len(rows[0]) if rows else 0} != expected {query_length}"
        )

    return rows[0], rows, headers


def extract_accession(header: str) -> str:
    """Extract the accession (first field after '>') from an a3m header line."""
    return header.lstrip(">").split("\t")[0].split()[0]


def compute_profile(a3m_path: Path, query_length: int) -> tuple[np.ndarray, list[str]]:
    """Return (L, 21) frequency profile and list of non-query accessions."""
    _query_row, rows, headers = parse_a3m_rows(a3m_path, query_length)
    non_query = rows[1:]
    non_query_headers = headers[1:]

    accessions = [extract_accession(h) for h in non_query_headers]

    profile = np.zeros((query_length, 21), dtype=np.float64)
    for col in range(query_length):
        counts: Counter[int] = Counter()
        for row in non_query:
            ch = row[col] if col < len(row) else "-"
            idx = AA_TO_IDX.get(ch, AA_TO_IDX["-"])
            counts[idx] += 1
        total = sum(counts.values())
        if total > 0:
            for idx, cnt in counts.items():
                profile[col, idx] = cnt / total

    return profile, accessions


def compare_profiles(
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    accessions_a: list[str],
    accessions_b: list[str],
    *,
    mut_pos: int | None = None,
    L: int,
) -> dict:
    l1_per_pos = np.abs(profile_b - profile_a).sum(axis=1)
    mean_l1 = float(l1_per_pos.mean())

    ids_a = set(accessions_a)
    ids_b = set(accessions_b)
    shared = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a
    pct_shared = len(shared) / max(len(ids_a | ids_b), 1) * 100

    result: dict = {
        "n_homologs_a": len(accessions_a),
        "n_homologs_b": len(accessions_b),
        "total_l1_norm_per_pos": mean_l1,
        "homolog_set_shared": len(shared),
        "homolog_set_only_a": len(only_a),
        "homolog_set_only_b": len(only_b),
        "pct_shared": pct_shared,
    }

    if mut_pos is not None:
        l1_at_mut = float(l1_per_pos[mut_pos])
        other_mask = np.ones(L, dtype=bool)
        other_mask[mut_pos] = False
        mean_l1_other = float(l1_per_pos[other_mask].mean())
        result["l1_at_mutated_col"] = l1_at_mut
        result["mean_l1_other_cols"] = mean_l1_other
        result["ratio_mut_vs_other"] = (
            l1_at_mut / mean_l1_other if mean_l1_other > 0 else float("inf")
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default="data/raw",
        help="Directory for cached PDB and MSA files",
    )
    parser.add_argument(
        "--out", default="results/msa_sensitivity.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_path = cache_dir / "1jtg.pdb"
    if not pdb_path.exists():
        sys.exit(f"PDB file not found: {pdb_path}")

    chains = read_pdb_chains(pdb_path)
    wt_seq = chains["B"]
    L = len(wt_seq)
    log.info("1JTG chain B: %d residues", L)

    positions = skempi_positions("1JTG", "B", cache_dir)
    log.info("SKEMPI positions on chain B (%d): %s", len(positions), positions)

    for pos, mut_aa in MUTATIONS:
        assert pos in positions, f"Position {pos} not in SKEMPI positions"
        assert wt_seq[pos] != mut_aa, f"WT is already {mut_aa} at position {pos}"
    log.info(
        "Mutations: %s",
        ", ".join(f"{wt_seq[p]}{p}{m}" for p, m in MUTATIONS),
    )

    # -- Fetch WT MSA (baseline, cached) --
    log.info("Fetching WT MSA (baseline)...")
    wt_a3m = fetch_msa(wt_seq, cache_dir)
    log.info("WT MSA cached at %s", wt_a3m)
    wt_profile, wt_accessions = compute_profile(wt_a3m, L)
    log.info("WT: %d homologs", len(wt_accessions))

    # -- Fetch WT replicates (bypass cache to measure server nondeterminism) --
    wt_replicate_profiles = []
    wt_replicate_accessions = []
    for rep_idx in range(1, 3):
        tag = f"wt_rep{rep_idx}"
        log.info("Fetching WT replicate %d...", rep_idx)
        rep_a3m = fetch_msa_uncached(wt_seq, cache_dir, tag)
        log.info("  cached at %s", rep_a3m)
        rep_profile, rep_acc = compute_profile(rep_a3m, L)
        wt_replicate_profiles.append(rep_profile)
        wt_replicate_accessions.append(rep_acc)
        log.info("  WT rep%d: %d homologs", rep_idx, len(rep_acc))

    # -- WT replicate comparisons --
    replicate_results = []

    cmp_01 = compare_profiles(
        wt_profile, wt_replicate_profiles[0],
        wt_accessions, wt_replicate_accessions[0],
        L=L,
    )
    cmp_01["label"] = "WT vs WT_rep1"
    replicate_results.append(cmp_01)

    cmp_02 = compare_profiles(
        wt_profile, wt_replicate_profiles[1],
        wt_accessions, wt_replicate_accessions[1],
        L=L,
    )
    cmp_02["label"] = "WT vs WT_rep2"
    replicate_results.append(cmp_02)

    cmp_12 = compare_profiles(
        wt_replicate_profiles[0], wt_replicate_profiles[1],
        wt_replicate_accessions[0], wt_replicate_accessions[1],
        L=L,
    )
    cmp_12["label"] = "WT_rep1 vs WT_rep2"
    replicate_results.append(cmp_12)

    # -- Fetch mutant MSAs --
    mutant_results = []
    for pos, mut_aa in MUTATIONS:
        label = f"{wt_seq[pos]}{pos}{mut_aa}"
        mut_seq = wt_seq[:pos] + mut_aa + wt_seq[pos + 1:]
        log.info("Fetching MSA for %s...", label)
        mut_a3m = fetch_msa(mut_seq, cache_dir)
        log.info("  cached at %s", mut_a3m)
        mut_profile, mut_accessions = compute_profile(mut_a3m, L)

        cmp = compare_profiles(
            wt_profile, mut_profile,
            wt_accessions, mut_accessions,
            mut_pos=pos, L=L,
        )
        cmp["mutation"] = label
        cmp["position"] = pos
        mutant_results.append(cmp)

        log.info(
            "  %s: n_homologs=%d, L1/pos=%.6f, L1@mut=%.6f, L1@other=%.6f, "
            "shared=%.1f%% (%d only_wt, %d only_mut)",
            label, cmp["n_homologs_b"], cmp["total_l1_norm_per_pos"],
            cmp["l1_at_mutated_col"], cmp["mean_l1_other_cols"],
            cmp["pct_shared"], cmp["homolog_set_only_a"], cmp["homolog_set_only_b"],
        )

    # -- Verdict --
    rep_mean_l1 = np.mean([r["total_l1_norm_per_pos"] for r in replicate_results])
    rep_mean_shared = np.mean([r["pct_shared"] for r in replicate_results])

    mut_mean_l1 = np.mean([r["total_l1_norm_per_pos"] for r in mutant_results])
    mut_mean_shared = np.mean([r["pct_shared"] for r in mutant_results])
    mut_mean_ratio = np.mean([r["ratio_mut_vs_other"] for r in mutant_results])

    if rep_mean_shared > 95:
        noise_floor_label = "low"
        if mut_mean_shared > 90 and mut_mean_l1 < 0.02:
            verdict = (
                "PROFILES NEAR-IDENTICAL: WT replicates share {:.1f}% of homologs "
                "(L1={:.6f}); mutants share {:.1f}% (L1={:.6f}). "
                "MSA re-query is NOT load-bearing; pin the WT MSA and move on."
            ).format(rep_mean_shared, rep_mean_l1, mut_mean_shared, mut_mean_l1)
        elif mut_mean_ratio > 3.0:
            verdict = (
                "MUTATED COLUMN MOVES SELECTIVELY: WT replicates share {:.1f}% "
                "(noise floor L1={:.6f}); mutant L1@mut/L1@other = {:.1f}x. "
                "Real conservation signal; pinned-vs-fresh delta is worth measuring."
            ).format(rep_mean_shared, rep_mean_l1, mut_mean_ratio)
        else:
            verdict = (
                "ALIGNMENT MEMBERSHIP SHIFTS BY MUTATION, PROFILE SHIFTS UNIFORMLY: "
                "WT replicates share {:.1f}% of homologs (noise floor L1={:.6f}), "
                "but mutants share only {:.1f}% (L1={:.6f}). "
                "L1@mut/L1@other ratio = {:.1f}x -- the profile shift is because "
                "different homologs are retrieved, not because the mutated column "
                "changes selectively. The mutation genuinely changes which borderline "
                "hits pass the search threshold, but the resulting profile delta is "
                "uniform across columns, not informative at the mutated position. "
                "Pin the WT MSA."
            ).format(
                rep_mean_shared, rep_mean_l1,
                mut_mean_shared, mut_mean_l1,
                mut_mean_ratio,
            )
    else:
        noise_floor_label = "high"
        verdict = (
            "SERVER NONDETERMINISM DOMINATES: even WT replicates share only {:.1f}% "
            "of homologs (L1={:.6f}). Mutant variation ({:.1f}% shared, L1={:.6f}) "
            "cannot be distinguished from noise. Pin the WT MSA immediately."
        ).format(rep_mean_shared, rep_mean_l1, mut_mean_shared, mut_mean_l1)

    output = {
        "query": "1JTG chain B",
        "chain_length": L,
        "wt_replicates": replicate_results,
        "mutations": mutant_results,
        "summary": {
            "replicate_mean_l1_per_pos": float(rep_mean_l1),
            "replicate_mean_pct_shared": float(rep_mean_shared),
            "mutant_mean_l1_per_pos": float(mut_mean_l1),
            "mutant_mean_pct_shared": float(mut_mean_shared),
            "mutant_mean_ratio_mut_vs_other": float(mut_mean_ratio),
            "noise_floor": noise_floor_label,
        },
        "verdict": verdict,
        "notes": {
            "msa_depth": (
                "mode='all' returns only UniRef hits (71 homologs for this query). "
                "mode='env' would add ~137 BFD/Mgnify/MetaEuk/SMAG environmental "
                "hits. The existing repo code (skempi.py) uses mode='all', so this "
                "diagnostic matches that."
            ),
            "existing_bug": (
                "src/igv/skempi.py:824 does resp.text on the /result/download/{id} "
                "response, which returns tar.gz binary. This corrupts the data via "
                "UTF-8 decoding of raw gzip bytes. Not fixed here; file separately."
            ),
        },
    }

    # -- Print tables --
    print()
    print("=" * 110)
    print("MSA SENSITIVITY DIAGNOSTIC: 1JTG chain B")
    print("=" * 110)

    print()
    print("--- WT REPLICATE CONTROL (noise floor) ---")
    print()
    hdr_rep = (
        f"{'Comparison':>20s}  {'#A':>5s}  {'#B':>5s}  "
        f"{'L1/pos':>10s}  {'Shared':>6s}  {'OnlyA':>6s}  {'OnlyB':>6s}  {'%Shared':>8s}"
    )
    print(hdr_rep)
    print("-" * len(hdr_rep))
    for r in replicate_results:
        print(
            f"{r['label']:>20s}  {r['n_homologs_a']:>5d}  {r['n_homologs_b']:>5d}  "
            f"{r['total_l1_norm_per_pos']:>10.6f}  {r['homolog_set_shared']:>6d}  "
            f"{r['homolog_set_only_a']:>6d}  {r['homolog_set_only_b']:>6d}  "
            f"{r['pct_shared']:>7.1f}%"
        )

    print()
    print("--- MUTANT vs WT ---")
    print()
    hdr_mut = (
        f"{'Mutation':>10s}  {'#WT':>5s}  {'#Mut':>5s}  "
        f"{'L1/pos':>10s}  {'L1@mut':>10s}  {'L1@other':>10s}  "
        f"{'Ratio':>6s}  {'Shared':>6s}  {'OnlyWT':>6s}  {'OnlyMut':>7s}  {'%Shared':>8s}"
    )
    print(hdr_mut)
    print("-" * len(hdr_mut))
    for r in mutant_results:
        print(
            f"{r['mutation']:>10s}  {r['n_homologs_a']:>5d}  {r['n_homologs_b']:>5d}  "
            f"{r['total_l1_norm_per_pos']:>10.6f}  {r['l1_at_mutated_col']:>10.6f}  "
            f"{r['mean_l1_other_cols']:>10.6f}  "
            f"{r['ratio_mut_vs_other']:>6.1f}  "
            f"{r['homolog_set_shared']:>6d}  {r['homolog_set_only_a']:>6d}  "
            f"{r['homolog_set_only_b']:>7d}  "
            f"{r['pct_shared']:>7.1f}%"
        )

    print()
    print("VERDICT:", verdict)
    print()

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
