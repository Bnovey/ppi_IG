#!/usr/bin/env python3
"""Compute per-substitution embedding deltas for gradient-based score prediction."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import build_library, download, download_rcsb, read_pdb_chains
from igv.dms import resolve_pdb_complex
from igv.gpu import require_vram
from igv.provenance import write as prov_write
from igv.skempi import skempi_positions

log = logging.getLogger(__name__)

STRUCTURE_FOR_DATASET = {
    "4fqi_h1": "4fqi_hlab",
    "4fqi_h3": "4fqi_hlab",
}

AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")


def _dms_positions(
    dataset: str, chain: str, cache_dir: Path, *, cutoff: float = 5.0,
) -> list[int]:
    """Return 0-based interface positions for a DMS complex."""
    from igv.dms import get_complex as dms_get_complex, interface_positions
    from igv.skempi import read_pdb_residue_ids

    dms_cx = dms_get_complex(dataset)
    pdb_path = download_rcsb(dms_cx.pdb_id, cache_dir)
    residue_ids, _seqs = read_pdb_residue_ids(pdb_path)
    return interface_positions(
        pdb_path,
        chain=chain,
        partner_chains=dms_cx.partner_chains,
        residue_ids=residue_ids[chain],
        cutoff=cutoff,
    )


def _parse_positions(
    spec: str,
    dataset: str,
    chain: str,
    cache_dir: Path,
    *,
    interface_cutoff: float = 5.0,
) -> list[int]:
    """Parse a position specification string into a sorted list of 0-based indices."""
    if spec == "skempi":
        return skempi_positions(dataset, chain, cache_dir)
    if spec == "dms_interface":
        return _dms_positions(dataset, chain, cache_dir, cutoff=interface_cutoff)
    return sorted(int(x) for x in spec.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-substitution embedding deltas"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. 4fqi_h1)")
    parser.add_argument("--chain", default="H", help="Varying chain id")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument(
        "--out",
        default="data/processed/{dataset}_deltas.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
        help="Boltz-2 checkpoint directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--structure", default=None, help="Override structure name")
    parser.add_argument("--no-msa-server", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument(
        "--positions",
        default=None,
        help=(
            "Positions to scan (full 19-aa grid). "
            "'skempi' = SKEMPI-measured positions; "
            "'dms_interface' = DMS interface positions (see --interface-cutoff); "
            "or comma-separated 0-based indices. "
            "Default (None): use AbBiBench library substitutions."
        ),
    )
    parser.add_argument(
        "--interface-cutoff",
        type=float,
        default=5.0,
        help="Heavy-atom distance cutoff (A) for dms_interface positions (default 5.0)",
    )
    parser.add_argument(
        "--max-substitutions",
        type=int,
        default=5000,
        help="Guard against mis-parsed input (default 5000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    out_path = Path(args.out.format(dataset=args.dataset))
    cache_dir = Path(args.cache_dir)
    chain = args.chain

    # 1. Load structure and reference sequence
    try:
        resolved = resolve_pdb_complex(
            args.dataset, chain, cache_dir, structure_override=args.structure,
        )
    except KeyError:
        resolved = None

    if resolved is not None:
        data_source = resolved.data_source
        log.info("%s complex %s (chain=%s)", data_source.upper(), resolved.struct_name, chain)
        pdb_path = resolved.pdb_path
        reference_seq = resolved.reference_seq
        struct_name = resolved.struct_name
        chains: dict[str, str] = resolved.chains
        n_tokens = resolved.n_tokens
        log.info("Chain subset: %s  L=%d", sorted(chains), n_tokens)
    else:
        data_source = "abbibench"
        log.info("Loading library %s (chain=%s)", args.dataset, chain)
        lib = build_library(args.dataset, cache_dir, chain=chain)
        reference_seq = lib.reference_seq

        struct_name = args.structure or STRUCTURE_FOR_DATASET.get(args.dataset)
        if struct_name is None:
            base = args.dataset.rsplit("_", 1)[0]
            struct_name = f"{base}_hlab"
            log.info("No structure mapping; guessing %s", struct_name)

        pdb_path = download(struct_name, "structure", cache_dir)
        pdb_chains = read_pdb_chains(pdb_path)

        chains = {}
        chains[chain] = lib.reference_seq
        for ch, seq in pdb_chains.items():
            if ch != chain:
                chains[ch] = seq

    for ch, seq in chains.items():
        log.info("  chain %s: %d aa", ch, len(seq))

    # 3. Enumerate substitutions
    substitutions: list[tuple[int, str]] = []

    if args.positions is not None:
        positions = _parse_positions(
            args.positions, args.dataset, chain, cache_dir,
            interface_cutoff=args.interface_cutoff,
        )
        log.info("Full-grid scan at %d positions", len(positions))
        for pos in positions:
            if pos < 0 or pos >= len(reference_seq):
                raise ValueError(
                    f"Position {pos} out of range for chain {chain} "
                    f"(length {len(reference_seq)})"
                )
            ref_aa = reference_seq[pos]
            for aa in AMINO_ACIDS:
                if aa != ref_aa:
                    substitutions.append((pos, aa))
    else:
        if resolved is not None:
            raise SystemExit(
                f"{data_source.upper()} complexes have no AbBiBench library. "
                "Use --positions (e.g. --positions skempi or --positions dms_interface) "
                "to specify which positions to scan."
            )
        for pos in lib.variable_positions:
            ref_aa = lib.reference_seq[pos]
            for aa in lib.alphabet_at[pos]:
                if aa != ref_aa:
                    substitutions.append((pos, aa))

    n_deltas = len(substitutions)
    if n_deltas > args.max_substitutions:
        raise SystemExit(
            f"Too many substitutions ({n_deltas}); limit is "
            f"--max-substitutions={args.max_substitutions}. "
            f"Raise the limit if this is intentional."
        )
    log.info("Distinct substitutions to embed: %d", n_deltas)

    # 4. Check for existing output with all expected keys
    expected_keys = set()
    for pos, aa in substitutions:
        expected_keys.add(f"{pos}_{aa}")
    expected_keys |= {
        "reference_s_inputs", "reference_seq", "token_indices",
        "chain", "variable_positions",
    }

    if out_path.exists() and not args.force:
        import numpy as np

        existing = np.load(out_path, allow_pickle=True)
        if expected_keys <= set(existing.files):
            log.info("Output %s already contains all keys; skipping (use --force)", out_path)
            return

    # 5. VRAM check (embedder-only is lighter but we keep the check for consistency)
    log.info(
        "This stage is embedder-only (no trunk), but VRAM check is kept for consistency"
    )
    # BEHAVIOUR CHANGE: this gate now actually runs. The local copy this replaced
    # read torch.cuda.get_device_properties(0).total_mem -- an attribute that does
    # not exist -- and the enclosing try caught only ImportError, so on a real GPU
    # this stage died with an unhandled AttributeError instead of enforcing 78 GiB.
    # min_gib is deliberately unchanged from the other stages; pass min_gib=... if
    # stage 02's embedder-only footprint should ever be gated more loosely.
    require_vram()

    import time

    import numpy as np
    from igv.attrib import free_cuda_memory
    from igv.boltz_score import (
        build_complex_feats,
        embedder_only,
        load_model,
        numerics_arm,
    )

    device = args.device
    log.info("Loading model from %s on %s", args.checkpoint_dir, device)
    model, boltz_version = load_model(args.checkpoint_dir, device)

    # 6. Reference embedding (server MSA)
    log.info("Featurising reference complex (server MSA)")
    ref_cache = cache_dir / "boltz_delta" / "ref_server"
    feats_ref_server, token_map = build_complex_feats(
        chains, pdb_path, ref_cache, device,
        use_msa_server=not args.no_msa_server,
    )

    # 6a. Locate per-chain MSA files written by boltz under the reference cache.
    # boltz writes <cache_dir>/msa/input_<N>.csv where N indexes the YAML
    # sequences list, which build_complex_feats emits in chains.items() order.
    ref_msa_dir = ref_cache / "msa"
    # Sort NUMERICALLY on <N>, not lexicographically: a plain sorted() puts
    # input_10 before input_2, so the chain->MSA mapping would silently shear
    # the moment a complex has more than nine chains. Today's have four, which
    # is exactly the condition under which this bug would go unnoticed.
    def _msa_index(path):
        return int(path.stem.rsplit("_", 1)[1])

    msa_files = sorted(ref_msa_dir.glob("input_*.csv"), key=_msa_index)
    chain_ids = list(chains.keys())
    if len(msa_files) != len(chain_ids):
        raise RuntimeError(
            f"Expected {len(chain_ids)} MSA files in {ref_msa_dir} "
            f"(one per chain: {chain_ids}), found {len(msa_files)}: "
            f"{[f.name for f in msa_files]}. Cannot build per-chain MSA mapping."
        )
    msa_by_chain: dict[str, Path] = {}
    for chain_id, msa_file in zip(chain_ids, msa_files):
        msa_by_chain[chain_id] = msa_file
    log.info("Per-chain MSA mapping: %s", {k: v.name for k, v in msa_by_chain.items()})

    # 6b. Re-featurise reference from file-loaded MSAs into a fresh cache dir.
    # Both sides of every delta subtraction use the same code path (file-loaded),
    # so no part of the delta can be a server-vs-file artifact.
    log.info("Re-featurising reference with file-loaded MSAs")
    ref_cache_from_files = cache_dir / "boltz_delta" / "ref_from_files"
    feats_ref, token_map = build_complex_feats(
        chains, pdb_path, ref_cache_from_files, device,
        use_msa_server=False, msa=msa_by_chain,
    )
    s_inputs_ref_server = embedder_only(model, feats_ref_server)
    s_inputs_ref = embedder_only(model, feats_ref)

    # Compare the two featurisations of the same reference. The correct pair
    # is file-loaded run 1 vs file-loaded run 2 (both sides of every delta use
    # the file path), but we only have server vs file here. Entry 18 in
    # Early profiling measured the file-vs-file gap at ~1.14 and server-vs-file
    # at ~1.14, both driven by unseeded ref_pos in RDKit conformer generation.
    # Tolerance 2.0 accommodates the measured noise; tighten once featurisation
    # is deterministic (ref_pos seeding).
    max_diff = float((s_inputs_ref - s_inputs_ref_server).abs().max().item())
    log.info(
        "Max abs difference between server-MSA and file-loaded-MSA reference "
        "embeddings: %.6e", max_diff,
    )
    if max_diff > 2.0:
        raise RuntimeError(
            f"Reference embeddings from server MSA vs file-loaded MSA differ by "
            f"{max_diff:.6e} (tolerance 2.0). This exceeds the measured ref_pos "
            f"noise scale (~1.5) and may indicate a real MSA-reuse problem."
        )
    del feats_ref_server, s_inputs_ref_server
    free_cuda_memory()

    log.info("Reference s_inputs shape: %s", s_inputs_ref.shape)

    # Token indices for the varying chain in residue order
    token_indices = np.array(
        [token_map[(chain, i)] for i in range(len(reference_seq))],
        dtype=np.int64,
    )

    # Collect variable positions for saving
    variable_positions = sorted({pos for pos, _aa in substitutions})

    # 7. Compute deltas (reusing wild-type MSA for every mutant)
    results: dict[str, np.ndarray] = {}
    t0 = time.monotonic()

    for idx, (pos, aa) in enumerate(substitutions, 1):
        log.info("Delta %d/%d: position %d -> %s", idx, n_deltas, pos, aa)

        mutant_seq = reference_seq[:pos] + aa + reference_seq[pos + 1 :]
        mut_chains = dict(chains)
        mut_chains[chain] = mutant_seq

        feats_mut, token_map_mut = build_complex_feats(
            mut_chains, pdb_path, cache_dir / f"boltz_delta/{pos}_{aa}", device,
            use_msa_server=False, msa=msa_by_chain,
        )
        s_inputs_mut = embedder_only(model, feats_mut)

        token_idx = token_map[(chain, pos)]
        delta = (
            s_inputs_mut[0, token_idx] - s_inputs_ref[0, token_idx]
        ).cpu().float().numpy()

        results[f"{pos}_{aa}"] = delta
        free_cuda_memory()

        if idx % 25 == 0 or idx == n_deltas:
            elapsed = time.monotonic() - t0
            per_sub = elapsed / idx
            remaining = per_sub * (n_deltas - idx)
            log.info(
                "Progress: %d/%d (%.1f%%) | %.1fs/sub | ETA %.0fs (%.1f min)",
                idx, n_deltas, 100.0 * idx / n_deltas,
                per_sub, remaining, remaining / 60.0,
            )

    # 8. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        reference_s_inputs=s_inputs_ref.cpu().float().numpy(),
        reference_seq=np.array(reference_seq),
        token_indices=token_indices,
        chain=np.array(chain),
        variable_positions=np.array(variable_positions, dtype=np.int64),
        **results,
    )
    log.info("Wrote %s (%d delta arrays)", out_path, n_deltas)

    # 9. Provenance
    prov_write(
        out_path,
        stage="02_embed_deltas",
        inputs={"dataset": args.dataset, "structure": struct_name, "cache_dir": str(cache_dir)},
        params={"chain": chain, "device": device, "boltz_version": boltz_version,
                "data_source": data_source,
                "positions_spec": args.positions},
        arm={"stage": "deltas", "chain": chain, "dataset": args.dataset,
             "n_deltas": n_deltas, "msa_reuse": True,
             "msa_ref_cache": str(ref_cache), **numerics_arm()},
    )


if __name__ == "__main__":
    main()
