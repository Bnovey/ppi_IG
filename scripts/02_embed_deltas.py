#!/usr/bin/env python3
"""Compute per-substitution embedding deltas for gradient-based score prediction."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import build_library, download, read_pdb_chains
from igv.gpu import require_vram
from igv.provenance import write as prov_write

log = logging.getLogger(__name__)

STRUCTURE_FOR_DATASET = {
    "4fqi_h1": "4fqi_hlab",
    "4fqi_h3": "4fqi_hlab",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-substitution embedding deltas"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. 4fqi_h1)")
    parser.add_argument("--chain", default="H", choices=["H", "L"])
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    out_path = Path(args.out.format(dataset=args.dataset))
    cache_dir = Path(args.cache_dir)
    chain = args.chain

    # 1. Load library and structure
    log.info("Loading library %s (chain=%s)", args.dataset, chain)
    lib = build_library(args.dataset, cache_dir, chain=chain)

    struct_name = args.structure or STRUCTURE_FOR_DATASET.get(args.dataset)
    if struct_name is None:
        base = args.dataset.rsplit("_", 1)[0]
        struct_name = f"{base}_hlab"
        log.info("No structure mapping; guessing %s", struct_name)

    pdb_path = download(struct_name, "structure", cache_dir)
    pdb_chains = read_pdb_chains(pdb_path)

    # 2. Build chains dict
    chains: dict[str, str] = {}
    chains[chain] = lib.reference_seq
    for ch, seq in pdb_chains.items():
        if ch != chain:
            chains[ch] = seq

    for ch, seq in chains.items():
        log.info("  chain %s: %d aa", ch, len(seq))

    # 3. Enumerate substitutions
    substitutions: list[tuple[int, str]] = []
    for pos in lib.variable_positions:
        ref_aa = lib.reference_seq[pos]
        for aa in lib.alphabet_at[pos]:
            if aa != ref_aa:
                substitutions.append((pos, aa))

    n_deltas = len(substitutions)
    assert n_deltas < 500, (
        f"Too many substitutions ({n_deltas}); library may be mis-parsed"
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

    import numpy as np
    import torch
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

    # 6. Reference embedding
    log.info("Featurising reference complex")
    feats_ref, token_map = build_complex_feats(
        chains, pdb_path, cache_dir, device, use_msa_server=not args.no_msa_server,
    )
    s_inputs_ref = embedder_only(model, feats_ref)
    log.info("Reference s_inputs shape: %s", s_inputs_ref.shape)

    # Token indices for the varying chain in residue order
    token_indices = np.array(
        [token_map[(chain, i)] for i in range(len(lib.reference_seq))],
        dtype=np.int64,
    )

    # 7. Compute deltas
    results: dict[str, np.ndarray] = {}

    for idx, (pos, aa) in enumerate(substitutions, 1):
        log.info("Delta %d/%d: position %d -> %s", idx, n_deltas, pos, aa)

        mutant_seq = lib.reference_seq[:pos] + aa + lib.reference_seq[pos + 1 :]
        mut_chains = dict(chains)
        mut_chains[chain] = mutant_seq

        feats_mut, token_map_mut = build_complex_feats(
            mut_chains, pdb_path, cache_dir, device,
            use_msa_server=not args.no_msa_server,
        )
        s_inputs_mut = embedder_only(model, feats_mut)

        token_idx = token_map[(chain, pos)]
        delta = (
            s_inputs_mut[0, token_idx] - s_inputs_ref[0, token_idx]
        ).cpu().float().numpy()

        results[f"{pos}_{aa}"] = delta
        free_cuda_memory()

    # 8. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        reference_s_inputs=s_inputs_ref.cpu().float().numpy(),
        reference_seq=np.array(lib.reference_seq),
        token_indices=token_indices,
        chain=np.array(chain),
        variable_positions=np.array(lib.variable_positions, dtype=np.int64),
        **results,
    )
    log.info("Wrote %s (%d delta arrays)", out_path, n_deltas)

    # 9. Provenance
    prov_write(
        out_path,
        stage="02_embed_deltas",
        inputs={"dataset": args.dataset, "structure": struct_name, "cache_dir": str(cache_dir)},
        params={"chain": chain, "device": device, "boltz_version": boltz_version},
        arm={"stage": "deltas", "chain": chain, "dataset": args.dataset,
             "n_deltas": n_deltas, **numerics_arm()},
    )


if __name__ == "__main__":
    main()
