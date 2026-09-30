#!/usr/bin/env python3
"""Stage 04 -- brute-force mutation scan (the ground truth for T1 and T2).

Re-scores sampled mutants with an explicit forward pass. This is the expensive
procedure the single backward pass in stage 03 is supposed to replace, so it
serves two of the study's three terms:

    T1  attribution vs THIS scan        -> is the gradient faithful to the model?
    T2  THIS scan vs measured affinity  -> is the model right about biology?

Two design points matter.

Fixed geometry. Every mutant reuses the wild-type structure's coordinates as
``x_pred``. AbBiBench supplies the WT complex and does not re-predict mutant
structures, so this removes both the cost of diffusion and its stochasticity,
and -- more importantly -- makes the scan and the gradient measure sensitivity
at *identical* geometry. Without that, T1 would conflate attribution error with
structural resampling noise.

Stratified sampling. A uniform sample of a combinatorially complete binary
library over 16 positions is overwhelmingly concentrated near 8 mutations
(C(16,8)=12870 of 65536). Sampling uniformly would leave the epistasis analysis
in stage 03's notebook with almost no low- or high-order points, which are
exactly the strata that reveal where first-order additivity breaks. So we
allocate the budget across n_mut strata instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.data import build_library, read_pdb_chains  # noqa: E402
from igv.dms import resolve_pdb_complex  # noqa: E402
from igv.gpu import require_vram  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402
from igv.skempi import skempi_positions  # noqa: E402

log = logging.getLogger("scan")

AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")

# 4fqi_h1 and 4fqi_h3 share one deposited complex.
_STRUCTURE_FOR = {"4fqi_h1": "4fqi_hlab", "4fqi_h3": "4fqi_hlab"}

_COLUMNS = [
    "row_index",
    "sequence",
    "substitutions",
    "n_mut",
    "binding_score",
    "model_score",
]


def stratified_sample(n_mut: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    """Spread the sampling budget across mutation-count strata.

    Returns row indices. Strata are filled round-robin so that every observed
    n_mut contributes before any stratum is sampled twice; this guarantees the
    tails (n_mut 0-2 and 14-16), which carry the epistasis signal, are present
    even at small ``n_sample``.
    """
    rng = np.random.default_rng(seed)
    by_stratum: dict[int, np.ndarray] = {}
    for k in np.unique(n_mut):
        idx = np.flatnonzero(n_mut == k)
        by_stratum[int(k)] = rng.permutation(idx)

    chosen: list[int] = []
    cursors = {k: 0 for k in by_stratum}
    strata = sorted(by_stratum)
    while len(chosen) < n_sample:
        progressed = False
        for k in strata:
            if len(chosen) >= n_sample:
                break
            c = cursors[k]
            if c < len(by_stratum[k]):
                chosen.append(int(by_stratum[k][c]))
                cursors[k] = c + 1
                progressed = True
        if not progressed:  # library smaller than the requested budget
            break
    return np.array(sorted(chosen), dtype=int)


def _fmt_subs(subs: tuple[tuple[int, str], ...]) -> str:
    return ";".join(f"{pos}{aa}" for pos, aa in subs) if subs else ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--chain", default="H")
    p.add_argument("--score", default="complex_pde")
    p.add_argument("--n-sample", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--structure", default=None, help="Structure stem override")
    p.add_argument("--out", default="results/{dataset}_{score}_scan.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    p.add_argument("--no-msa-server", action="store_true")
    p.add_argument(
        "--positions",
        default=None,
        help=(
            "Positions to scan (full 19-aa grid). "
            "'skempi' = SKEMPI-measured positions; "
            "'dms_interface' = DMS interface positions (see --interface-cutoff); "
            "or comma-separated 0-based indices. "
            "Required for SKEMPI/DMS datasets."
        ),
    )
    p.add_argument(
        "--interface-cutoff",
        type=float,
        default=5.0,
        help="Heavy-atom distance cutoff (A) for dms_interface positions (default 5.0)",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan the sample, no GPU")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    out = Path(args.out.format(dataset=args.dataset, score=args.score))
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    # --- Resolve dataset source ---
    registry_error: KeyError | None = None
    try:
        resolved = resolve_pdb_complex(
            args.dataset, args.chain, cache_dir, structure_override=args.structure,
        )
    except KeyError as exc:
        resolved = None
        registry_error = exc

    is_dms = resolved is not None and resolved.data_source == "dms"
    is_skempi = resolved is not None and resolved.data_source == "skempi"

    if is_dms:
        # DMS path: exhaustive scan of measured interface mutants.
        from igv.dms import (
            enumerate_mutants,
            get_complex as dms_get_complex,
            interface_positions,
            load_starr2020,
            map_sites_to_indices,
            singles,
        )
        from igv.skempi import read_pdb_residue_ids

        dms_cx = dms_get_complex(args.dataset)
        pdb = resolved.pdb_path
        reference_seq = resolved.reference_seq
        struct_name = resolved.struct_name
        struct_chains = resolved.chains

        residue_ids, _seqs = read_pdb_residue_ids(pdb)
        positions = interface_positions(
            pdb,
            chain=args.chain,
            partner_chains=dms_cx.partner_chains,
            residue_ids=residue_ids[args.chain],
            cutoff=args.interface_cutoff,
        )

        df = load_starr2020(cache_dir)
        df = singles(df)
        mapped, _ = map_sites_to_indices(df, residue_ids[args.chain], reference_seq)
        mutant_pairs = enumerate_mutants(
            df, positions, sequence=reference_seq, site_to_idx=mapped,
        )
        log.info(
            "DMS exhaustive scan: %d interface positions, %d mutant pairs",
            len(positions), len(mutant_pairs),
        )

        if args.dry_run:
            log.info(
                "--dry-run: would score %d mutants with score=%s",
                len(mutant_pairs), args.score,
            )
            return

    elif is_skempi:
        # SKEMPI path: all 19 substitutions at positions selected by --positions.
        if args.positions is None:
            raise SystemExit(
                "SKEMPI complexes require --positions to specify which positions "
                "to scan (e.g. --positions skempi for SKEMPI-measured positions)."
            )

        pdb = resolved.pdb_path
        reference_seq = resolved.reference_seq
        struct_name = resolved.struct_name
        struct_chains = resolved.chains

        if args.positions == "skempi":
            positions = skempi_positions(args.dataset, args.chain, cache_dir)
        elif args.positions == "dms_interface":
            raise SystemExit(
                "dms_interface positions are not applicable to SKEMPI complexes. "
                "Use --positions skempi or explicit indices."
            )
        else:
            positions = sorted(int(x) for x in args.positions.split(","))

        mutant_pairs: list[tuple[int, str]] = []
        for pos in positions:
            if pos < 0 or pos >= len(reference_seq):
                raise ValueError(
                    f"Position {pos} out of range for chain {args.chain} "
                    f"(length {len(reference_seq)})"
                )
            ref_aa = reference_seq[pos]
            for aa in AMINO_ACIDS:
                if aa != ref_aa:
                    mutant_pairs.append((pos, aa))

        log.info(
            "SKEMPI brute-force scan: %d positions, %d mutant pairs",
            len(positions), len(mutant_pairs),
        )

        if args.dry_run:
            log.info(
                "--dry-run: would score %d mutants with score=%s",
                len(mutant_pairs), args.score,
            )
            return

    else:
        # --- AbBiBench path (existing logic) ---
        try:
            lib = build_library(args.dataset, cache_dir, chain=args.chain)
        except Exception:
            if registry_error is not None:
                raise SystemExit(
                    f"Dataset {args.dataset!r} not found in any registry. "
                    f"{registry_error}"
                ) from registry_error
            raise
        seq_col = "heavy_chain_seq" if args.chain == "H" else "light_chain_seq"
        n_mut = lib.frame["n_mut"].to_numpy()

        idx = stratified_sample(n_mut, args.n_sample, args.seed)
        log.info("Sampled %d of %d rows (seed=%d)", len(idx), len(lib.frame), args.seed)
        dist = pd.Series(n_mut[idx]).value_counts().sort_index()
        log.info("Achieved n_mut distribution: %s", dist.to_dict())

        # Resume: a scan is the most expensive stage and gets interrupted.
        done: set[int] = set()
        if out.exists():
            prev = pd.read_csv(out)
            done = set(prev["row_index"].astype(int))
            log.info("Resuming: %d rows already scored in %s", len(done), out)
        todo = [i for i in idx if i not in done]
        if not todo:
            log.info("Nothing to do; all %d sampled rows already scored.", len(idx))
            return

        if args.dry_run:
            log.info("--dry-run: would score %d mutants with score=%s", len(todo), args.score)
            return

    # Same 78 GiB gate as before, and still AFTER the --dry-run return above so the
    # no-GPU dry run keeps working. require_vram() logs the usable GiB itself, which
    # is why the old log.info("GPU 0: %.1f GiB", gib) line is not repeated here.
    require_vram()

    from igv.attrib import free_cuda_memory
    from igv.boltz_score import (
        SCORES,
        build_complex_feats,
        confidence_forward,
        embedder_only,
        load_model,
        numerics_arm,
    )
    import torch

    if args.score not in SCORES:
        raise SystemExit(f"--score must be one of {sorted(SCORES)}")

    scan_suffix = f"_{args.dataset}"

    if is_dms:
        model, _boltz_version = load_model(args.checkpoint_dir, args.device)

        def chains_for(seq: str) -> dict[str, str]:
            d = dict(struct_chains)
            d[args.chain] = seq
            return d

        ref_feats, _ = build_complex_feats(
            chains_for(reference_seq),
            pdb,
            cache_dir / f"boltz_ref{scan_suffix}",
            args.device,
            use_msa_server=not args.no_msa_server,
        )
        x_pred = ref_feats["coords"].detach()
        log.info("Geometry FIXED from %s; x_pred %s", struct_name, tuple(x_pred.shape))

        write_header = not out.exists()
        t0 = time.time()
        for n, (pos, aa) in enumerate(mutant_pairs, 1):
            mut_seq = reference_seq[:pos] + aa + reference_seq[pos + 1:]
            feats, _ = build_complex_feats(
                chains_for(mut_seq),
                pdb,
                cache_dir / f"boltz_scan{scan_suffix}/{pos}_{aa}",
                args.device,
                use_msa_server=not args.no_msa_server,
            )
            with torch.no_grad():
                s_inputs = embedder_only(model, feats)
                scalar = confidence_forward(
                    model, s_inputs, feats, x_pred, args.score,
                    gradient_checkpointing=False,
                )
                model_score = float(scalar)

            pd.DataFrame(
                [
                    {
                        "row_index": n - 1,
                        "sequence": mut_seq,
                        "substitutions": f"{pos}{aa}",
                        "n_mut": 1,
                        "binding_score": float("nan"),
                        "model_score": model_score,
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)
            write_header = False

            free_cuda_memory()
            rate = (time.time() - t0) / n
            log.info(
                "[%d/%d] pos=%d aa=%s score=%.6f (%.1fs/mutant, eta %.0f min)",
                n, len(mutant_pairs), pos, aa, model_score,
                rate, rate * (len(mutant_pairs) - n) / 60,
            )

        prov_write(
            out,
            stage="04_scan",
            inputs={"dataset": args.dataset, "structure": str(pdb)},
            params={
                "n_sample": len(mutant_pairs),
                "seed": args.seed,
                "chain": args.chain,
                "sampling": "dms_interface_exhaustive",
            },
            arm={
                "score": args.score,
                "method": "scan",
                "trunk": "forward_only",
                "geometry": "fixed_from_featurisation",
                "dataset": args.dataset,
                "chain": args.chain,
                **numerics_arm(),
            },
            notes="DMS exhaustive interface scan.",
        )
        log.info("Wrote %s (%d rows, %.1f min)", out, len(mutant_pairs), (time.time() - t0) / 60)
    elif is_skempi:
        model, _boltz_version = load_model(args.checkpoint_dir, args.device)

        def chains_for(seq: str) -> dict[str, str]:
            d = dict(struct_chains)
            d[args.chain] = seq
            return d

        ref_feats, _ = build_complex_feats(
            chains_for(reference_seq),
            pdb,
            cache_dir / f"boltz_ref{scan_suffix}",
            args.device,
            use_msa_server=not args.no_msa_server,
        )
        x_pred = ref_feats["coords"].detach()
        log.info("Geometry FIXED from %s; x_pred %s", struct_name, tuple(x_pred.shape))

        write_header = not out.exists()
        t0 = time.time()
        for n, (pos, aa) in enumerate(mutant_pairs, 1):
            mut_seq = reference_seq[:pos] + aa + reference_seq[pos + 1:]
            feats, _ = build_complex_feats(
                chains_for(mut_seq),
                pdb,
                cache_dir / f"boltz_scan{scan_suffix}/{pos}_{aa}",
                args.device,
                use_msa_server=not args.no_msa_server,
            )
            with torch.no_grad():
                s_inputs = embedder_only(model, feats)
                scalar = confidence_forward(
                    model, s_inputs, feats, x_pred, args.score,
                    gradient_checkpointing=False,
                )
                model_score = float(scalar)

            pd.DataFrame(
                [
                    {
                        "row_index": n - 1,
                        "sequence": mut_seq,
                        "substitutions": f"{pos}{aa}",
                        "n_mut": 1,
                        "binding_score": float("nan"),
                        "model_score": model_score,
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)
            write_header = False

            free_cuda_memory()
            rate = (time.time() - t0) / n
            log.info(
                "[%d/%d] pos=%d aa=%s score=%.6f (%.1fs/mutant, eta %.0f min)",
                n, len(mutant_pairs), pos, aa, model_score,
                rate, rate * (len(mutant_pairs) - n) / 60,
            )

        prov_write(
            out,
            stage="04_scan",
            inputs={"dataset": args.dataset, "structure": str(pdb)},
            params={
                "n_sample": len(mutant_pairs),
                "seed": args.seed,
                "chain": args.chain,
                "sampling": "skempi_brute_force",
            },
            arm={
                "score": args.score,
                "method": "scan",
                "trunk": "forward_only",
                "geometry": "fixed_from_featurisation",
                "dataset": args.dataset,
                "chain": args.chain,
                **numerics_arm(),
            },
            notes="SKEMPI brute-force scan: all 19 substitutions at measured positions.",
        )
        log.info("Wrote %s (%d rows, %.1f min)", out, len(mutant_pairs), (time.time() - t0) / 60)
    else:
        stem = args.structure or _STRUCTURE_FOR.get(args.dataset)
        if stem is None:
            raise SystemExit(
                f"No structure known for {args.dataset}; pass --structure explicitly."
            )
        pdb = cache_dir / f"{stem}.pdb"
        if not pdb.exists():
            raise SystemExit(f"Missing {pdb}. Run scripts/00_fetch_data.py first.")

        struct_chains = read_pdb_chains(pdb)
        log.info("Structure %s chains: %s", stem, {c: len(s) for c, s in struct_chains.items()})

        model, _boltz_version = load_model(args.checkpoint_dir, args.device)

        def chains_for(seq: str) -> dict[str, str]:
            d = dict(struct_chains)
            d[args.chain] = seq
            return d

        # Fixed geometry: featurise the wild type once and keep its coordinates for
        # every mutant, so scan and gradient share identical structure.
        ref_feats, _ = build_complex_feats(
            chains_for(lib.reference_seq),
            pdb,
            cache_dir / f"boltz_ref{scan_suffix}",
            args.device,
            use_msa_server=not args.no_msa_server,
        )
        x_pred = ref_feats["coords"].detach()
        log.info("Geometry FIXED from %s; shared with stage 03. x_pred %s", stem, tuple(x_pred.shape))

        write_header = not out.exists()
        t0 = time.time()
        for n, row_i in enumerate(todo, 1):
            row = lib.frame.iloc[row_i]
            seq = row[seq_col]
            feats, _ = build_complex_feats(
                chains_for(seq),
                pdb,
                cache_dir / f"boltz_scan{scan_suffix}/{row_i}",
                args.device,
                use_msa_server=not args.no_msa_server,
            )
            with torch.no_grad():
                s_inputs = embedder_only(model, feats)
                scalar = confidence_forward(
                    model, s_inputs, feats, x_pred, args.score, gradient_checkpointing=False
                )
                model_score = float(scalar)

            pd.DataFrame(
                [
                    {
                        "row_index": int(row_i),
                        "sequence": seq,
                        "substitutions": _fmt_subs(lib.substitutions[row_i]),
                        "n_mut": int(row["n_mut"]),
                        "binding_score": float(row["binding_score"]),
                        "model_score": model_score,
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)
            write_header = False

            free_cuda_memory()
            rate = (time.time() - t0) / n
            log.info(
                "[%d/%d] row=%d n_mut=%d score=%.6f (%.1fs/mutant, eta %.0f min)",
                n, len(todo), row_i, int(row["n_mut"]), model_score,
                rate, rate * (len(todo) - n) / 60,
            )

        prov_write(
            out,
            stage="04_scan",
            inputs={"dataset": args.dataset, "structure": str(pdb)},
            params={
                "n_sample": args.n_sample,
                "seed": args.seed,
                "chain": args.chain,
                "sampling": "stratified_by_n_mut",
            },
            arm={
                "score": args.score,
                "method": "scan",
                "trunk": "forward_only",
                "geometry": "fixed_from_featurisation",
                "dataset": args.dataset,
                "chain": args.chain,
                **numerics_arm(),
            },
            notes="Brute-force ground truth for T1/T2. Fixed WT geometry shared with stage 03.",
        )
        log.info("Wrote %s (%d new rows, %.1f min)", out, len(todo), (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
