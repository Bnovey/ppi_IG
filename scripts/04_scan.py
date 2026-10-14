#!/usr/bin/env python3
"""Stage 04 -- brute-force mutation scan (the ground truth for T1 and T2).

Re-scores sampled mutants with an explicit forward pass. This is the expensive
procedure the single backward pass in stage 03 is supposed to replace, so it
serves two of the study's three terms:

    T1  attribution vs THIS scan        -> is the gradient faithful to the model?
    T2  THIS scan vs measured affinity  -> is the model right about biology?

Two scan modes are available.

``resswap`` (default)
    Reuse the exact wild-type features and swap only ``res_type[pos]`` to the
    mutant one-hot. This is the ground truth for a first-order Taylor / TISM
    prediction ``grad[pos, mut] - grad[pos, wt]``, because the gradient is
    taken with respect to ``res_type`` while all other features are held fixed.

``refeaturise``
    Build the mutant sequence from scratch via ``build_complex_feats``. The
    model sees the true mutant (different atom features, MSA, etc.) but the
    wild-type coordinates are reused. A shape guard on ``atom_pad_mask`` and
    ``token_to_rep_atom`` catches mutations that change the atom count
    (e.g. Gly->Trp), which would make ``x_pred`` index the wrong atoms.

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
    "mutation_id",
    "row_index",
    "sequence",
    "substitutions",
    "n_mut",
    "binding_score",
    "model_score",
    "wt_score",
    "delta",
    "mode",
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


def _mutation_id(pos: int, aa: str) -> str:
    return f"{pos}{aa}"


def _load_done_ids(out_path: Path) -> set[str]:
    """Load already-scored mutation_id values from a CSV for resumability."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    try:
        prev = pd.read_csv(out_path)
    except Exception:
        return done
    if "mutation_id" in prev.columns:
        done = set(prev["mutation_id"].astype(str))
    return done


def _load_done_row_indices(out_path: Path) -> set[int]:
    """Load already-scored row indices from CSV for AbBiBench resume."""
    done: set[int] = set()
    if not out_path.exists():
        return done
    try:
        prev = pd.read_csv(out_path)
    except Exception:
        return done
    if "row_index" in prev.columns:
        done = set(prev["row_index"].astype(int))
    return done


def _resolve_msa_by_chain(
    cache_dir: Path, dataset: str, chain: str, chains: dict[str, str],
) -> dict[str, Path] | None:
    """Locate the per-chain MSA files written by stage 03 under boltz_attr."""
    msa_dir = cache_dir / "boltz_attr" / f"{dataset}_{chain}" / "msa"
    if not msa_dir.is_dir():
        return None

    def _msa_index(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[1])

    msa_files = sorted(msa_dir.glob("input_*.csv"), key=_msa_index)
    chain_ids = list(chains.keys())
    if len(msa_files) != len(chain_ids):
        log.warning(
            "Expected %d MSA files in %s (one per chain: %s), found %d. "
            "Cannot pin MSA; will query MSA server.",
            len(chain_ids), msa_dir, chain_ids, len(msa_files),
        )
        return None
    msa_by_chain: dict[str, Path] = {}
    for chain_id, msa_file in zip(chain_ids, msa_files):
        msa_by_chain[chain_id] = msa_file
    log.info("Pinned MSA from stage 03: %s", {k: v.name for k, v in msa_by_chain.items()})
    return msa_by_chain


def _check_atom_shape_guard(wt_feats: dict, mut_feats: dict, pos: int, aa: str) -> bool:
    """Compare atom_pad_mask shape and token_to_rep_atom between WT and mutant.

    Returns True if safe. Logs a WARNING and returns False when shapes differ,
    meaning wild-type coordinates cannot be validly reused.
    """
    wt_mask = wt_feats["atom_pad_mask"]
    mut_mask = mut_feats["atom_pad_mask"]
    if wt_mask.shape != mut_mask.shape:
        log.warning(
            "ATOM SHAPE MISMATCH at pos=%d aa=%s: WT atom_pad_mask %s vs mutant %s. "
            "Wild-type coordinates cannot be reused; skipping this mutant.",
            pos, aa, tuple(wt_mask.shape), tuple(mut_mask.shape),
        )
        return False

    if "token_to_rep_atom" in wt_feats and "token_to_rep_atom" in mut_feats:
        import torch

        wt_rep = wt_feats["token_to_rep_atom"]
        mut_rep = mut_feats["token_to_rep_atom"]
        if wt_rep.shape != mut_rep.shape:
            log.warning(
                "TOKEN_TO_REP_ATOM SHAPE MISMATCH at pos=%d aa=%s: "
                "WT %s vs mutant %s. Skipping.",
                pos, aa, tuple(wt_rep.shape), tuple(mut_rep.shape),
            )
            return False
        if not torch.equal(wt_rep, mut_rep):
            log.warning(
                "TOKEN_TO_REP_ATOM CONTENT DIFFERS at pos=%d aa=%s. "
                "Representative atoms map to different physical atoms; "
                "x_pred would read wrong coordinates. Skipping.",
                pos, aa,
            )
            return False

    return True


def _resswap_scan(
    model, feats, x_pred, score_name, reference_seq, chain, token_map,
    positions, out, mutant_pairs, done_ids, wt_score,
):
    """Run the res_type-swap scan: only change res_type[pos], keep everything else."""
    import torch

    from igv.attrib import CANONICAL_AMINO_ACIDS, free_cuda_memory
    from igv.boltz_score import (
        canonical_aa_token_indices,
        confidence_forward,
        embedder_only,
    )

    aa_col_indices = canonical_aa_token_indices()
    aa_to_col = {aa: int(aa_col_indices[i]) for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}

    res_type_wt = feats["res_type"].detach().clone()

    write_header = not out.exists()
    t0 = time.time()
    n_scored = 0
    n_skipped = 0

    for pos, aa in mutant_pairs:
        mid = _mutation_id(pos, aa)
        if mid in done_ids:
            n_skipped += 1
            continue

        token_idx = token_map.get((chain, pos))
        if token_idx is None:
            log.warning("Position %d not in token_map; skipping %s", pos, mid)
            continue

        mutated_rt = res_type_wt.clone()
        mutated_rt[0, token_idx] = 0.0
        mutated_rt[0, token_idx, aa_to_col[aa]] = 1.0
        feats["res_type"] = mutated_rt

        with torch.no_grad():
            s_inputs = embedder_only(model, feats)
            scalar = confidence_forward(
                model, s_inputs, feats, x_pred, score_name,
                gradient_checkpointing=False,
            )
            model_score = float(scalar)

        feats["res_type"] = res_type_wt

        mut_seq = reference_seq[:pos] + aa + reference_seq[pos + 1:]
        pd.DataFrame(
            [
                {
                    "mutation_id": mid,
                    "row_index": n_scored,
                    "sequence": mut_seq,
                    "substitutions": f"{pos}{aa}",
                    "n_mut": 1,
                    "binding_score": float("nan"),
                    "model_score": model_score,
                    "wt_score": wt_score,
                    "delta": model_score - wt_score,
                    "mode": "resswap",
                }
            ],
            columns=_COLUMNS,
        ).to_csv(out, mode="a", header=write_header, index=False)
        write_header = False

        free_cuda_memory()
        n_scored += 1
        rate = (time.time() - t0) / n_scored
        log.info(
            "[%d/%d] pos=%d aa=%s score=%.6f delta=%.6f (%.1fs/mutant, eta %.0f min)",
            n_scored + n_skipped, len(mutant_pairs), pos, aa, model_score,
            model_score - wt_score,
            rate, rate * (len(mutant_pairs) - n_scored - n_skipped) / 60,
        )

    return n_scored, n_skipped


def _refeaturise_scan(
    model, ref_feats, x_pred, score_name, reference_seq, chain,
    struct_chains, pdb, cache_dir, dataset, device,
    positions, out, mutant_pairs, done_ids, wt_score,
    use_msa_server, msa_by_chain,
):
    """Run the refeaturise scan: build full mutant features, reuse WT coords."""
    import torch

    from igv.attrib import free_cuda_memory
    from igv.boltz_score import (
        build_complex_feats,
        confidence_forward,
        embedder_only,
    )

    scan_suffix = f"_{dataset}"

    def chains_for(seq: str) -> dict[str, str]:
        d = dict(struct_chains)
        d[chain] = seq
        return d

    write_header = not out.exists()
    t0 = time.time()
    n_scored = 0
    n_skipped_resume = 0
    n_skipped_shape = 0

    for pos, aa in mutant_pairs:
        mid = _mutation_id(pos, aa)
        if mid in done_ids:
            n_skipped_resume += 1
            continue

        mut_seq = reference_seq[:pos] + aa + reference_seq[pos + 1:]
        feats, _ = build_complex_feats(
            chains_for(mut_seq),
            pdb,
            cache_dir / f"boltz_scan{scan_suffix}/{pos}_{aa}",
            device,
            use_msa_server=use_msa_server,
            msa=msa_by_chain,
        )

        if not _check_atom_shape_guard(ref_feats, feats, pos, aa):
            n_skipped_shape += 1
            pd.DataFrame(
                [
                    {
                        "mutation_id": mid,
                        "row_index": n_scored,
                        "sequence": mut_seq,
                        "substitutions": f"{pos}{aa}",
                        "n_mut": 1,
                        "binding_score": float("nan"),
                        "model_score": float("nan"),
                        "wt_score": wt_score,
                        "delta": float("nan"),
                        "mode": "refeaturise_SKIPPED_SHAPE",
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)
            write_header = False
            continue

        with torch.no_grad():
            s_inputs = embedder_only(model, feats)
            scalar = confidence_forward(
                model, s_inputs, feats, x_pred, score_name,
                gradient_checkpointing=False,
            )
            model_score = float(scalar)

        pd.DataFrame(
            [
                {
                    "mutation_id": mid,
                    "row_index": n_scored,
                    "sequence": mut_seq,
                    "substitutions": f"{pos}{aa}",
                    "n_mut": 1,
                    "binding_score": float("nan"),
                    "model_score": model_score,
                    "wt_score": wt_score,
                    "delta": model_score - wt_score,
                    "mode": "refeaturise",
                }
            ],
            columns=_COLUMNS,
        ).to_csv(out, mode="a", header=write_header, index=False)
        write_header = False

        free_cuda_memory()
        n_scored += 1
        rate = (time.time() - t0) / max(n_scored, 1)
        log.info(
            "[%d/%d] pos=%d aa=%s score=%.6f delta=%.6f (%.1fs/mutant, eta %.0f min)",
            n_scored + n_skipped_resume, len(mutant_pairs), pos, aa, model_score,
            model_score - wt_score,
            rate, rate * (len(mutant_pairs) - n_scored - n_skipped_resume) / 60,
        )

    if n_skipped_shape:
        log.warning(
            "Skipped %d mutants due to atom shape mismatch with wild type.",
            n_skipped_shape,
        )

    return n_scored, n_skipped_resume


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
        "--mode",
        default="resswap",
        choices=["resswap", "refeaturise"],
        help=(
            "Scan mode. 'resswap' (default) edits only res_type[pos], keeping "
            "all other features identical to the wild type -- the exact function "
            "the gradient differentiates. 'refeaturise' builds the true mutant "
            "features from scratch, reusing wild-type coordinates."
        ),
    )
    p.add_argument(
        "--x-pred",
        default="predicted",
        choices=["zeros", "predicted"],
        help="x_pred mode. Default: predicted (reuse stage 03's cached structure).",
    )
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
                "--dry-run: would score %d mutants with score=%s mode=%s",
                len(mutant_pairs), args.score, args.mode,
            )
            return

    elif is_skempi:
        if args.positions is None:
            raise SystemExit(
                "SKEMPI complexes require --positions to specify which positions "
                "to scan (e.g. --positions skempi for SKEMPI-measured positions)."
            )

        pdb = resolved.pdb_path
        reference_seq = resolved.reference_seq
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
            "SKEMPI brute-force scan: %d positions, %d mutant pairs, mode=%s",
            len(positions), len(mutant_pairs), args.mode,
        )

        # Resume: key on mutation_id
        done_ids = _load_done_ids(out)
        if done_ids:
            log.info("Resuming: %d mutations already scored in %s", len(done_ids), out)
        todo = [(p, a) for p, a in mutant_pairs if _mutation_id(p, a) not in done_ids]
        if not todo:
            log.info("Nothing to do; all %d mutant pairs already scored.", len(mutant_pairs))
            return

        if args.dry_run:
            log.info(
                "--dry-run: would score %d mutants (%d remaining) with score=%s mode=%s",
                len(mutant_pairs), len(todo), args.score, args.mode,
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
        done: set[int] = _load_done_row_indices(out)
        if done:
            log.info("Resuming: %d rows already scored in %s", len(done), out)
        todo_abbibench = [i for i in idx if i not in done]
        if not todo_abbibench:
            log.info("Nothing to do; all %d sampled rows already scored.", len(idx))
            return

        if args.dry_run:
            log.info(
                "--dry-run: would score %d mutants with score=%s",
                len(todo_abbibench), args.score,
            )
            return

    require_vram()

    from igv.attrib import free_cuda_memory
    from igv.boltz_score import (
        SCORES,
        build_complex_feats,
        confidence_forward,
        embedder_only,
        load_model,
        numerics_arm,
        predict_structure_coords,
    )
    import torch

    if args.score not in SCORES:
        raise SystemExit(f"--score must be one of {sorted(SCORES)}")

    scan_suffix = f"_{args.dataset}"

    if is_dms or is_skempi:
        model, _boltz_version = load_model(args.checkpoint_dir, args.device)

        def chains_for(seq: str) -> dict[str, str]:
            d = dict(struct_chains)
            d[args.chain] = seq
            return d

        # Pin the wild-type MSA from stage 03
        msa_by_chain = _resolve_msa_by_chain(
            cache_dir, args.dataset, args.chain, struct_chains,
        )
        use_msa_server = not args.no_msa_server and msa_by_chain is None
        if msa_by_chain is not None:
            use_msa_server = False

        ref_feats, token_map = build_complex_feats(
            chains_for(reference_seq),
            pdb,
            cache_dir / f"boltz_ref{scan_suffix}",
            args.device,
            use_msa_server=use_msa_server,
            msa=msa_by_chain,
        )

        # Resolve x_pred
        if args.x_pred == "predicted":
            x_pred = predict_structure_coords(
                model, ref_feats, cache_dir, struct_chains, args.dataset,
            )
        else:
            x_pred = ref_feats["coords"].detach()

        _coords_max = float(x_pred.abs().max())
        log.info(
            "x_pred (mode=%s, shape %s), coords.abs().max()=%g",
            args.x_pred, list(x_pred.shape), _coords_max,
        )
        if args.x_pred == "predicted" and _coords_max == 0.0:
            raise RuntimeError(
                "x_pred mode is 'predicted' but predicted coordinates are all zeros."
            )

        # Wild-type score (row 0)
        wt_score: float
        wt_done = _load_done_ids(out)
        if "WT" in wt_done:
            prev = pd.read_csv(out)
            wt_row = prev[prev["mutation_id"] == "WT"]
            wt_score = float(wt_row.iloc[0]["model_score"])
            log.info("Wild-type score (from resume): %.6f", wt_score)
        else:
            with torch.no_grad():
                s_inputs = embedder_only(model, ref_feats)
                scalar = confidence_forward(
                    model, s_inputs, ref_feats, x_pred, args.score,
                    gradient_checkpointing=False,
                )
                wt_score = float(scalar)
            log.info("Wild-type %s = %.6f", args.score, wt_score)

            write_header = not out.exists()
            pd.DataFrame(
                [
                    {
                        "mutation_id": "WT",
                        "row_index": -1,
                        "sequence": reference_seq,
                        "substitutions": "",
                        "n_mut": 0,
                        "binding_score": float("nan"),
                        "model_score": wt_score,
                        "wt_score": wt_score,
                        "delta": 0.0,
                        "mode": args.mode,
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)

        # Build the pair list (with resume already applied for SKEMPI)
        if is_dms:
            done_ids = _load_done_ids(out)
            pairs_to_scan = [(p, a) for p, a in mutant_pairs
                             if _mutation_id(p, a) not in done_ids]
        else:
            done_ids = _load_done_ids(out)
            pairs_to_scan = [(p, a) for p, a in mutant_pairs
                             if _mutation_id(p, a) not in done_ids]

        if not pairs_to_scan:
            log.info("Nothing to do; all mutant pairs already scored.")
        elif args.mode == "resswap":
            n_scored, n_skipped = _resswap_scan(
                model, ref_feats, x_pred, args.score, reference_seq,
                args.chain, token_map, positions, out, pairs_to_scan,
                done_ids, wt_score,
            )
        else:
            n_scored, n_skipped = _refeaturise_scan(
                model, ref_feats, x_pred, args.score, reference_seq,
                args.chain, struct_chains, pdb, cache_dir, args.dataset,
                args.device, positions, out, pairs_to_scan, done_ids,
                wt_score, use_msa_server, msa_by_chain,
            )

        prov_write(
            out,
            stage="04_scan",
            inputs={"dataset": args.dataset, "structure": str(pdb)},
            params={
                "n_sample": len(mutant_pairs),
                "seed": args.seed,
                "chain": args.chain,
                "sampling": "skempi_brute_force" if is_skempi else "dms_interface_exhaustive",
                "mode": args.mode,
                "x_pred": args.x_pred,
            },
            arm={
                "score": args.score,
                "method": "scan",
                "trunk": "forward_only",
                "geometry": f"x_pred_{args.x_pred}",
                "dataset": args.dataset,
                "chain": args.chain,
                "mode": args.mode,
                **numerics_arm(),
            },
            notes=f"Brute-force scan, mode={args.mode}.",
        )
        log.info("Wrote %s", out)
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
        for n, row_i in enumerate(todo_abbibench, 1):
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
                        "mutation_id": str(row_i),
                        "row_index": int(row_i),
                        "sequence": seq,
                        "substitutions": _fmt_subs(lib.substitutions[row_i]),
                        "n_mut": int(row["n_mut"]),
                        "binding_score": float(row["binding_score"]),
                        "model_score": model_score,
                        "wt_score": float("nan"),
                        "delta": float("nan"),
                        "mode": "refeaturise",
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)
            write_header = False

            free_cuda_memory()
            rate = (time.time() - t0) / n
            log.info(
                "[%d/%d] row=%d n_mut=%d score=%.6f (%.1fs/mutant, eta %.0f min)",
                n, len(todo_abbibench), row_i, int(row["n_mut"]), model_score,
                rate, rate * (len(todo_abbibench) - n) / 60,
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
        log.info(
            "Wrote %s (%d new rows, %.1f min)",
            out, len(todo_abbibench), (time.time() - t0) / 60,
        )


if __name__ == "__main__":
    main()
