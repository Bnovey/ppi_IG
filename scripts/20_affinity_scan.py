#!/usr/bin/env python3
"""Stage 20 -- brute-force affinity scan (the model's own answer, B).

For each measured mutant in the Hauser 2018 ABL1 TKI resistance dataset,
mutate ``res_type`` in the featurised wild-type complex and run a FORWARD
pass through Boltz-2's affinity head to get ``affinity_pred_value``.

    B  =  score(mutant) - score(wild_type)

No backward pass, no IG.  This is cheap (~21 forward passes for imatinib)
and is what makes the attribution result interpretable: A-vs-C alone is
unfalsifiable because it is the product of surrogate fidelity (A-vs-B) and
model biology (B-vs-C).

Sign convention
~~~~~~~~~~~~~~~
``affinity_pred_value`` is log10(IC50 in uM), so HIGHER = weaker binding.
Hauser's ``DDG_exp`` is positive = resistance = weaker binding.  They are
already on the same convention, so ``delta_score`` and ``DDG_exp`` should
correlate POSITIVELY with no negation.

Featurise-once strategy
~~~~~~~~~~~~~~~~~~~~~~~
The complex is featurised ONCE as wild type.  Each mutant is created by
editing ``res_type`` in place (zero the row, set the mutant AA column to 1).
This is correct because ``AffinityCropper`` selects tokens by distance from
``center_coords`` (the INPUT structure's coordinates, not ``x_pred``), so
with geometry fixed at one PDB the crop is identical for WT and every mutant.
Re-featurising per mutant would risk a different index space and silently
make ``score(mut) - score(wt)`` a comparison across two different protein
subsets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.abl import (  # noqa: E402
    ABL_TKIS,
    TKI_SMILES,
    load_hauser2018,
    load_hauser2018_from_csv,
    measured,
    parse_mutation,
    xtal_only,
)
from igv.attrib import CANONICAL_AMINO_ACIDS  # noqa: E402

log = logging.getLogger("affinity_scan")

CSV_COLUMNS = [
    "tki",
    "mutation",
    "resnum",
    "wt_aa",
    "mut_aa",
    "token_index",
    "wt_score",
    "mut_score",
    "delta_score",
    "crop_survived",
    "DDG_exp",
]

AA_LIST = list(CANONICAL_AMINO_ACIDS)


# ---------------------------------------------------------------------------
# Pure helpers (CPU, no torch, no boltz -- testable and usable in --dry-run)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--protein-sequence", default=None,
        help="Protein sequence (264-aa ABL1 235-498 construct). Required unless --dry-run.",
    )
    p.add_argument(
        "--ligand-smiles", default=None,
        help="SMILES string for the ligand.",
    )
    p.add_argument(
        "--ligand-ccd", default=None,
        help="CCD code for the ligand.",
    )
    p.add_argument(
        "--pdb", default=None,
        help="PDB file with real coordinates (e.g. data/raw/2hyy.pdb).",
    )
    p.add_argument(
        "--resnum-offset", type=int, default=0,
        help="Added to 0-based residue indices to get PDB author numbering "
             "(e.g. 235 for ABL1 kinase domain W235-Q498).",
    )
    p.add_argument(
        "--score", default="affinity_pred_value",
        choices=["affinity_pred_value", "affinity_logits_binary"],
    )
    p.add_argument(
        "--tki", default="imatinib",
        help="TKI to scan. 'all' scans all 6 ABL TKIs. Default: imatinib.",
    )
    p.add_argument(
        "--hauser-csv", default=None,
        help="Path to pre-fetched Hauser 2018 CSV (offline mode).",
    )
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    p.add_argument("--no-msa-server", action="store_true")
    p.add_argument(
        "--out", default="results/affinity_scan_{tki}.csv",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def _resolve_tkis(tki_arg: str) -> list[str]:
    if tki_arg == "all":
        return list(ABL_TKIS)
    if tki_arg not in TKI_SMILES:
        raise SystemExit(
            f"Unknown TKI {tki_arg!r}. Available: {sorted(TKI_SMILES)} or 'all'."
        )
    return [tki_arg]


def mutate_res_type(
    res_type,
    token_index: int,
    wt_aa: str,
    mut_aa: str,
    aa_col_indices,
):
    """Mutate a single position in the res_type tensor (in-place modification of a clone).

    Returns a modified clone. Asserts the current one-hot decodes to the
    expected wild-type amino acid before mutating.

    Parameters
    ----------
    res_type : Tensor (1, L, V)
        The one-hot res_type tensor (will NOT be modified; a clone is returned).
    token_index : int
        Crop-local token index of the position to mutate.
    wt_aa : str
        Expected wild-type amino acid (single letter).
    mut_aa : str
        Target mutant amino acid (single letter).
    aa_col_indices : array-like, shape (20,)
        Column indices for the 20 canonical AAs, in CANONICAL_AMINO_ACIDS order.

    Returns
    -------
    Tensor (1, L, V)
        A cloned tensor with the mutation applied.
    """
    import torch

    aa_to_col = {aa: int(aa_col_indices[i]) for i, aa in enumerate(AA_LIST)}

    wt_col = aa_to_col.get(wt_aa)
    if wt_col is None:
        raise ValueError(f"Wild-type AA {wt_aa!r} not in canonical amino acids")
    mut_col = aa_to_col.get(mut_aa)
    if mut_col is None:
        raise ValueError(f"Mutant AA {mut_aa!r} not in canonical amino acids")

    row = res_type[0, token_index]
    active_col = int(torch.argmax(row).item())
    if active_col != wt_col:
        actual_aa = "?"
        for aa, col in aa_to_col.items():
            if col == active_col:
                actual_aa = aa
                break
        raise AssertionError(
            f"Wild-type assertion failed at token {token_index}: "
            f"expected {wt_aa} (col {wt_col}) but found {actual_aa} (col {active_col}). "
            f"The token map or resnum_offset is wrong."
        )

    mutated = res_type.clone()
    mutated[0, token_index] = 0.0
    mutated[0, token_index, mut_col] = 1.0
    return mutated


def _load_done_keys(out_path: Path) -> set[str]:
    """Load already-scored (tki, mutation) keys from a CSV for resumability."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['tki']}|{row['mutation']}"
            done.add(key)
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_dir = Path(args.cache_dir)
    tkis = _resolve_tkis(args.tki)

    # Load Hauser ground truth
    if args.hauser_csv is not None:
        df_all = load_hauser2018_from_csv(Path(args.hauser_csv))
    else:
        df_all = load_hauser2018(cache_dir)
    df_measured = xtal_only(measured(df_all))

    # Build the mutation plan per TKI
    plan: list[dict] = []
    for tki in tkis:
        df_tki = df_measured[df_measured["TKI"] == tki].reset_index(drop=True)
        for _, row in df_tki.iterrows():
            mut = parse_mutation(row["mutation"])
            plan.append({
                "tki": tki,
                "mutation": str(mut),
                "resnum": mut.resnum,
                "wt_aa": mut.wt_aa,
                "mut_aa": mut.mut_aa,
                "DDG_exp": float(row["DDG_exp"]),
            })

    log.info("Scan plan: %d mutations across %d TKI(s)", len(plan), len(tkis))

    # Dry run
    if args.dry_run:
        print("Affinity scan: brute-force forward-pass scan on the affinity head\n")
        print(f"  TKI(s):          {', '.join(tkis)}")
        print(f"  mutations:       {len(plan)}")
        print(f"  score:           {args.score}")
        print(f"  pdb:             {args.pdb}")
        print(f"  resnum_offset:   {args.resnum_offset}")
        print(f"  device:          {args.device}")
        print()
        for entry in plan:
            print(f"  {entry['tki']:12s} {entry['mutation']:6s} resnum={entry['resnum']} DDG_exp={entry['DDG_exp']:+.2f}")
        print()
        distinct_muts = {(e["resnum"], e["wt_aa"], e["mut_aa"]) for e in plan}
        print(f"  Distinct mutations (position+AA): {len(distinct_muts)}")
        print(f"  Forward passes needed: {len(distinct_muts)} mutants + 1 wild-type = {len(distinct_muts) + 1}")
        return 0

    # Validate inputs
    if args.protein_sequence is None:
        raise SystemExit("--protein-sequence is required (not in --dry-run mode)")
    if args.ligand_smiles is None and args.ligand_ccd is None:
        raise SystemExit("One of --ligand-smiles / --ligand-ccd is required")
    if args.ligand_smiles is not None and args.ligand_ccd is not None:
        raise SystemExit("Only one of --ligand-smiles / --ligand-ccd may be given")

    # GPU path
    from igv.gpu import require_vram

    require_vram()

    import torch

    from igv.boltz_score import (
        AFFINITY_SCORES,
        affinity_forward,
        build_affinity_feats,
        canonical_aa_token_indices,
        check_crop_coverage,
        enable_confidence_checkpointing,
        load_affinity_model,
        numerics_arm,
    )
    from igv.provenance import write as prov_write

    if args.score not in AFFINITY_SCORES:
        raise SystemExit(f"--score must be one of {sorted(AFFINITY_SCORES)}")

    protein_chains = {"A": args.protein_sequence}
    pdb_path = Path(args.pdb) if args.pdb else None

    # Load model
    model, boltz_version = load_affinity_model(args.checkpoint_dir, args.device)
    enable_confidence_checkpointing(model)

    # Featurise ONCE as wild type
    cache_suffix = "_affinity_scan"
    feats, token_map = build_affinity_feats(
        protein_chains,
        "B",
        cache_dir / f"boltz_affinity{cache_suffix}",
        args.device,
        ligand_smiles=args.ligand_smiles,
        ligand_ccd=args.ligand_ccd,
        use_msa_server=not args.no_msa_server,
        structure_pdb=pdb_path,
        resnum_offset=args.resnum_offset,
    )

    n_tokens = int(feats["token_pad_mask"][0].sum().item())
    log.info("Featurised: %d tokens after crop, token_map has %d entries", n_tokens, len(token_map))

    aa_col_indices = canonical_aa_token_indices()

    # Check crop coverage for all measured positions
    all_resnums = sorted({e["resnum"] for e in plan})
    coverage = check_crop_coverage(token_map, chain_id="0", sites=all_resnums)
    log.info(
        "Crop coverage: %d/%d measured positions surviving",
        coverage["n_surviving"], coverage["n_requested"],
    )
    if coverage["cropped"]:
        log.warning("Positions lost to crop: %s", coverage["cropped"])

    # Wild-type forward pass (once)
    res_type_wt = feats["res_type"].detach().clone()
    x_pred = feats["coords"].detach()

    with torch.no_grad():
        s_inputs_wt = model.input_embedder(feats, affinity=True)
        wt_scalar = affinity_forward(
            model, s_inputs_wt, feats, x_pred, args.score,
            gradient_checkpointing=False,
            recycling_steps=1,
        )
        wt_score = float(wt_scalar)

    log.info("Wild-type %s = %.6f", args.score, wt_score)

    # Scan each TKI
    for tki in tkis:
        out_path = Path(args.out.format(tki=tki))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        done_keys = _load_done_keys(out_path)

        tki_plan = [e for e in plan if e["tki"] == tki]

        # Cache: distinct mutations share forward passes across TKIs
        mut_cache: dict[str, float] = {}
        write_header = not out_path.exists()

        t0 = time.time()
        n_scored = 0
        n_skipped_crop = 0
        n_skipped_resume = 0

        for entry in tki_plan:
            row_key = f"{entry['tki']}|{entry['mutation']}"
            if row_key in done_keys:
                n_skipped_resume += 1
                continue

            resnum = entry["resnum"]
            key = (str("0"), resnum)
            if key not in token_map:
                log.info(
                    "SKIP %s %s: resnum %d not in crop (cropped out)",
                    tki, entry["mutation"], resnum,
                )
                row_data = {
                    "tki": tki,
                    "mutation": entry["mutation"],
                    "resnum": resnum,
                    "wt_aa": entry["wt_aa"],
                    "mut_aa": entry["mut_aa"],
                    "token_index": "",
                    "wt_score": wt_score,
                    "mut_score": "",
                    "delta_score": "",
                    "crop_survived": False,
                    "DDG_exp": entry["DDG_exp"],
                }
                with open(out_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(row_data)
                n_skipped_crop += 1
                continue

            token_idx = token_map[key]
            mut_key = f"{entry['wt_aa']}{resnum}{entry['mut_aa']}"

            if mut_key in mut_cache:
                mut_score = mut_cache[mut_key]
            else:
                mutated_rt = mutate_res_type(
                    res_type_wt, token_idx,
                    entry["wt_aa"], entry["mut_aa"],
                    aa_col_indices,
                )
                feats["res_type"] = mutated_rt
                with torch.no_grad():
                    s_inputs_mut = model.input_embedder(feats, affinity=True)
                    mut_scalar = affinity_forward(
                        model, s_inputs_mut, feats, x_pred, args.score,
                        gradient_checkpointing=False,
                        recycling_steps=1,
                    )
                    mut_score = float(mut_scalar)
                mut_cache[mut_key] = mut_score
                feats["res_type"] = res_type_wt

            delta = mut_score - wt_score

            row_data = {
                "tki": tki,
                "mutation": entry["mutation"],
                "resnum": resnum,
                "wt_aa": entry["wt_aa"],
                "mut_aa": entry["mut_aa"],
                "token_index": token_idx,
                "wt_score": wt_score,
                "mut_score": mut_score,
                "delta_score": delta,
                "crop_survived": True,
                "DDG_exp": entry["DDG_exp"],
            }

            with open(out_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row_data)

            n_scored += 1
            elapsed = time.time() - t0
            log.info(
                "[%s] %d/%d %s mut_score=%.6f delta=%.6f DDG_exp=%.2f (%.1fs elapsed)",
                tki, n_scored, len(tki_plan) - n_skipped_resume,
                entry["mutation"], mut_score, delta, entry["DDG_exp"], elapsed,
            )

        wall = time.time() - t0
        log.info(
            "%s done: %d scored, %d cropped-out, %d resumed, %.1f min",
            tki, n_scored, n_skipped_crop, n_skipped_resume, wall / 60,
        )

        prov_write(
            out_path,
            stage="20_affinity_scan",
            inputs={
                "protein_sequence_length": len(args.protein_sequence),
                "ligand_smiles": args.ligand_smiles,
                "ligand_ccd": args.ligand_ccd,
                "structure_pdb": str(pdb_path) if pdb_path else None,
                "tki": tki,
            },
            params={
                "score": args.score,
                "resnum_offset": args.resnum_offset,
                "n_mutations": len(tki_plan),
                "n_scored": n_scored,
                "n_cropped_out": n_skipped_crop,
            },
            arm={
                "score": args.score,
                "method": "brute_force_affinity_scan",
                "trunk": "forward_only",
                "geometry": "fixed_from_featurisation",
                "tki": tki,
                **numerics_arm(),
            },
            notes=(
                "Brute-force affinity scan: forward pass per mutant, "
                "featurised once as wild type, res_type edited in place."
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
