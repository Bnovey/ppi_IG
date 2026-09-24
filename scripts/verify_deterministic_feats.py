#!/usr/bin/env python3
"""Verify that featurisation is deterministic under the seeding fix.

Run on the VM (where boltz is installed).  Featurises the same input twice
and asserts every feature tensor is identical.  Reports per-tensor max
absolute difference so ``ref_pos`` is visible specifically.

This is modelled on what the deleted ``probe_msa3.py`` did (see
``ERRORS_LOG.md`` entry 18 for its output format).

Usage
-----
    # Default: 1JTG chains A+B, seed 42
    python scripts/verify_deterministic_feats.py

    # Override seed
    IGV_FEAT_SEED=123 python scripts/verify_deterministic_feats.py

    # Custom complex
    python scripts/verify_deterministic_feats.py --complex 3HFM

Exit codes
----------
    0   All feature tensors are byte-identical across the two runs, AND
        ref_pos agrees between a reference and point mutant at shared residues.
    1   At least one check fails.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complex",
        default="1JTG",
        help="SKEMPI complex id (default: 1JTG)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override IGV_FEAT_SEED (default: read env or 42)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device (default: cpu; use 'cuda' on GPU VM)",
    )
    args = parser.parse_args()

    import torch

    from igv.boltz_score import build_complex_feats
    from igv.deterministic import resolve_feat_seed

    seed = resolve_feat_seed(args.seed)
    device = torch.device(args.device)

    # Define the complex chains
    complexes = {
        "1JTG": {
            "A": _load_1jtg_chain_a(),
            "B": _load_1jtg_chain_b(),
        },
        "3HFM": {
            "H": _load_3hfm_chain_h(),
            "L": _load_3hfm_chain_l(),
            "Y": _load_3hfm_chain_y(),
        },
    }
    if args.complex not in complexes:
        print(f"Unknown complex {args.complex!r}. Known: {sorted(complexes)}")
        sys.exit(1)

    chains = complexes[args.complex]
    print(f"Complex: {args.complex}")
    print(f"Chains: {list(chains.keys())}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print()

    # Featurise twice with the same seed, each in a fresh cache dir
    feats_list = []
    for run_i in range(2):
        cache_dir = Path(tempfile.mkdtemp(prefix=f"verify_feat_run{run_i}_"))
        print(f"Run {run_i}: cache_dir = {cache_dir}")
        try:
            feats, _token_map = build_complex_feats(
                chains=chains,
                structure_pdb=Path("/dev/null"),
                cache_dir=cache_dir,
                device=device,
                use_msa_server=False,
                msa="empty",
                feat_seed=seed,
            )
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

        # Detach and move to CPU for comparison
        feats_cpu = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                feats_cpu[k] = v.detach().cpu()
        feats_list.append(feats_cpu)
        print(f"  {len(feats_cpu)} tensor features")

    print()

    # Compare every tensor
    f1, f2 = feats_list
    all_keys = sorted(set(f1) | set(f2))
    any_diff = False

    print(f"{'tensor':<30s} {'shape':>20s} {'dtype':>10s} {'max_abs_diff':>14s} {'status':>8s}")
    print("-" * 86)

    for key in all_keys:
        if key not in f1 or key not in f2:
            print(f"{key:<30s} {'MISSING IN ONE RUN':>54s} {'FAIL':>8s}")
            any_diff = True
            continue

        t1, t2 = f1[key], f2[key]
        if t1.shape != t2.shape:
            print(
                f"{key:<30s} {str(t1.shape):>20s} {str(t1.dtype):>10s} "
                f"{'SHAPE MISMATCH':>14s} {'FAIL':>8s}"
            )
            any_diff = True
            continue

        if t1.numel() == 0:
            # Boltz emits zero-size tensors for absent features (chiral_*,
            # connected_*, contact_* on a protein-only complex). .max() raises
            # on an empty reduction -- the same crash probe_msa3.py hit.
            diff = 0.0
        elif t1.dtype.is_floating_point:
            diff = (t1 - t2).abs().max().item()
        else:
            diff = (t1 != t2).sum().item()

        status = "ok" if diff == 0 else "DIFFERS"
        if status == "DIFFERS":
            any_diff = True

        print(
            f"{key:<30s} {str(tuple(t1.shape)):>20s} {str(t1.dtype):>10s} "
            f"{diff:>14.4f} {status:>8s}"
        )

    print()
    if any_diff:
        print("FAIL: at least one tensor differs between runs.")
        sys.exit(1)
    else:
        print("PASS: all feature tensors are byte-identical across two runs.")

    # -----------------------------------------------------------------
    # Part 2: reference vs point mutant -- shared residues must agree
    # -----------------------------------------------------------------
    print()
    print("=" * 86)
    print("Part 2: reference vs point mutant (shared residues must agree)")
    print("=" * 86)
    print()

    # Make a point mutant: change one residue in the first chain
    first_chain_id = list(chains.keys())[0]
    wt_seq = chains[first_chain_id]
    # Pick a position near the middle; substitute to alanine (unless already A)
    mut_pos = len(wt_seq) // 2
    wt_aa = wt_seq[mut_pos]
    mut_aa = "A" if wt_aa != "A" else "G"
    mut_seq = wt_seq[:mut_pos] + mut_aa + wt_seq[mut_pos + 1:]
    mut_chains = dict(chains)
    mut_chains[first_chain_id] = mut_seq
    print(f"Mutating chain {first_chain_id} position {mut_pos}: {wt_aa} -> {mut_aa}")
    print()

    # Featurise reference and mutant
    ref_cache = Path(tempfile.mkdtemp(prefix="verify_ref_"))
    mut_cache = Path(tempfile.mkdtemp(prefix="verify_mut_"))
    try:
        ref_feats, ref_tmap = build_complex_feats(
            chains=chains,
            structure_pdb=Path("/dev/null"),
            cache_dir=ref_cache,
            device=device,
            use_msa_server=False,
            msa="empty",
            feat_seed=seed,
        )
        mut_feats, mut_tmap = build_complex_feats(
            chains=mut_chains,
            structure_pdb=Path("/dev/null"),
            cache_dir=mut_cache,
            device=device,
            use_msa_server=False,
            msa="empty",
            feat_seed=seed,
        )
    finally:
        shutil.rmtree(ref_cache, ignore_errors=True)
        shutil.rmtree(mut_cache, ignore_errors=True)

    ref_pos_ref = ref_feats["ref_pos"].detach().cpu()
    ref_pos_mut = mut_feats["ref_pos"].detach().cpu()

    # Which token was mutated?
    rt_ref = ref_feats["res_type"].detach().cpu()
    rt_mut = mut_feats["res_type"].detach().cpu()
    rt_diff = (rt_ref != rt_mut).any(dim=-1)[0]
    mutated_tokens = rt_diff.nonzero().flatten().tolist()
    print(f"res_type differs at {len(mutated_tokens)} token(s) (expected: 1)"
          f"  -> {mutated_tokens}")

    if ref_pos_ref.shape != ref_pos_mut.shape:
        print(f"ref_pos shapes differ: {ref_pos_ref.shape} vs {ref_pos_mut.shape}")
        print("FAIL: atom counts differ, so element-wise comparison is meaningless.")
        sys.exit(1)

    # A global max is not the question. ref_pos MUST change at the mutated
    # residue -- a different side chain has different atoms. What must NOT
    # happen is contamination of any OTHER residue, because every embedding
    # delta is s_mut - s_ref and leakage there is exactly the defect entry 18
    # describes. Localise the difference via atom_to_token.
    per_atom = (ref_pos_ref - ref_pos_mut).abs().amax(dim=-1)[0]   # (n_atoms,)
    token_of_atom = ref_feats["atom_to_token"].detach().cpu()[0].argmax(dim=-1)

    changed = (per_atom > 0).nonzero().flatten()
    changed_tokens = sorted(set(token_of_atom[changed].tolist()))
    leaked = [t for t in changed_tokens if t not in mutated_tokens]

    print(f"ref_pos global max_abs_diff: {per_atom.max().item():.6e}")
    print(f"ref_pos changed at {len(changed)} atom(s) across "
          f"{len(changed_tokens)} token(s): {changed_tokens[:10]}"
          f"{' ...' if len(changed_tokens) > 10 else ''}")

    if leaked:
        worst = max(per_atom[changed][
            [i for i, a in enumerate(changed) if token_of_atom[a].item() in leaked]
        ].tolist())
        print(f"FAIL: {len(leaked)} token(s) outside the mutation changed, "
              f"max {worst:.4e} A. Leakage contaminates every delta.")
        print(f"      leaked tokens: {leaked[:20]}")
        sys.exit(1)

    print("PASS: ref_pos changes are confined to the mutated residue. "
          "Every other residue is byte-identical.")
    if len(mutated_tokens) != 1:
        print("WARNING: expected exactly 1 res_type difference for a point mutant")
        sys.exit(1)

    sys.exit(0)


# ---------------------------------------------------------------------------
# Placeholder sequences -- replace with actual sequences or fetch from PDB.
# These are the first 20 residues as placeholders; the real script should
# load from data/ or fetch via igv.data.
# ---------------------------------------------------------------------------

def _load_1jtg_chain_a():
    """TEM-1 beta-lactamase chain A from 1JTG."""
    try:
        from igv.data import read_pdb_chains
        chains = read_pdb_chains(Path("data/raw/1JTG.pdb"))
        return chains.get("A", _FALLBACK_1JTG_A)
    except Exception:
        return _FALLBACK_1JTG_A


def _load_1jtg_chain_b():
    """BLIP chain B from 1JTG."""
    try:
        from igv.data import read_pdb_chains
        chains = read_pdb_chains(Path("data/raw/1JTG.pdb"))
        return chains.get("B", _FALLBACK_1JTG_B)
    except Exception:
        return _FALLBACK_1JTG_B


def _load_3hfm_chain_h():
    try:
        from igv.data import read_pdb_chains
        chains = read_pdb_chains(Path("data/raw/3hfm.pdb"))
        return chains.get("H", "EVQLQQSGAE")
    except Exception:
        return "EVQLQQSGAE"


def _load_3hfm_chain_l():
    try:
        from igv.data import read_pdb_chains
        chains = read_pdb_chains(Path("data/raw/3hfm.pdb"))
        return chains.get("L", "DIQMTQTTSS")
    except Exception:
        return "DIQMTQTTSS"


def _load_3hfm_chain_y():
    try:
        from igv.data import read_pdb_chains
        chains = read_pdb_chains(Path("data/raw/3hfm.pdb"))
        return chains.get("Y", "KVFGRCELAA")
    except Exception:
        return "KVFGRCELAA"


# Short fallback sequences so the script can at least run (with msa="empty")
# even without the PDB files.  The real verification uses full-length chains.
_FALLBACK_1JTG_A = (
    "HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRIDAGQEQLGRR"
    "IHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDR"
    "WEPELNEAIPNDERDTTMPVAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKS"
    "GAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)
_FALLBACK_1JTG_B = (
    "ADTLHFTTSQEELHSLHAESTLANIALDSSAKLAEEVKFACFATAQSSIEADKILHNTLNFENAAEFADKK"
    "HVELVHFLPGSAQSTMALPNIDFNGEALSSINLGQCLGPNDTFVHNAVKRGDKILKILEENPNLNLADLT"
    "DKSTAAGFITNQIQFISQ"
)


if __name__ == "__main__":
    main()
