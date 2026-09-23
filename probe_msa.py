"""Is the 1.07 embedding difference lossy MSA reuse, or nondeterministic featurisation?

Featurises the SAME reference complex four times and compares embedder outputs:

  A, B : server MSA, two independent cache dirs   -> tests server-path determinism
  C, D : file-loaded MSA, two independent dirs    -> tests file-path determinism
  A vs C                                          -> the comparison that failed

If A vs B is also ~1.0, featurisation is stochastic and every delta
s_mut - s_ref is contaminated by that noise; holding the MSA fixed is then the
fix rather than the fault. If A vs B is ~0 but A vs C is ~1.0, the CSV round
trip is genuinely lossy and reuse needs a different mechanism.

Also prints feats["msa"].shape per run: a differing MSA DEPTH would explain it
outright.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
logging.basicConfig(level=logging.WARNING)

from igv.data import build_library, read_pdb_chains  # noqa: E402
from igv.boltz_score import build_complex_feats, embedder_only, load_model  # noqa: E402

cache = Path("data/raw")
lib = build_library("4fqi_h1", cache, chain="H")
struct = read_pdb_chains(cache / "4fqi_hlab.pdb")
chains = dict(struct)
chains["H"] = lib.reference_seq
pdb = cache / "4fqi_hlab.pdb"
model, _ = load_model("/root/.boltz", "cuda")

def run(tag, msa, use_server):
    d = cache / f"probe_msa/{tag}"
    feats, _ = build_complex_feats(chains, pdb, d, "cuda", use_msa_server=use_server, msa=msa)
    s = embedder_only(model, feats)
    print(f"{tag:12s} msa_shape={tuple(feats[chr(34)+chr(34).join([])+chr(34)] if False else feats[str(chr(109)+chr(115)+chr(97))].shape)}  |s| = {float(s.norm()):.6f}", flush=True)
    return s, d

sA, dA = run("A_server1", None, True)
sB, dB = run("B_server2", None, True)
msa_by_chain = {}
files = sorted((dA / "msa").glob("input_*.csv"), key=lambda p: int(p.stem.rsplit("_",1)[1]))
for cid, f in zip(chains.keys(), files):
    msa_by_chain[cid] = f
print("mapping:", {k: v.name for k, v in msa_by_chain.items()}, flush=True)
sC, _ = run("C_files1", msa_by_chain, False)
sD, _ = run("D_files2", msa_by_chain, False)

def cmp(n1, s1, n2, s2):
    print(f"{n1} vs {n2}: max_abs={float((s1-s2).abs().max()):.6e}  rel={float((s1-s2).norm()/s1.norm()):.6e}", flush=True)

cmp("A_server1", sA, "B_server2", sB)
cmp("C_files1",  sC, "D_files2",  sD)
cmp("A_server1", sA, "C_files1",  sC)
