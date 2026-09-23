"""Where does the ~3% embedding nondeterminism come from, and does seeding kill it?

probe_msa.py established that featurising the SAME complex twice gives
embeddings differing by max_abs ~1.1-1.5 (rel ~3%), with identical MSA depth,
and that reusing the written MSA files is no worse than re-querying the server.
That noise floor sits directly on top of every s_mut - s_ref delta.

Two candidate sources, distinguished here:
  (a) the FEATURISATION is stochastic  -> feats tensors themselves differ
  (b) the EMBEDDER FORWARD is stochastic (e.g. boltz subsampling MSA rows per
      forward, or nondeterministic bf16 kernels) -> identical feats, different s

and two candidate fixes: a fixed RNG seed before each forward, and fp32.

The prize: if seeding makes repeated embedder calls bit-identical, then
reference and mutant see the same subsample, the noise CANCELS in the delta,
and stage 02 is trustworthy. If it does not, every embedding delta is
noise-dominated and T3 cannot be computed this way.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
logging.basicConfig(level=logging.WARNING)

import torch  # noqa: E402
from igv.data import build_library, read_pdb_chains  # noqa: E402
from igv.boltz_score import build_complex_feats, embedder_only, load_model  # noqa: E402

cache = Path("data/raw")
lib = build_library("4fqi_h1", cache, chain="H")
struct = read_pdb_chains(cache / "4fqi_hlab.pdb")
chains = dict(struct)
chains["H"] = lib.reference_seq
pdb = cache / "4fqi_hlab.pdb"
model, _ = load_model("/root/.boltz", "cuda")

# One featurisation, reused for every forward: isolates the FORWARD from
# featurisation entirely.
d = cache / "probe_msa/A_server1"
feats, _ = build_complex_feats(chains, pdb, d, "cuda", use_msa_server=True)

def emb(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return embedder_only(model, feats)

print("--- same feats, NO seed ---", flush=True)
s1, s2 = emb(), emb()
print("max_abs=%.6e rel=%.6e" % (float((s1-s2).abs().max()), float((s1-s2).norm()/s1.norm())), flush=True)

print("--- same feats, SEEDED identically ---", flush=True)
s3, s4 = emb(0), emb(0)
print("max_abs=%.6e rel=%.6e" % (float((s3-s4).abs().max()), float((s3-s4).norm()/s3.norm())), flush=True)

print("--- same feats, DIFFERENT seeds ---", flush=True)
s5, s6 = emb(0), emb(1)
print("max_abs=%.6e rel=%.6e" % (float((s5-s6).abs().max()), float((s5-s6).norm()/s5.norm())), flush=True)

# Does a SECOND featurisation of the same input give identical feats?
d2 = cache / "probe_msa/E_refeat"
feats2, _ = build_complex_feats(chains, pdb, d2, "cuda", use_msa_server=True)
same = {}
for k in ("msa", "res_type", "token_index"):
    if k in feats and k in feats2:
        a, b = feats[k], feats2[k]
        same[k] = (tuple(a.shape) == tuple(b.shape)) and bool((a == b).all()) if a.dtype == b.dtype else "dtype"
print("feats identical across two featurisations:", same, flush=True)

# And how big is a REAL single-substitution delta, against that noise?
mut = dict(chains)
pos = 28
mut["H"] = lib.reference_seq[:pos] + "S" + lib.reference_seq[pos+1:]
files = sorted((d / "msa").glob("input_*.csv"), key=lambda p: int(p.stem.rsplit("_",1)[1]))
msa_by_chain = {cid: f for cid, f in zip(chains.keys(), files)}
fm, tm = build_complex_feats(mut, pdb, cache / "probe_msa/F_mut", "cuda",
                             use_msa_server=False, msa=msa_by_chain)
sm = emb_m = None
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
sm = embedder_only(model, fm)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
sr = embedder_only(model, feats)
ti = tm[("H", pos)]
delta = (sm[0, ti] - sr[0, ti])
noise = (s1[0, ti] - s2[0, ti])
print("SUBSTITUTION delta at mutated token: |d|=%.6f  max_abs=%.6f" % (float(delta.norm()), float(delta.abs().max())), flush=True)
print("NOISE          at same token (unseeded): |n|=%.6f  max_abs=%.6f" % (float(noise.norm()), float(noise.abs().max())), flush=True)
print("signal-to-noise at that token: %.3f" % (float(delta.norm())/max(float(noise.norm()), 1e-12)), flush=True)
