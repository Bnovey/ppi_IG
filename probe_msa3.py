"""WHICH feats key differs between two featurisations of the identical complex?

probe_msa2 proved the embedder forward is bit-deterministic and that msa,
res_type and token_index are stable -- yet probe_msa fnd embeddings differing by
~1.1-1.5 across separately-featurised runs. So the variation is in some OTHER
feature. Name it, so the next person does not re-derive this.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
logging.basicConfig(level=logging.WARNING)

import torch  # noqa: E402
from igv.data import build_library, read_pdb_chains  # noqa: E402
from igv.boltz_score import build_complex_feats  # noqa: E402

cache = Path("data/raw")
lib = build_library("4fqi_h1", cache, chain="H")
chains = dict(read_pdb_chains(cache / "4fqi_hlab.pdb"))
chains["H"] = lib.reference_seq
pdb = cache / "4fqi_hlab.pdb"

f1, _ = build_complex_feats(chains, pdb, cache / "probe_msa/X1", "cuda", use_msa_server=True)
f2, _ = build_complex_feats(chains, pdb, cache / "probe_msa/X2", "cuda", use_msa_server=True)

print("keys only in one:", set(f1) ^ set(f2), flush=True)
for k in sorted(set(f1) & set(f2)):
    a, b = f1[k], f2[k]
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        continue
    if tuple(a.shape) != tuple(b.shape):
        print("%-24s SHAPE DIFFERS %s vs %s" % (k, tuple(a.shape), tuple(b.shape)), flush=True)
        continue
    if a.dtype != b.dtype:
        print("%-24s DTYPE DIFFERS" % k, flush=True)
        continue
    if a.numel() == 0:
        continue
    if a.dtype.is_floating_point:
        d = float((a - b).abs().max())
    else:
        d = float((a != b).sum())
    if d != 0:
        print("%-24s DIFFERS  metric=%.6g  shape=%s  dtype=%s" % (k, d, tuple(a.shape), a.dtype), flush=True)
print("done", flush=True)
