"""Measure Boltz-2 trainable params -- CPU only, so it cannot perturb a running sweep.

Settles docs/MEMORY.md section 6: "Model parameters are never frozen", so the
first backward allocates a full fp32 gradient copy of every trainable weight
and it survives free_cuda_memory() for the whole IG loop.

Measured 2026-09-04 on igv-gpu, boltz 2.2.1:
    total params   506,724,992
    requires_grad   25,026,048  (4.9%)  -- all in confidence_module
    fp32 grad copy       0.09 GiB
i.e. boltz already freezes everything outside the confidence head
(boltz2.py:350-357), and the lever is worth 0.09 GiB. Closed as negligible.

Run inside the container WITHOUT --gpus, so it cannot touch a measurement:
    sudo docker run --rm --ipc=host -v $HOME/boltz_cache:/root/.boltz \
        -v $HOME/IG:/app -w /app igv:latest python3 probe_params.py
"""
import logging
import sys

sys.path.insert(0, "src")
logging.basicConfig(level=logging.WARNING)

from igv.boltz_score import load_model  # noqa: E402

model, ver = load_model("/root/.boltz", "cpu")
tot = sum(p.numel() for p in model.parameters())
train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"boltz {ver}")
print(f"total params   {tot:,}")
print(f"requires_grad  {train:,}  ({100 * train / tot:.1f}%)")
print(f"fp32 grad copy {train * 4 / 2**30:.2f} GiB")
print("--- trainable by top-level child ---")
for name, child in model.named_children():
    n = sum(p.numel() for p in child.parameters() if p.requires_grad)
    if n:
        print(f"  {name:34s} {n:>13,}  {n * 4 / 2**30:6.2f} GiB")
