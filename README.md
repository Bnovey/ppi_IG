# ppi_IG — can one gradient replace a mutation scan?

Antibodies bind to targets, and drug designers want to know which mutations at
the binding interface make that binding stronger or weaker.

The usual way to answer this with a model like Boltz-2 is brute force: change
one amino acid, run the model, write down the score, repeat a few thousand
times. That costs roughly **615 GPU-hours** for the datasets here.

This repo tests a shortcut. Run the model **once**, and instead of just reading
the score, ask the model how the score would change if you nudged each part of
the input. That is a gradient, and in principle it predicts every mutation at
once from a single pass. The technique is
[Integrated Gradients](https://arxiv.org/abs/1703.01365).

**Does the shortcut actually work?** That is the whole question. This is a
validation project, not a tool — the answer may well be no.

## The three questions

Ground truth is [AbBiBench](https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset):
184,500 measured binding affinities across 14 antibodies and 9 antigens.

| | What we compare | What it tells us |
|---|---|---|
| **T1** | gradient shortcut vs the model's own brute-force scan | Does the shortcut match the model it came from? |
| **T2** | the model's scan vs real lab measurements | Is the model right about biology at all? |
| **T3** | gradient shortcut vs real lab measurements | What would someone actually get from this? |

T1 is the honest test of the shortcut. T2 is a property of Boltz-2 and is
already known to be weak (Spearman ~0.13, below ProteinMPNN's 0.30). If T2 is
near zero, T3 cannot be good no matter how well the shortcut works — so the
three terms have to be read together.

## Status

**The memory problem is solved.** Getting the gradient requires a backward pass
through the whole model, which did not fit on an 80 GB GPU at the size we need
(730 tokens). Two settings fix it:

```bash
IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16
```

That brings the requirement from 93.7 GB down to a measured **55.2 GB**, and
runs faster than before. Details and all measurements:
[docs/MEMSCALE_RESULTS.md](docs/MEMSCALE_RESULTS.md).

**The correctness problem is open.** With the memory fixed, the main
correctness check could finally run — and it fails. Integrated Gradients
guarantees that the individual attributions add up to the total change in
score; ours overshoot by 4.6x. We do not yet know whether that is caused by the
half-precision setting above, by too few integration steps, or by a poor
choice of reference point. **Until that is resolved, no result from this
pipeline should be trusted.**

Also unresolved: the model is being scored on a structure where every atom sits
at the origin, because the input is built from sequence only. Fixed geometry,
but not the real structure.

## Running it

CPU, on a laptop:

```bash
pip install -e ".[dev]"    # or: make setup
make test                  # 236 tests, no GPU needed
make fetch library         # download the data, build the mutant list
```

GPU (stages 02, 03, 04, 07 and the `08_memscale` diagnostic):

```bash
docker build -t igv -f docker/Dockerfile .
docker run --gpus all --shm-size=32g --ipc=host \
  -e IGV_TRI_ATTN_CKPT=1 -e IGV_AUTOCAST=bf16 \
  -v $HOME/boltz_cache:/root/.boltz -v $(pwd):/app -w /app igv \
  bash scripts/run_all.sh
```

`make help` lists every stage. `--dry-run` works on most of them.

One 80 GB GPU is enough with the two settings above. Without them you need
about 94 GB, which no single card of that class has.

## Two warnings

**Do not add packages.** Only `torch==2.7.1+cu126` and `boltz==2.2.1` work.
Installing `boltzgen`, `protenix`, `chai_lab`, `gnina`, or `cuequivariance`
quietly replaces the pinned CUDA build of torch and then segfaults on A100.
See [constraints.txt](constraints.txt).

**Scores from `IGV_AUTOCAST=bf16` runs cannot be compared with older ones.**
Half precision shifts the numbers slightly, so any reference score has to be
recomputed under the same setting. Every artifact records which settings
produced it.

## Docs

- [docs/PIPELINE.md](docs/PIPELINE.md) — the stages, what each one writes, and how provenance works
- [docs/MEMSCALE_RESULTS.md](docs/MEMSCALE_RESULTS.md) — measured memory requirements, and the correctness failure
- [docs/MEMORY.md](docs/MEMORY.md) — why the backward pass is so large, and what each `IGV_*` setting does
- [docs/CLOUD.md](docs/CLOUD.md) — renting a suitable GPU
- [ERRORS_LOG.md](ERRORS_LOG.md) — every failure so far, with its real cause
- [PLAN.md](PLAN.md) — the research plan
