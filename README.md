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

Ground truth comes from two sources:
[AbBiBench](https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset)
(184,500 antibody binding affinities) and
[SKEMPI 2.0](https://life.bsc.es/pid/skempi2/) (7,085 measured ddG values
across 348 complexes). The primary target is **1JTG** (TEM-1 / BLIP, 49
measured positions, 13 hot spots); **3HFM** (HyHEL-10 / lysozyme) is the
second validation arm.

| | What we compare | What it tells us |
|---|---|---|
| **shortcut vs slow way** | gradient shortcut vs the model's own brute-force scan | Does the shortcut match the model it came from? |
| **model vs lab** | the model's scan vs real lab measurements | Is the model right about biology at all? |
| **shortcut vs lab** | gradient shortcut vs real lab measurements | What would someone actually get from this? |

The first is the honest test of the shortcut. The second is a property of
Boltz-2 and is already known to be weak (Spearman ~0.13, below ProteinMPNN's
0.30). If it is near zero, the third cannot be good no matter how well the
shortcut works — so the three have to be read together.

## Status

**The memory problem is solved.** Getting the gradient requires a backward pass
through the whole model, which did not fit on an 80 GB GPU at the size we need
(730 tokens). Two settings fix it:

```bash
IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16
```

That brings the requirement from 93.7 GB down to a measured **55.2 GB**, and a
single backward pass at 730 tokens takes 104 seconds. Details and all
measurements: [docs/MEMSCALE_RESULTS.md](docs/MEMSCALE_RESULTS.md).

**The correctness problem is resolved.** The Integrated Gradients completeness
check (attributions must sum to the total score change) failed at m=16
integration steps. The cause was quadrature resolution, not half precision: fp32
fails *worse* than bf16 at the same step count (rel err 1.7205 vs 1.1709). At
m=32, completeness passes with rel err **0.0442** on L=554. bf16 is exonerated.

**The `random_weights` sanity check passes.** Randomising all 5,035 parameter
tensors destroys the attribution (Spearman 0.077 against a 0.3 threshold), so
the gradient depends on what Boltz-2 learned rather than on input geometry.

**No end-to-end result yet.** The pipeline has never produced a shortcut-vs-lab
or shortcut-vs-slow-way number. The next step is seeding the `ref_pos` conformer
generation (which introduces up to 10.7 Angstrom of noise between featurisations)
and running 1JTG on the SKEMPI validation arm.

## Running it

CPU, on a laptop:

```bash
pip install -e ".[dev]"    # or: make setup
make test                  # 351 tests, no GPU needed
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
