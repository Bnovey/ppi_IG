# igv -- Integrated-gradient validation for Boltz-2

Can a single backward pass through Boltz-2's affinity head predict the effect
of every interface mutation, replacing a brute-force scan that costs ~615
GPU-hours? Ground truth: AbBiBench (184,500 measured affinities, 14 antibodies,
9 antigens).

## Three-term decomposition

| Term | Compares | Answers |
|---|---|---|
| **T1** | attribution vs the model's own mutation scan | Is the gradient faithful to the model? |
| **T2** | the model's scan vs experiment | Is the model right about biology? |
| **T3** | attribution vs experiment | What does a practitioner actually get? |

The efficiency claim: one backward pass per system (~16 total) predicts every
mutation, versus ~615 GPU-hours of brute-force forward passes.

## Quickstart (CPU, laptop)

```bash
pip install -e ".[dev]"        # or: make setup
make test                      # 43 unit tests, no GPU needed
make fetch library             # download AbBiBench, build mutant library
jupyter notebook notebooks/01_dataset_overview.ipynb
```

## GPU steps

Stages 02 (embed deltas), 03 (attribution), 04 (mutation scan), and
07 (sanity checks) require a GPU with **>=80 GB VRAM** -- full-trunk Boltz-2
backprop fills an entire 80 GB card.

Recommended instances: AWS `p4de.24xlarge` / `p5.48xlarge`, GCP
`a2-ultragpu` / `a3-highgpu`. Containers need `--shm-size=32g --ipc=host`.

```bash
docker build -t igv docker/
docker run --gpus all --shm-size=32g --ipc=host igv

# Inside the container (or on a bare-metal GPU box):
bash scripts/run_all.sh                          # full pipeline
DRY_RUN=1 bash scripts/run_all.sh                # preview commands
DATASET=4fqi_h3 SCORE=complex_pde bash scripts/run_all.sh  # override
```

## Dependency warning

Only `torch==2.7.1+cu126` and `boltz==2.2.1` are supported. **Never** install
`boltzgen`, `protenix`, `chai_lab`, `gnina`, or `cuequivariance` -- they
silently replace the pinned CUDA torch wheel and segfault on A100/sm_80. See
`constraints.txt`.

## Published baselines (per-dataset Spearman, averaged)

| Method | Spearman |
|---|---|
| ProteinMPNN | 0.30 |
| ESM-IF1 | 0.28 |
| AntiFold | 0.21 |
| **Boltz-2** (brute-force scan) | **0.13** |
| FoldX | 0.12 |
| AF3 | -0.02 |

Note: `1mlc` and `1n8z` are near-zero for every model.

## Documentation

- Pipeline stages, artifact DAG, and provenance convention: [docs/PIPELINE.md](docs/PIPELINE.md)
- Research plan: [PLAN.md](PLAN.md)
