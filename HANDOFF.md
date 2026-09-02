# Handoff — 2026-09-02

State of the "can one backward pass replace an alanine scan?" project.
Written for a fresh session with no prior context.

---

## 1. What this project is

**Question.** Can a single gradient computation on Boltz-2 substitute for a
brute-force mutation scan, when scored against real measured binding affinities?

**Method.** Take the gradient of a Boltz-2 confidence score with respect to the
input sequence embedding, dot it with per-substitution embedding deltas to get a
predicted effect for every possible mutation, and score those predictions
against AbBiBench's 184,500 measured antibody affinities.

**The contribution is a three-term decomposition**, not the headline number.
This matters: a weak headline is still publishable because the decomposition
localises *where* the failure is.

| Term | Compares | Answers |
|---|---|---|
| **T1** | attribution vs Boltz-2's own mutation scan | Is the gradient faithful to the model? |
| **T2** | Boltz-2's scan vs experiment | Is the model right about biology? |
| **T3** | attribution vs experiment | What does a practitioner actually get? |

T3 uses all 184,500 mutants (one backward pass covers them all). Only T1/T2 need
the expensive scan, subsampled to ~300 mutants per dataset.

**Why the efficiency claim works here.** ~16 backward passes replace ~615
GPU-hours of brute-force forward passes. On the earlier peptide framing the
saving was only ~3.4×, which motivated nothing.

**Bar to beat** (AbBiBench leaderboard, avg per-dataset Spearman): Boltz-2 0.13,
FoldX 0.12, AF3 −0.02; leaders ProteinMPNN 0.30, ESM-IF1 0.28, AntiFold 0.21.
Datasets `1mlc` and `1n8z` are near-zero-to-negative for *every* model — a low
score there is not evidence against this method.

**Full plan:** `/Users/bowmannovey/.claude/plans/ok-read-the-plan-gentle-nova.md`
**Original proposal:** `PLAN.md` — superseded, and **must not be modified**.

---

## 2. Status

### Complete and verified

- **All 8 pipeline stages written**: `scripts/00_fetch_data.py` …
  `07_sanity.py`, plus `run_all.sh`
- **All 6 library modules**: `src/igv/{provenance,attrib,boltz_score,data,metrics}.py`
- **44/44 CPU tests pass** (`python3 -m pytest tests/ -q`, ~7 s)
- **Docs**: `docs/CLOUD.md`, `docs/PIPELINE.md`
- **Notebooks 01 and 02**: `01_dataset_overview.ipynb`, `02_three_terms.ipynb`
- **GCP infrastructure provisioned and GPU verified** (details in §3)

### Not done

| Gap | Notes |
|---|---|
| `notebooks/03_epistasis.ipynb` | Stratify T3 error by `n_mut` on the two CR9114 combinatorial sets. CPU-only. The most original result available. |
| `notebooks/04_sanity_checks.ipynb` | Render `results/sanity_*.json` from stage 07. CPU-only. |
| Docker image never built successfully | Blocked by a `COPY` bug, now fixed but **not yet retested** — see §4 |
| **Zero GPU stages have run** | No `results/` artifacts exist yet |
| Stage 0 go/no-go | Never reached |

### Never validated (the real unknown)

Whether `torch==2.7.1+cu126` and `boltz==2.2.1` coexist in the container. The
predecessor repo hit torch-clobbering **four separate times**, and it never
announces itself as a torch problem — symptoms are an exit-1 with empty stderr,
or a job that hangs for 7+ hours. `bootstrap.sh`'s verification step checks this
explicitly. **Until it passes, treat the whole environment as unproven.**

---

## 3. Infrastructure

### GCP

| | |
|---|---|
| Project | `agrosbio` |
| Billing account | `017BD0-C84E33-6C9E0B` (open; a second account `014C9F-95B3FE-7C2258` is **closed** — ignore it) |
| Instance | `igv-gpu`, `a2-ultragpu-1g`, `us-central1-a` |
| Status | **TERMINATED** (= stopped, not deleted) |
| Boot disk | 500 GB pd-ssd, `READY`, **preserved** |
| GPU verified | `NVIDIA A100-SXM4-80GB`, 81920 MiB, driver `580.173.02` |
| Host | 12 vCPU, 167 GB RAM, 468 GB free on `/` |

Credits pay for this. Credits apply before the card, automatically, but they
**expire by date** and when they run out billing silently rolls to the card.
Budget alerts only notify; they do not cap.

### Quota (approved 2026-09-02, region `us-central1`)

```
NVIDIA_A100_80GB_GPUS              limit=1   usage=0
A2_CPUS                            limit=12  usage=0
NVIDIA_A100_GPUS       (40GB)      limit=1
PREEMPTIBLE_NVIDIA_A100_GPUS       limit=16
PREEMPTIBLE_NVIDIA_A100_80GB_GPUS  limit=0
```

`A2_CPUS=12` is exactly what `a2-ultragpu-1g` consumes, so **only one A2 machine
can exist at a time.** A second one — including a Colab Enterprise runtime —
fails with what looks like a capacity error.

### Cost

| Item | Rate |
|---|---|
| `a2-ultragpu-1g` running | **~$5.07/hr** (~$122/day) |
| Stopped, disk only | ~$0.12/hr (~$3/day) |
| Project budget | ~40 GPU-hours ≈ $200 |

Spent so far: ~$1.50.

### Start / stop

```bash
gcloud compute instances start igv-gpu --zone=us-central1-a --project=agrosbio
# ~30 s. External IP changes; connect by name, not IP.

gcloud compute instances stop igv-gpu --zone=us-central1-a --project=agrosbio \
  --discard-local-ssd=true
# --discard-local-ssd is REQUIRED on a2-ultragpu (it has Local SSD attached).
# Safe: our work lives on the boot disk, the Local SSD is unmounted scratch.
```

### SSH

The GCE key `~/.ssh/google_compute_engine` is **passphrase-protected**. Agent
keys are shared via the macOS launchd socket, so if the agent is empty, SSH from
an automated session fails with `Permission denied (publickey)` *after* logging
`Server accepts key` — that specific pattern means "can't sign", not "wrong key".

Fix, run by the user in **Terminal.app** (the prompt cannot be answered from
inside Claude Code):

```bash
ssh-add --apple-use-keychain ~/.ssh/google_compute_engine
```

Verify with `ssh-add -l`.

### Colab Enterprise — evaluated and rejected

APIs are already enabled; nothing to request. Rejected because runtime templates
have **no `--container-image` flag**, so the pinned container can't be used and
you must pip-install over Google's base image — precisely the torch-clobber
risk. Also: mandatory idle shutdown, ephemeral disk, no detached runs. A free
template `igv-a100-40` (ID `691054053173493760`) exists as an unused fallback.

Consumer Colab Pro/Pro+ cannot be paid with GCP credits (separate billing
system). Colab Enterprise can, but draws on the same Compute Engine GPU quota.

---

## 4. Uncommitted changes — four real bugs fixed

```
 M docker/Dockerfile
 M scripts/cloud/bootstrap.sh
 M scripts/cloud/gcp_launch.sh
?? .dockerignore
```

**These are fixed locally but NOT yet on the VM and NOT tested.**

1. **`docker/Dockerfile` — `COPY .. /app`**
   Reaches outside the build context, so Docker copied *nothing* (`#10 DONE
   0.0s`) and `pip install -e .` failed with "does not appear to be a Python
   project". Now `COPY . /app`, built from the repo root with
   `-f docker/Dockerfile`. **This is the bug that stopped the last run.**

2. **`bootstrap.sh` — `get_device_properties(0).total_mem`**
   No such attribute; it is `.total_memory`. Would have raised `AttributeError`
   inside the container, so the verification JSON never printed and bootstrap
   died with the misleading "Verification script produced no parseable output".

3. **`bootstrap.sh` — bare `docker` calls**
   `usermod -aG docker` does not take effect in the session that runs it, so
   `docker build` failed on the socket. Added a probed `$DOCKER` prefix that
   falls back to `sudo docker`.

4. **`gcp_launch.sh` — two stale values**
   Price `$10.04/hr` → `$5.07` (contradicted `docs/CLOUD.md`); image family
   `common-cu126-ubuntu-2204` **no longer exists**. Only these are published now:
   ```
   common-cu129-ubuntu-2204-nvidia-580
   common-cu129-ubuntu-2404-nvidia-580
   pytorch-2-9-cu129-ubuntu-2204-nvidia-580
   pytorch-2-9-cu129-ubuntu-2404-nvidia-580
   ```
   Default is now `common-cu129-ubuntu-2204-nvidia-580`. Host CUDA 12.9 is fine
   for a cu126 container — only the driver needs to be new enough, and 580 is.

Plus new `.dockerignore` (there was none, so the whole repo was build context).

**Commit these before the next GPU run** so provenance sidecars record a clean
commit rather than `git_dirty: true`.

---

## 5. Resume — exact steps

```bash
cd /Users/bowmannovey/Agros/IG

# 0. Load the SSH key (Terminal.app, once, if `ssh-add -l` is empty)
ssh-add --apple-use-keychain ~/.ssh/google_compute_engine

# 1. Commit the four fixes
git add -A && git commit -m "Fix Dockerfile COPY context, total_memory, docker group, stale image family"

# 2. Start the VM (~30 s, ~$5.07/hr from here)
gcloud compute instances start igv-gpu --zone=us-central1-a --project=agrosbio

# 3. Push the fixed files
for f in docker/Dockerfile scripts/cloud/bootstrap.sh scripts/cloud/gcp_launch.sh .dockerignore; do
  gcloud compute scp "$f" "igv-gpu:~/IG/$f" --zone=us-central1-a --project=agrosbio
done

# 4. Bootstrap (detached; torch/boltz layers are already cached)
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='cd ~/IG && nohup bash scripts/cloud/bootstrap.sh > ~/bootstrap.log 2>&1 & echo started'

# 5. Watch it
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='tail -40 ~/bootstrap.log'
```

**Gate:** bootstrap's verification table must show `torch 2.7.1+cu126`,
`cuda_available True`, `gpu_0_vram_gib ≥ 80`, `boltz 2.2.1`, and every
`forbidden_*` row as "not found". If torch is any other version, the container
lost its fight with the base image — stop and fix that before running science.

Then stage 0 on `4fqi_h1`:

```bash
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='cd ~/IG && tmux new -s igv -d "bash scripts/run_all.sh 2>&1 | tee ~/run_all.log"'
```

**Stop the VM when done.** It does not stop itself.

---

## 6. Traps

- **`4fqi_h1` is the go/no-go dataset.** Boltz-2 already scores 0.71 there. If
  attribution finds nothing on it, the method is dead — you learn that in an
  afternoon instead of after the full sweep.
- **Only one A2 machine at a time** (`A2_CPUS=12`). Never run the VM and a Colab
  runtime together.
- **Stage 03 is not resumable.** Do not use spot/preemptible; a preemption kills
  a 20–40 min run with no recovery.
- **Use the path-averaged gradient, never the `(x − baseline) × grad` product**,
  for per-substitution predictions. `attrib.py:206-219` documents why. The
  `(x − baseline)` factor belongs only to the completeness identity.
- **`plain_gradient` is not `integrated_gradient(m_steps=1, "uniform")`** — the
  latter averages α=0 and α=1.
- **`compute_ptms` swallows exceptions** and returns zeros with only a bare
  `print`. `boltz_score.py` asserts non-zero before backward; keep that.
- **Never write to `/Users/bowmannovey/Agros/IG_Agros`** — read-only predecessor
  reference.
- **`PLAN.md` must not be modified.**
- **`signal_control` is the load-bearing sanity check.** Boltz-2 is documented to
  give uniformly high confidence regardless of biological relevance. If the score
  itself doesn't move under mutation, its gradient can't either, and a null
  result is uninterpretable. T1 must show the explicit scan moves before any
  claim about the gradient.
- Trust `assert_provenance` on the recorded `arm`, not on the flags you passed.
  The predecessor's two worst bugs were silent no-ops found only by auditing
  artifacts.

---

## 7. Open questions

- Confirm AbBiBench accepts a predicted score *change* per mutant rather than an
  absolute likelihood. Interface is one float per mutant and Spearman is
  rank-based, so a delta should be fine — but verify.
- `PLAN.md` §5.3 wants attribution on the target chain, but
  `build_mean_aa_baseline` copies receptor positions verbatim, making their
  attribution identically zero by construction. Needs a different baseline.
- Is the ipTM argmax frame residue stable across IG path steps? If it jumps, the
  path integral is summing gradients of a piecewise-different function. Log it
  per step — interesting either way.
- Contamination: which antibodies predate Boltz-2's 2023-06-01 training cutoff.

---

## 8. Venue

MLSB, 5 pages excluding references, NeurIPS style; welcomes work-in-progress.
2025 deadline was Oct 1, so 2026 is plausibly ~5 weeks out. EuroMLSB shares the
portal.

Do **not** claim first-to-validate. Engage and distinguish: TISM (iScience
2024), arXiv:2606.22181 (same logic on pLMs, negative), GearBind (Nat Commun
2024), Vogt et al. (Front Bioinform 2026), ProtDBench, BindEnergyCraft, the AF3
SKEMPI paper (NeurIPS 2024), and Majdandzic (Genome Biology 2023) on off-simplex
gradient contamination.
