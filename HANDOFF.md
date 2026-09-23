# Handoff — operational runbook

How to start the VM, run the pipeline, and avoid known traps.

For the research plan and science context, see `ROADMAP.md`. For the original
proposal, see `PLAN.md` (must not be modified).

---

## 1. Infrastructure

### GCP

| | |
|---|---|
| Project | `agrosbio` |
| Billing account | see `gcloud billing accounts list` (there are two; the second is **closed**, ignore it) |
| Instance | `igv-gpu`, `a2-ultragpu-1g`, `us-central1-a` |
| Status | **TERMINATED** (= stopped, not deleted) |
| Boot disk | 500 GB pd-ssd, `READY`, **preserved** |
| GPU verified | `NVIDIA A100-SXM4-80GB`, 81920 MiB, driver `580.173.02` |
| Host | 12 vCPU, 167 GB RAM, 468 GB free on `/` |

Credits pay for this. Credits apply before the card automatically, but they
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

---

## 2. Resume — exact steps

```bash
cd /Users/bowmannovey/Agros/IG

# 0. Load the SSH key (Terminal.app, once, if `ssh-add -l` is empty)
ssh-add --apple-use-keychain ~/.ssh/google_compute_engine

# 1. Start the VM (~30 s, ~$5.07/hr from here)
gcloud compute instances start igv-gpu --zone=us-central1-a --project=agrosbio

# 2. Push any local changes
for f in docker/Dockerfile scripts/cloud/bootstrap.sh scripts/cloud/gcp_launch.sh .dockerignore; do
  gcloud compute scp "$f" "igv-gpu:~/IG/$f" --zone=us-central1-a --project=agrosbio
done

# 3. Bootstrap (detached; torch/boltz layers are already cached)
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='cd ~/IG && nohup bash scripts/cloud/bootstrap.sh > ~/bootstrap.log 2>&1 & echo started'

# 4. Watch it
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='tail -40 ~/bootstrap.log'
```

**Gate:** bootstrap's verification table must show `torch 2.7.1+cu126`,
`cuda_available True`, `gpu_0_vram_gib >= 80`, `boltz 2.2.1`, and every
`forbidden_*` row as "not found". If torch is any other version, the container
lost its fight with the base image — stop and fix that before running science.

**Stop the VM when done.** It does not stop itself.

---

## 3. Environment facts worth not rediscovering

- **Verification table (passing):** python 3.11.15, torch 2.7.1+cu126,
  cuda_available True, gpu_0_vram_gib 79.2, NVIDIA A100-SXM4-80GB, gpu_count 1,
  boltz 2.2.1, and `boltzgen` / `protenix` / `chai_lab` / `gnina` /
  `cuequivariance` all "not found".
- **The VM's `~/IG` is a real checkout**, synced by shipping a tarball that
  includes `.git` (only 488 KB) and running `git reset --hard`, so provenance
  records a genuine commit. Build the tarball with `COPYFILE_DISABLE=1` on macOS
  or it carries `._*` AppleDouble files that land untracked and flip `git_dirty`.
- **Container runs as root**, so artifacts under `results/` and `data/` come out
  root-owned on the host. `sudo chown` before `gcloud compute scp`.
- **Timings on A100:** model load ~45 s; one Boltz-2 structure prediction
  (600 steps) ~45 s; one scored mutant in `signal_control` ~40 s; a full
  `07_sanity` run ~25 min; one backward pass at L=730 ~104 s; Docker rebuild
  ~30 s when only the `pip install -e` layer is invalidated.
- **Rebuild cheaply:** put new `apt`/`pip` layers *after* the torch and boltz
  installs, or you re-download multiple GB.

### `bootstrap.sh` gotchas

1. **`docker/Dockerfile` — `COPY . /app`**. Must be built from the repo root
   with `-f docker/Dockerfile`. An earlier `COPY .. /app` reached outside the
   build context and silently copied nothing.
2. **`get_device_properties(0).total_memory`** (not `.total_mem`). The wrong
   attribute name produces `AttributeError` inside the container, so the
   verification JSON never prints and bootstrap dies with the misleading
   "Verification script produced no parseable output".
3. **Bare `docker` calls fail** because `usermod -aG docker` does not take
   effect in the session that runs it. `bootstrap.sh` probes for this and
   falls back to `sudo docker`.

### Diagnostics on the VM

`~/IG/probe_mem.py` (VRAM at each stage + OOM traceback) and
`~/IG/probe_snap2.py` (allocation-trace replay attributing live bytes at peak by
call site). Both gitignored. `probe_snap2.py` is the one that turned the memory
investigation from guesswork into measurement.

---

## 4. Traps

- **Only one A2 machine at a time** (`A2_CPUS=12`). Never run the VM and a Colab
  runtime together.
- **Stage 03 is not resumable.** Do not use spot/preemptible; a preemption kills
  a 20–40 min run with no recovery.
- **Use the path-averaged gradient, never the `(x − baseline) × grad` product**,
  for per-substitution predictions. `attrib.py:206-219` documents why. The
  `(x − baseline)` factor belongs only to the completeness identity.
- **`plain_gradient` is not `integrated_gradient(m_steps=1, "uniform")`** — the
  latter averages alpha=0 and alpha=1.
- **`compute_ptms` swallows exceptions** and returns zeros with only a bare
  `print`. `boltz_score.py` asserts non-zero before backward; keep that.
- **Never write to `/Users/bowmannovey/Agros/IG_Agros`** — read-only predecessor
  reference.
- **`PLAN.md` must not be modified.**
- **`signal_control` is the load-bearing sanity check.** Boltz-2 gives uniformly
  high confidence regardless of biological relevance. If the score itself doesn't
  move under mutation, its gradient can't either, and a null result is
  uninterpretable. The check passes (std=1.492e-02, n=30).
- Trust `assert_provenance` on the recorded `arm`, not on the flags you passed.
  The predecessor's two worst bugs were silent no-ops found only by auditing
  artifacts.
- **Do not switch the outer checkpoint to `use_reentrant=False`.** It is not a
  style choice. Reentrant mode is load-bearing for memory — the non-reentrant
  path builds the full forward graph and OOMs immediately. See `ERRORS_LOG.md`
  entry 9.
