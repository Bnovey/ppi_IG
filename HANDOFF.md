# Handoff — operational runbook

How to start the VM, run the pipeline, and avoid known traps.

For the research plan and science context, see `ROADMAP.md`. For the original
proposal, see `PLAN.md` (must not be modified).

---

## 0. Where we are — 2026-09-27

**Local: `main` at `4962fbc`, tree clean, 549 tests, ruff 11 (all pre-existing).**

**No GPU result is newer than 2026-09-24.** The 2026-09-27 session never ran a
model. The VM was started once for a health check (A100-SXM4-80GB confirmed,
429 GB free) and stopped again about 12 minutes later, roughly $1. The only
measured science in the project remains the 1JTG chain B table in `ROADMAP.md`
section 8 — headline **IG m=32 with `mean_aa` = 0.357**, above a null p95 of
0.313, with a CI of [-0.002, 0.656] that excludes zero by two thousandths. Do
not describe that as replicated; it has not been.

### What the 2026-09-27 session actually did

1. **Rejected GB1, adopted Starr 2020 / 6M0J as the saturation arm.** GB1's
   floor lands on the hot spots (position 27 has all 19 substitutions pinned),
   it couples folding to binding by its own definition, and 1FCC is the wrong
   protein G paralogue. Entries 25-27, `ROADMAP.md` sections 9-10.
2. **Verified the within-position design removes the volume confound** —
   +0.44 pooled to -0.034 within position on Starr, with pooled-on-Starr at
   +0.027, which shows the +0.44 is a property of the alanine-scan *design*.
3. **Established that the folding confound misses the interface.** Binding and
   expression correlate +0.64 pooled but only +0.074 over interface mutants,
   where residualising keeps 100% of the binding variance. Entry 28.
4. **Built the saturation arm:** `src/igv/dms.py`,
   `scripts/11_within_position.py`, `scripts/12_compare.py`.
5. **Fixed four pipeline defects, each of which would have wasted paid GPU
   time.** See section 2.1 — this is the part most worth reading.

### Two numbers corrected this session

- The within-position volume residual **on the 21 interface positions we will
  actually analyse is +0.092 mean / +0.245 median**, not the -0.034 measured
  over all 194. Quote the interface figure next to any interface result.
- **A perfect binding predictor scores +0.355 against expression**, because the
  two readouts are themselves correlated at the interface. So a gradient at
  ~0.35 there is not contamination; the diagnostic is the *gap* between
  prediction-vs-binding and prediction-vs-expression.

### The one thing blocking a GPU run

**The VM's checkout is 9 commits behind local `main`** (it sits at `b3d3763`).
Syncing needs a decision that has not been made: `git push origin main`
(publishes to `github.com:Bnovey/ppi_IG.git`, and backs the work up off the
laptop) versus `gcloud compute scp` straight to the box (nothing leaves, but the
only copies are the laptop and the VM). **Do not assume the push is wanted.**

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
# ~5-6 min in practice, not ~30 s. `start` returns BEFORE sshd is listening --
# the next ssh gets "Connection refused". Poll with the until-loop in section 2.
# External IP changes on every start; connect by name, not IP.

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

# 0. Load the SSH key (Terminal.app, once, if `ssh-add -l` is empty).
#    The trap: ssh reports "Server accepts key" AND "agent contains no
#    identities" -- that is the passphrase not being in the agent, not a
#    permissions problem.
ssh-add --apple-use-keychain ~/.ssh/google_compute_engine

# 1. Start the VM. ~5-6 min in practice, not the ~30 s this file used to claim.
#    Billing starts here at ~$5.07/hr. sshd is NOT up when `start` returns --
#    the first ssh gets "Connection refused". Wait for it, do not sleep blindly:
gcloud compute instances start igv-gpu --zone=us-central1-a --project=agrosbio
until gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
        --command=true >/dev/null 2>&1; do :; done; echo SSH_READY

# 2. Get the code onto the box. DECIDE FIRST -- see section 0.
#    Option A (publishes to GitHub):
#      git push origin main
#      gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
#        --command='cd ~/IG && git pull --ff-only'
#    Option B (nothing leaves): scp the changed files, as the old flow did.
#    Verify either way -- the VM must report the same commit as local HEAD:
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='cd ~/IG && git log --oneline -1'

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

### Running a job

`run_all.sh` is **AbBiBench-only** — stages 00 and 01 do not apply to a SKEMPI
or DMS complex, which is why the 1JTG session was driven stage by stage by hand.
Use `run_complex.sh`, which covers both other arms, skips 00/01, and fails fast
on an unknown dataset name **before** any GPU work:

```bash
# SKEMPI arm: 07 -> 02 -> 03 -> 04 -> 05 -> 12 -> 10
DATASET=1JTG CHAIN=B POSITIONS=skempi bash scripts/run_complex.sh

# DMS/saturation arm: 07 -> 02 -> 03 -> 04 -> 05 -> 12 -> 11
DATASET=spike_rbd CHAIN=E POSITIONS=dms_interface bash scripts/run_complex.sh
```

Add `--dry-run` to print every command without executing. Do that first; it is
free and it is how three of the four defects below were caught.

Env overrides: `SCORE` (default `complex_pde`), `METHOD` (default `ig`),
`M_STEPS` (32), `BASELINE` (`mean_aa`), `POSITIONS`, `INTERFACE_CUTOFF` (5.0).

### Planned job sizes, measured not estimated

| Job | Scope | Note |
|---|---|---|
| Pool 3HFM, 1VFB, 1JRH, 2JEL | 4 complexes | ~$12, ~2.5 h. Takes effective n from 28 to ~128. Do this **before** the scan: if 0.357 does not reproduce, it was noise and that is cheap to learn. |
| 1JTG brute force | 28 positions x 19 = **532 mutants** | ~$13. `ROADMAP.md` said 931; the real number comes from `04_scan.py --dry-run`. |
| 6M0J saturation | 791 tokens | ~1.5-3 h per 32-step IG run. One IG run covers **all 194 positions x 20**; only the scan scales, at 399 forward passes for the 5 A set. |

**Stop the VM when done. It does not stop itself.**

```bash
gcloud compute instances stop igv-gpu --zone=us-central1-a --project=agrosbio \
  --discard-local-ssd=true
```

## 2.1 The four defects fixed on 2026-09-27 — read before trusting the pipeline

Every one of these would have burned paid A100 time, and none of them would have
raised an error at the point of the mistake.

1. **`04_scan.py` had no SKEMPI branch.** `--dataset 1JTG` fell through to the
   AbBiBench path and 404'd on a benchmarking CSV that does not exist. The 404
   was the *lucky* case: a name that happens to exist in AbBiBench would have
   downloaded a different protein and scanned it silently — the same shape as
   the boltz cache trap in entry 24. Resolution now goes registry-first through
   `igv.dms.resolve_pdb_complex`, so fallthrough is impossible rather than
   merely unlikely.
2. **The runner ended on `06_metrics.py`**, which requires columns
   `pred`/`binding_score`/`n_mut` while `05_predict.py` writes
   `position`/`mut_aa`/`score_delta` on these paths. It would have aborted
   *after* stages 02-04 had already spent the hours. Stage 06 is AbBiBench-only.
3. **`07_sanity.py` had no DMS branch** — it stopped at "No structure known for
   spike_rbd", and it is the first stage in the runner, so the saturation arm
   could not start at all.
4. **Stage 04's scan CSV was written and read by nothing.** The most expensive
   GPU stage produced an artifact that was never compared to anything, and
   Phase 4 had no implementation. Now `scripts/12_compare.py`. Entry 29.

**The lesson, which is more reusable than the fixes:** defect 4 survived a
validation pass that walked the stage list confirming *every input is produced
by an earlier stage*. That was true and was never the question. The check that
finds this class of bug is the reverse — **is every output consumed** — and a
produced-but-unread file is invisible to any test that runs a stage in
isolation. Check both directions of the file graph when adding a stage.

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
- **Always pass `--project=agrosbio` to every `gcloud` command.** The local
  default config points at `g-prs-478707`, which has the Compute Engine API
  disabled. A command missing `--project` fails with a confusing
  `PERMISSION_DENIED` and then *offers to enable Compute Engine on that other
  project*. Do not accept that prompt. The project id is recorded nowhere in
  `gcloud config` — only in this file and `docs/CLOUD.md`.
- **`run_all.sh` is AbBiBench-only.** For a SKEMPI or DMS complex use
  `run_complex.sh`. Running `run_all.sh` on either will fail in stage 00 or 01.
- **Dry-run first, every time.** `bash scripts/run_complex.sh --dry-run` is free
  and caught three of the four defects in section 2.1.
- **`complex_pde` is lower-is-better; the other five scores are not.**
  `12_compare.py` derives orientation from the score name
  (`LOWER_IS_BETTER = {"complex_pde"}`). Hardcoding a sign anywhere else will
  invert a conclusion while still producing a plausible number.
- **Starr `bind_avg` is positive for *tighter* binding; SKEMPI ddG is positive
  for *weaker*.** Everything must go through `igv.dms.binding_ddg`, which
  negates onto SKEMPI's convention.
- **Known minor test gap:** nothing pins `interface_positions`' cutoff boundary
  at exactly 5.000 A — a `<=` to `<` mutation survives the suite. Measure-zero
  in practice, recorded rather than fixed.
- **Every claim in this repo's docs should be traceable to a command.** The
  Phase 3 estimate of 931 mutants sat in `ROADMAP.md` for days and was wrong;
  `04_scan.py --dry-run` says 532. Prefer printing the number to estimating it.
