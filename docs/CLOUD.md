# Cloud Provisioning Runbook

Run the IG (Boltz-2 gradient attribution) pipeline on GCP GPU instances.

> **Always pass `--project=<YOUR_PROJECT>` explicitly.** If your local `gcloud`
> default project differs from the one with GPU quota, a command missing
> `--project` fails with a confusing `PERMISSION_DENIED`. Every command below
> uses a placeholder `$GCP_PROJECT`; set it once at the top of your session.

## Quick reference

| Step | Command |
|------|---------|
| 1. Launch instance | `bash scripts/cloud/gcp_launch.sh --project $GCP_PROJECT` |
| 2. SSH in | `gcloud compute ssh igv-gpu --zone=us-central1-a --project=$GCP_PROJECT` |
| 3. Clone repo | `git clone ... && cd IG` |
| 4. Bootstrap | `bash scripts/cloud/bootstrap.sh` |
| 5. Fetch weights | `bash scripts/cloud/fetch_weights.sh` (one time; several GB into `~/boltz_cache`) |
| 6. Run pipeline | `tmux new -s igv` then `docker run --rm --gpus all --shm-size=32g --ipc=host -v $(pwd):/app -w /app -v $HOME/boltz_cache:/root/.boltz igv:latest bash scripts/run_all.sh` |
| 7. Sync results | `bash scripts/cloud/sync_results.sh --host ubuntu@<IP> --remote-dir /home/ubuntu/IG` |
| 8. Tear down | `gcloud compute instances stop igv-gpu --zone=us-central1-a --project=$GCP_PROJECT --discard-local-ssd=true` |

## Instance selection

### Hard requirement: >= 80 GB VRAM per GPU

Full-trunk Boltz-2 backprop with per-block gradient checkpointing fills an entire 80 GB card. A 48 GB GPU (L40S, A40, A6000) is a documented failure -- OOM during the attribution stage (03) with no workaround short of model parallelism, which this pipeline does not implement.

### Cost table (on-demand, as of Aug 2026 -- rates change, verify before launching)

| Provider | Instance | GPUs | VRAM/GPU | $/hr | $/day | Notes |
|----------|----------|------|----------|------|-------|-------|
| GCP | a2-ultragpu-1g | 1x A100-80GB | 80 GB | **~$5.07** | ~$122 | **Current instance.** The only way to rent a *single* 80 GB GPU. 12 vCPU / 170 GB RAM. |
| GCP | a2-ultragpu-8g | 8x A100-80GB | 80 GB | ~$80.29 | ~$1,927 | |
| GCP | a3-highgpu-8g | 8x H100-80GB | 80 GB | ~$98.32 | ~$2,360 | Fastest |

Workload is approximately 40 GPU-hours. GPU stages are 02 (embed), 03 (attribute), 04 (scan), and 07 (sanity). CPU-only stages are 00, 01, 05, 06.

### On-demand vs. spot

**Use on-demand for GPU stages.** Spot/preemptible instances will kill a running job. Stage 04 (scan) is resumable (skips computed rows), but stage 03 (attribute) is not -- a preemption during attribution loses all progress with no recovery. At ~40 GPU-hours total, the spot savings are not worth the risk of lost runs.

## GCP state

**Quota and capacity are different things.** Quota is an account limit you can
raise by asking. Capacity is whether the zone physically has a free machine
right now. You can hold quota and still get `ZONE_RESOURCE_POOL_EXHAUSTED`.

### Checking your GCP state

Verify that the Compute Engine API is enabled and billing is linked to a paid
account. GPU quota in common regions:

| Metric | Limit | Usable here? |
|---|---|---|
| `NVIDIA_A100_80GB_GPUS` | **0** | this is the one we need |
| `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` | **0** | |
| `NVIDIA_A100_GPUS` (40 GB) | 1 | no -- 40 GB OOMs at stage 03 |
| `PREEMPTIBLE_NVIDIA_A100_GPUS` (40 GB) | 16 | no -- same reason |
| `NVIDIA_L4_GPUS` (24 GB) | 8 | no |
| `NVIDIA_T4_GPUS` (16 GB) | 4 | no |
| H100 (`a3-*`) | metric absent | no |

New GCP projects typically have **zero 80 GB-class quota**. An increase must be
requested and takes 24-48 h (see "Requesting A100-80GB quota" below).

### Current instance

The scripts default to VM name `igv-gpu`, machine type `a2-ultragpu-1g`, zone
`us-central1-a`. A 500 GB pd-ssd boot disk is recommended; it costs ~$2.83/day
while the VM is stopped.

### Do this first, today

GCP `a2`/`a3` GPU quota is **commonly zero on new projects**, and approval takes
24-48 hours. This is the single most likely thing to block you, and the error
message when quota is missing is unhelpful. Request it before anything else:

```bash
gcloud compute regions describe us-central1 \
  --format="value(quotas[].metric,quotas[].limit)" | tr ';' '\n' | grep -i a2
```
If `NVIDIA_A100_80GB_GPUS` is 0, request an increase at
console.cloud.google.com/iam-admin/quotas before booking time to run this.

## Requesting A100-80GB quota on GCP

Only one metric needs raising: the regional `NVIDIA_A100_80GB_GPUS`. Some
projects also have a `GPUS_ALL_REGIONS` cap; check whether yours does and raise
it too if needed.

### Console path (recommended -- the CLI needs the `beta` component)

1. https://console.cloud.google.com/iam-admin/quotas?project=$GCP_PROJECT
2. Filter box, paste exactly: **`NVIDIA A100 80GB GPUs`**

   Note the spaces. The console filters on the *display* name, not the
   underscored API metric id -- searching `NVIDIA_A100_80GB_GPUS` returns
   nothing. Verified display names and quota ids, from the Cloud Quotas API:

   | Display name | Quota id | Dimension |
   |---|---|---|
   | NVIDIA A100 80GB GPUs | `NVIDIA-A100-80GB-GPUS-per-project-region` | `region` <- use this |
   | NVIDIA A100 80GB GPUs | `NVIDIA-A100-80GB-GPUS-per-project-zone` | `zone` |
   | Preemptible NVIDIA A100 80GB GPUs | `PREEMPTIBLE-NVIDIA-A100-80GB-GPUS-per-project-region` | `region` |
   | NVIDIA A100 GPUs | `NVIDIA-A100-GPUS-per-project-region` | `region` (40 GB -- too small) |

   Two rows share the display name "NVIDIA A100 80GB GPUs"; pick the one whose
   Dimensions chip reads `region: us-central1`.
3. Tick the row whose **Region** is `us-central1`
4. **EDIT QUOTAS** -> New limit: **1**
5. Submit.

Ask for **1**, not 8. `a2-ultragpu-1g` is a single A100-80GB, small requests are
approved far more often and faster, and 8 GPUs costs ~$40/hr anyway -- at which
point AWS `p4de` (already in quota) is the better box.

Worth filing at the same time, as a separate row in the same UI:
`PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` = 1. Preemptible quota is usually granted
more readily. Stage 04 (scan) is resumable so preemption is survivable there;
**never run stage 03 on preemptible** -- attribution is not resumable and a
preemption loses the whole run.

### Justification text

Reviewers read this. Be specific and modest:

> Academic research evaluating gradient-based attribution on protein structure
> prediction models (Boltz-2). Requires a single A100-80GB; smaller GPUs are
> insufficient because full-trunk backpropagation needs ~80 GB. Expected usage
> ~40 GPU-hours total, instances terminated after each run.

### Timeline and denial reasons

Typically 24-48 h, sometimes minutes if auto-approved. The two common denials:

- **Free-trial billing.** GPU increases are frequently refused on trial
  accounts. Confirm the billing account is a paid upgrade, not a trial.
- **Asking for too much.** A request for 8 on a project with no GPU history is
  more likely to be rejected than a request for 1.

Note that quota is not capacity: once granted, `a2-ultragpu-1g` can still fail
with `ZONE_RESOURCE_POOL_EXHAUSTED`. Try `us-central1-a`, then `-b`, `-c`, `-f`.

### API alternative (no `beta` component needed)

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://cloudquotas.googleapis.com/v1/projects/$GCP_PROJECT/locations/global/quotaPreferences?quotaPreferenceId=a100-80gb-us-central1" \
  -d '{
    "service": "compute.googleapis.com",
    "quotaId": "NVIDIA-A100-80GB-GPUS-per-project-region",
    "quotaConfig": {"preferredValue": "1"},
    "dimensions": {"region": "us-central1"},
    "contactEmail": "YOUR_EMAIL",
    "justification": "..."
  }'
```

Check status:

```bash
curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://cloudquotas.googleapis.com/v1/projects/$GCP_PROJECT/locations/global/quotaPreferences"
```

Discover ids for any other quota:

```bash
curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://cloudquotas.googleapis.com/v1/projects/$GCP_PROJECT/locations/global/services/compute.googleapis.com/quotaInfos?pageSize=500"
```

## GCP launch

### GPU quota (read this first)

A2 and A3 GPU quota is **commonly zero on new GCP projects**. You must request quota before launching. The error when quota is zero says "ZONE_RESOURCE_POOL_EXHAUSTED" or "Quota exceeded" -- it does NOT say "you have no quota".

Check and request at: https://console.cloud.google.com/iam-admin/quotas

Filter by "NVIDIA A100 80GB GPUs" or "NVIDIA H100 GPUs" in your target region. Quota requests may take 24-48 hours.

```bash
# Launch (dry run first):
bash scripts/cloud/gcp_launch.sh \
  --project my-project-id \
  --dry-run

# Launch for real:
bash scripts/cloud/gcp_launch.sh \
  --project my-project-id

# Delete when done:
bash scripts/cloud/gcp_launch.sh --delete igv-gpu --project my-project-id
```

See `bash scripts/cloud/gcp_launch.sh --help` for all options.

## Bootstrap

After SSH-ing into the instance:

```bash
git clone <your-repo-url> && cd IG
bash scripts/cloud/bootstrap.sh
```

Bootstrap is idempotent. It:
1. Installs NVIDIA drivers + container toolkit + Docker (if absent)
2. Builds the Docker image from `docker/Dockerfile`
3. Runs verification checks and prints a summary table

The verification block fails loudly on any of:
- Wrong Python version (requires 3.11 final or 3.12+)
- `torch.__version__` != `2.7.1+cu126`
- `torch.cuda.is_available()` false
- GPU 0 VRAM < 80 GiB
- Any forbidden package importable (boltzgen, protenix, chai_lab, gnina, cuequivariance)

## Running the pipeline

**Always use `tmux` or `nohup`.** Long runs die when SSH drops.

```bash
tmux new -s igv

docker run --rm --gpus all --shm-size=32g --ipc=host \
  -v $HOME/boltz_cache:/root/.boltz \
  -v $(pwd):/app -w /app igv:latest \
  bash scripts/run_all.sh

# Detach: Ctrl-B then D
# Reattach: tmux attach -t igv
```

### Critical Docker flags

| Flag | Why |
|------|-----|
| `--shm-size=32g` | Docker's default 64 MB `/dev/shm` causes DataLoader workers to be killed, producing a silent hang at 0% GPU. |
| `--ipc=host` | Allows shared memory between DataLoader workers. |
| `--gpus all` | Expose all GPUs to the container. |

**Never omit `--shm-size=32g --ipc=host`.** The failure mode is an 18-minute silent hang misreported as "DataLoader worker exited unexpectedly".

### Forbidden packages

These packages must NEVER be installed in the container:
- `boltzgen` -- vendors incompatible torch build
- `protenix` -- vendors incompatible torch build
- `chai_lab` -- vendors incompatible torch build
- `gnina` -- not needed, pulls conflicting deps
- `cuequivariance` -- segfaults on A100/sm_80

Installing any of them silently replaces the pinned `torch==2.7.1+cu126` CUDA wheel. Symptoms are never "wrong torch" -- they are exit 1 with empty stderr, or a job hanging 7+ hours.

## Syncing results

```bash
# Pull results to local machine:
bash scripts/cloud/sync_results.sh \
  --host ubuntu@<IP> \
  --remote-dir /home/ubuntu/IG

# Also push to S3:
bash scripts/cloud/sync_results.sh \
  --host ubuntu@<IP> \
  --remote-dir /home/ubuntu/IG \
  --s3 s3://my-bucket/ig-results

# Also push to GCS:
bash scripts/cloud/sync_results.sh \
  --host ubuntu@<IP> \
  --remote-dir /home/ubuntu/IG \
  --gcs gs://my-bucket/ig-results
```

The sync script explicitly includes `*.prov.json` provenance sidecars and verifies after transfer that every artifact has its sidecar. Results without provenance are not interpretable (see `docs/PIPELINE.md` for why).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Silent hang at 0% GPU utilization, 18+ minutes | `/dev/shm` exhaustion. Docker's 64 MB default kills DataLoader workers. | Add `--shm-size=32g --ipc=host` to `docker run`. |
| `FileNotFoundError: No .ckpt files in /root/.boltz` | Boltz-2 weights were never downloaded, or the persistent cache is not mounted. `docker run --rm` destroys `~/.boltz` with the container. | Run `bash scripts/cloud/fetch_weights.sh` once, then add `-v $HOME/boltz_cache:/root/.boltz` to every `docker run`. |
| Exit code 1 with empty stderr | Torch version clobbered. A forbidden package replaced `torch==2.7.1+cu126` with a PyPI wheel. | Rebuild the image from scratch. Check `pip list \| grep torch` inside the container. Never install boltzgen, protenix, chai_lab, gnina, or cuequivariance. |
| Segfault on A100 | `cuequivariance` present. Multiple versions segfault on sm_80. | `pip uninstall cuequivariance cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12 cuequivariance-torch` |
| `import torch` fails with missing `sys.get_int_max_str_digits` | Python 3.11.0rc1 or earlier pre-release. | Use Python 3.11 final (3.11.0+) or 3.12. The Dockerfile uses deadsnakes 3.11 which is always final. |
| Job hangs for 7+ hours | Torch clobber (see "empty stderr" above). | Same fix: rebuild image, avoid forbidden packages. |
| OOM during stage 03 | GPU has < 80 GB VRAM. | Use an instance with A100-80GB or H100-80GB. No workaround on 48 GB cards. |
| GCP: "ZONE_RESOURCE_POOL_EXHAUSTED" or "Quota exceeded" | A2/A3 GPU quota is zero. | Request quota at console.cloud.google.com/iam-admin/quotas. May take 24-48 hours. |

## Teardown checklist

Run through this list **every time** you finish a session:

- [ ] Pipeline results synced locally (run `sync_results.sh`)
- [ ] Provenance sidecars present for all artifacts (sync script reports orphans)
- [ ] **GCP**: VM stopped (`gcloud compute instances stop igv-gpu --zone=us-central1-a --project=$GCP_PROJECT --discard-local-ssd=true`)
- [ ] Verify in cloud console that no instances are running
- [ ] Check for leftover persistent disks (GCP) that may still incur charges
