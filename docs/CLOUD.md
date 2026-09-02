# Cloud Provisioning Runbook

Run the IG (Boltz-2 gradient attribution) pipeline on AWS or GCP GPU instances.

## Quick reference

| Step | Command |
|------|---------|
| 1. Launch instance | `bash scripts/cloud/aws_launch.sh --key-name ... --security-group ...` |
| 2. SSH in | `ssh -i ~/.ssh/key.pem ubuntu@<IP>` |
| 3. Clone repo | `git clone ... && cd IG` |
| 4. Bootstrap | `bash scripts/cloud/bootstrap.sh` |
| 5. Run pipeline | `tmux new -s igv` then `docker run --rm --gpus all --shm-size=32g --ipc=host -v $(pwd):/app -w /app igv:latest bash scripts/run_all.sh` |
| 6. Sync results | `bash scripts/cloud/sync_results.sh --host ubuntu@<IP> --remote-dir /home/ubuntu/IG` |
| 7. Tear down | `bash scripts/cloud/aws_launch.sh --terminate <instance-id>` |

## Instance selection

### Hard requirement: >= 80 GB VRAM per GPU

Full-trunk Boltz-2 backprop with per-block gradient checkpointing fills an entire 80 GB card. A 48 GB GPU (L40S, A40, A6000) is a documented failure -- OOM during the attribution stage (03) with no workaround short of model parallelism, which this pipeline does not implement.

### Cost table (on-demand, as of Aug 2026 -- rates change, verify before launching)

| Provider | Instance | GPUs | VRAM/GPU | $/hr | $/day | Notes |
|----------|----------|------|----------|------|-------|-------|
| AWS | p4de.24xlarge | 8x A100-80GB | 80 GB | ~$40.97 | ~$983 | AWS has **no single-GPU 80 GB option** -- 8 GPUs is the floor, billed whether used or not. Worth it only for the parallel sweep. |
| AWS | p5.48xlarge | 8x H100-80GB | 80 GB | ~$98.32 | ~$2,360 | Fastest |
| GCP | a2-ultragpu-1g | 1x A100-80GB | 80 GB | **~$5.07** | ~$122 | **Recommended start.** The only way to rent a *single* 80 GB GPU. 12 vCPU / 170 GB RAM. |
| GCP | a2-ultragpu-8g | 8x A100-80GB | 80 GB | ~$80.29 | ~$1,927 | |
| GCP | a3-highgpu-8g | 8x H100-80GB | 80 GB | ~$98.32 | ~$2,360 | Fastest |

Workload is approximately 40 GPU-hours. GPU stages are 02 (embed), 03 (attribute), 04 (scan), and 07 (sanity). CPU-only stages are 00, 01, 05, 06.

### On-demand vs. spot

**Use on-demand for GPU stages.** Spot/preemptible instances will kill a running job. Stage 04 (scan) is resumable (skips computed rows), but stage 03 (attribute) is not -- a preemption during attribution loses all progress with no recovery. At ~40 GPU-hours total, the spot savings are not worth the risk of lost runs.

## Which provider: AWS today (verified for account 516962256450)

**Quota and capacity are different things.** Quota is an account limit you can
raise by asking. Capacity is whether the zone physically has a free machine
right now. You can hold quota and still get `InsufficientInstanceCapacity`.

### Verified quota state, checked 2026-08-30

| | Needs | Your quota | Verdict |
|---|---|---|---|
| `p4de.24xlarge` (8x A100-80GB) | 96 vCPU | **97 vCPU** | **launchable today, no request needed** |
| `p5.48xlarge` (8x H100-80GB) | 192 vCPU | 97 vCPU | needs an increase to >= 192 |

"Running On-Demand P instances" is 97 in both `us-east-1` and `us-west-2`, and
`p4de.24xlarge` is 96 vCPU -- so there is room for exactly one.

`p4de.24xlarge` is offered only in **us-east-1c** and **us-east-1d**.
`p5.48xlarge` is offered in all six us-east-1 AZs.

### Verified GCP state, project `agrosbio`, checked 2026-08-30

Compute Engine API is enabled and billing is linked to the open account
(`017BD0-C84E33-6C9E0B`). GPU quota in us-central1 / us-east4 / us-west4 /
europe-west4:

| Metric | Limit | Usable here? |
|---|---|---|
| `NVIDIA_A100_80GB_GPUS` | **0** | this is the one we need |
| `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` | **0** | |
| `NVIDIA_A100_GPUS` (40 GB) | 1 | no -- 40 GB OOMs at stage 03 |
| `PREEMPTIBLE_NVIDIA_A100_GPUS` (40 GB) | 16 | no -- same reason |
| `NVIDIA_L4_GPUS` (24 GB) | 8 | no |
| `NVIDIA_T4_GPUS` (16 GB) | 4 | no |
| H100 (`a3-*`) | metric absent | no |

**GCP has zero 80 GB-class quota.** Every GPU it will currently let you start is
too small for full-trunk Boltz-2 backprop. An increase must be requested and
takes 24-48h.

Note also that the default `gcloud` project was `g-prs-478707`, which is not in
this account's project list. The correct project is `agrosbio`:
`gcloud config set project agrosbio`.

### Therefore

Earlier guidance in this file preferred GCP on cost-per-GPU-hour, and that is
still true in isolation: `a2-ultragpu-1g` is ~$5.07/hr for one A100-80GB versus
~$41/hr for eight on AWS. But **having quota now beats being cheaper later.**

Since the AWS box comes with 8 GPUs whether you use them or not, do not run the
pipeline sequentially on it. Run the sanity gate, then fan the 16 datasets out
across all 8 GPUs -- that is where p4de earns its rate.

Use GCP instead if you would rather wait for quota and spend ~$20 on the
go/no-go than ~$150.



The deciding fact is instance shape, not price per GPU.

**AWS has no single-GPU 80 GB instance.** The smallest option meeting the >= 80 GB
requirement is `p4de.24xlarge`, which is 8x A100-80GB at ~$41/hr -- billed in full
even while 7 GPUs sit idle. **GCP `a2-ultragpu-1g` rents exactly one A100-80GB at
~$5.07/hr.**

That matters because the first milestone is inherently sequential: run stage 07
(sanity gate), then stage 0 on a single dataset, and look at one number before
committing to anything else.

| Phase | Instance | Wall clock | Cost |
|---|---|---|---|
| Sanity gate + stage 0 go/no-go | GCP `a2-ultragpu-1g` | ~3-4 h | **~$20** |
| Same on AWS `p4de.24xlarge` | 8 GPUs, 7 idle | ~3-4 h | ~$160 |
| Full 16-dataset sweep, parallel | AWS `p4de.24xlarge` or GCP `a2-ultragpu-8g` | ~5-6 h | ~$210-250 |

So: **GCP single GPU to find out whether the method works at all; an 8-GPU box
later only if it does.** Stages 03 and 04 are embarrassingly parallel across
datasets, so the multi-GPU box genuinely pays off for the sweep -- but only then.

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

## AWS launch

```bash
# Find the latest Deep Learning AMI:
aws ec2 describe-images --owners amazon \
  --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)*' \
  --query 'reverse(sort_by(Images,&CreationDate))[0].[ImageId,Name]' \
  --output text --region us-east-1

# Launch (dry run first):
bash scripts/cloud/aws_launch.sh \
  --key-name my-keypair \
  --security-group sg-0123456789abcdef0 \
  --dry-run

# Launch for real:
bash scripts/cloud/aws_launch.sh \
  --key-name my-keypair \
  --security-group sg-0123456789abcdef0

# Terminate when done:
bash scripts/cloud/aws_launch.sh --terminate i-0123456789abcdef0
```

See `bash scripts/cloud/aws_launch.sh --help` for all options.

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
- [ ] **AWS**: Instance terminated (`aws_launch.sh --terminate <id>` or console)
- [ ] **GCP**: VM deleted (`gcp_launch.sh --delete <name>` or console)
- [ ] Verify in cloud console that no instances are running
- [ ] Check for leftover EBS volumes (AWS) or persistent disks (GCP) that may still incur charges
- [ ] Revoke any temporary security group rules you added
