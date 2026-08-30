#!/usr/bin/env bash
# gcp_launch.sh -- Create or delete a GCP GPU VM for the IG pipeline.
#
# Usage:
#   bash scripts/cloud/gcp_launch.sh [OPTIONS]
#   bash scripts/cloud/gcp_launch.sh --dry-run
#   bash scripts/cloud/gcp_launch.sh --delete igv-gpu
#   bash scripts/cloud/gcp_launch.sh --help
#
# COST WARNING:
#   a2-ultragpu-1g:  1x A100-80GB   ~$10.04/hr on-demand (us-central1, Aug 2026)
#   a2-ultragpu-8g:  8x A100-80GB   ~$80.29/hr on-demand
#   a3-highgpu-8g:   8x H100-80GB   ~$98.32/hr on-demand
#   ALWAYS delete when done.
#
# GPU QUOTA WARNING:
#   A2 and A3 instances require GPU quota that is commonly ZERO on new projects.
#   You must request quota BEFORE launching. Check and request at:
#     https://console.cloud.google.com/iam-admin/quotas
#   Filter by: "NVIDIA A100 80GB GPUs" or "NVIDIA H100 GPUs" in your region.
#   The error message when quota is zero is confusing -- it says
#   "ZONE_RESOURCE_POOL_EXHAUSTED" or "Quota exceeded", not "you have no quota".
#   Quota requests for A100/H100 may take 24-48 hours to approve.
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults -- override via flags or environment variables
# ---------------------------------------------------------------------------
MACHINE_TYPE="${GCP_MACHINE_TYPE:-a2-ultragpu-1g}"
ZONE="${GCP_ZONE:-us-central1-a}"
PROJECT="${GCP_PROJECT:-}"
VM_NAME="${GCP_VM_NAME:-igv-gpu}"
BOOT_DISK_SIZE="${GCP_BOOT_DISK_SIZE:-500}"

# Deep Learning VM image family -- Ubuntu 22.04 with CUDA pre-installed.
# To list available images:
#   gcloud compute images list --project=deeplearning-platform-release \
#     --filter="family:common-cu126-ubuntu-2204" --format="table(name,family)"
IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-common-cu126-ubuntu-2204}"
IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"

DRY_RUN=false
DELETE_NAME=""

# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: gcp_launch.sh [OPTIONS]

Create a GCP GPU VM for the IG pipeline.

Options:
  --machine-type TYPE    Machine type (default: a2-ultragpu-1g)
  --zone ZONE            GCP zone (default: us-central1-a)
  --project PROJECT      GCP project ID (REQUIRED, or set GCP_PROJECT)
  --name NAME            VM name (default: igv-gpu)
  --boot-disk-size GB    Boot disk size in GB (default: 500)
  --image-family FAMILY  VM image family (default: common-cu126-ubuntu-2204)
  --image-project PROJ   Image project (default: deeplearning-platform-release)
  --dry-run              Print the gcloud command without executing it
  --delete NAME          Delete the specified VM and exit

Environment variables:
  GCP_MACHINE_TYPE, GCP_ZONE, GCP_PROJECT, GCP_VM_NAME,
  GCP_BOOT_DISK_SIZE, GCP_IMAGE_FAMILY, GCP_IMAGE_PROJECT

To list Deep Learning VM images:
  gcloud compute images list --project=deeplearning-platform-release \
    --filter="family~common-cu.*ubuntu-2204" --format="table(name,family)"

COST WARNING (on-demand, rates as of Aug 2026 -- verify current pricing):
  a2-ultragpu-1g   1x A100-80GB   ~$10.04/hr   (~$241/day)
  a2-ultragpu-8g   8x A100-80GB   ~$80.29/hr   (~$1,927/day)
  a3-highgpu-8g    8x H100-80GB   ~$98.32/hr   (~$2,360/day)
  ALWAYS delete when done.

GPU QUOTA:
  A2/A3 GPU quota is commonly ZERO on new GCP projects. You MUST request it
  before launching. The denial error says "ZONE_RESOURCE_POOL_EXHAUSTED" or
  "Quota exceeded" -- it does NOT say "you have no quota". Check at:
    https://console.cloud.google.com/iam-admin/quotas
  Filter: "NVIDIA A100 80GB GPUs" (for a2) or "NVIDIA H100 GPUs" (for a3).

VRAM requirement: >= 80 GB per GPU. Full-trunk Boltz-2 backprop with per-block
gradient checkpointing fills an entire 80 GB card. 48 GB GPUs will OOM.

Recommended: Use ON-DEMAND instances for GPU stages.
  Spot/preemptible VMs will kill a 40-minute IG run mid-flight. Stage 03
  (attribute) is not resumable -- lost work cannot be recovered.
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine-type)    MACHINE_TYPE="$2"; shift 2 ;;
        --zone)            ZONE="$2"; shift 2 ;;
        --project)         PROJECT="$2"; shift 2 ;;
        --name)            VM_NAME="$2"; shift 2 ;;
        --boot-disk-size)  BOOT_DISK_SIZE="$2"; shift 2 ;;
        --image-family)    IMAGE_FAMILY="$2"; shift 2 ;;
        --image-project)   IMAGE_PROJECT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --delete)          DELETE_NAME="$2"; shift 2 ;;
        --help)            usage ;;
        *)                 echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Delete mode
# ---------------------------------------------------------------------------
if [[ -n "$DELETE_NAME" ]]; then
    if [[ -z "$PROJECT" ]]; then
        echo "ERROR: --project is required for delete (or set GCP_PROJECT)." >&2
        exit 1
    fi
    REGION="${ZONE%-*}"
    echo "Deleting VM: ${DELETE_NAME} in zone ${ZONE}, project ${PROJECT}"
    gcloud compute instances delete "$DELETE_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --quiet
    echo "VM deleted. Verify in the console:"
    echo "  https://console.cloud.google.com/compute/instances?project=${PROJECT}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Validate required parameters for create
# ---------------------------------------------------------------------------
if [[ -z "$PROJECT" ]]; then
    echo "ERROR: --project is required (or set GCP_PROJECT)." >&2
    echo "  List projects: gcloud projects list" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the command
# ---------------------------------------------------------------------------
CMD=(
    gcloud compute instances create "$VM_NAME"
    --machine-type="$MACHINE_TYPE"
    --zone="$ZONE"
    --project="$PROJECT"
    --image-family="$IMAGE_FAMILY"
    --image-project="$IMAGE_PROJECT"
    --boot-disk-size="${BOOT_DISK_SIZE}GB"
    --boot-disk-type=pd-ssd
    --maintenance-policy=TERMINATE
    --metadata="install-nvidia-driver=True"
    --scopes=default,storage-rw
)

# ---------------------------------------------------------------------------
# Execute or dry-run
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN -- would execute:"
    echo ""
    printf '%s' "${CMD[0]}"
    for arg in "${CMD[@]:1}"; do
        printf ' \\\n  %s' "$arg"
    done
    echo ""
    echo ""
    echo "(No GCP API call was made.)"
    echo ""
    echo "REMINDER: Verify you have GPU quota before running for real:"
    echo "  https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT}"
    exit 0
fi

echo "Creating VM ${VM_NAME} (${MACHINE_TYPE}) in ${ZONE}..."
echo ""

"${CMD[@]}"

EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "============================================================"
echo "  VM READY"
echo "============================================================"
echo "  VM Name:      ${VM_NAME}"
echo "  External IP:  ${EXTERNAL_IP}"
echo "  Machine:      ${MACHINE_TYPE}"
echo "  Zone:         ${ZONE}"
echo "  Project:      ${PROJECT}"
echo "============================================================"
echo ""
echo "Connect:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT}"
echo "  # or: ssh ${EXTERNAL_IP}"
echo ""
echo "Next steps on the VM:"
echo "  1. git clone <your-repo> && cd IG"
echo "  2. bash scripts/cloud/bootstrap.sh"
echo "  3. tmux new -s igv   # detached execution -- SSH drops won't kill the run"
echo "  4. docker run --rm --gpus all --shm-size=32g --ipc=host \\"
echo "       -v \$(pwd):/app -w /app igv:latest bash scripts/run_all.sh"
echo ""
echo "IMPORTANT: When done, delete to stop billing:"
echo "  bash scripts/cloud/gcp_launch.sh --delete ${VM_NAME} --zone ${ZONE} --project ${PROJECT}"
echo ""
echo "  Or via the console:"
echo "  https://console.cloud.google.com/compute/instances?project=${PROJECT}"
