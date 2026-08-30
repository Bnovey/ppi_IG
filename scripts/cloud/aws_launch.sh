#!/usr/bin/env bash
# aws_launch.sh -- Launch or terminate an EC2 instance for the IG pipeline.
#
# Usage:
#   bash scripts/cloud/aws_launch.sh [OPTIONS]
#   bash scripts/cloud/aws_launch.sh --dry-run
#   bash scripts/cloud/aws_launch.sh --terminate i-0123456789abcdef0
#   bash scripts/cloud/aws_launch.sh --help
#
# COST WARNING:
#   p4de.24xlarge: ~$40.97/hr on-demand (us-east-1, Aug 2026)
#   p5.48xlarge:   ~$98.32/hr on-demand (us-east-1, Aug 2026)
#   ALWAYS terminate when done. A forgotten p5 costs ~$2,360/day.
#
# Requirements:
#   - AWS CLI v2 configured with valid credentials
#   - A key pair created in the target region
#   - A security group allowing SSH (port 22) from your IP
#   - (Optional) A specific subnet if launching in a VPC
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults -- override via flags or environment variables
# ---------------------------------------------------------------------------
INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-p4de.24xlarge}"
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${AWS_KEY_NAME:-}"
SECURITY_GROUP="${AWS_SECURITY_GROUP:-}"
SUBNET="${AWS_SUBNET:-}"
EBS_SIZE="${AWS_EBS_SIZE:-500}"
NAME_TAG="${AWS_NAME_TAG:-igv-gpu}"

# Deep Learning AMI (Ubuntu 22.04) -- override with AWS_AMI or --ami.
# To find the latest DLAMI in your region:
#   aws ec2 describe-images --owners amazon \
#     --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)*' \
#     --query 'reverse(sort_by(Images,&CreationDate))[0].[ImageId,Name]' \
#     --output text --region us-east-1
AMI="${AWS_AMI:-ami-0a9a5d2b65cce04eb}"

DRY_RUN=false
TERMINATE_ID=""

# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: aws_launch.sh [OPTIONS]

Launch an EC2 GPU instance for the IG pipeline.

Options:
  --instance-type TYPE   EC2 instance type (default: p4de.24xlarge)
  --region REGION        AWS region (default: us-east-1)
  --key-name NAME        SSH key pair name (REQUIRED for launch)
  --security-group SG    Security group ID (REQUIRED for launch)
  --subnet SUBNET        Subnet ID (optional; uses default VPC if omitted)
  --ami AMI_ID           AMI ID (default: Deep Learning Base AMI, Ubuntu 22.04)
  --ebs-size GB          Root EBS volume size in GB (default: 500)
  --name TAG             Name tag for the instance (default: igv-gpu)
  --dry-run              Print the aws command without executing it
  --terminate ID         Terminate the specified instance and exit

Environment variables:
  AWS_INSTANCE_TYPE, AWS_REGION, AWS_KEY_NAME, AWS_SECURITY_GROUP,
  AWS_SUBNET, AWS_AMI, AWS_EBS_SIZE, AWS_NAME_TAG

To find the latest Deep Learning AMI:
  aws ec2 describe-images --owners amazon \
    --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)*' \
    --query 'reverse(sort_by(Images,&CreationDate))[0].[ImageId,Name]' \
    --output text --region us-east-1

COST WARNING (on-demand, us-east-1, rates as of Aug 2026 -- verify current pricing):
  p4de.24xlarge  8x A100-80GB   ~$40.97/hr   (~$983/day)
  p5.48xlarge    8x H100-80GB   ~$98.32/hr   (~$2,360/day)
  ALWAYS terminate when done.

VRAM requirement: >= 80 GB per GPU. Full-trunk Boltz-2 backprop with per-block
gradient checkpointing fills an entire 80 GB card. 48 GB GPUs (L40S, A40) will
fail with OOM during the attribution stage.

Recommended: Use ON-DEMAND instances for GPU stages (02, 03, 04, 07).
  Spot preemption will kill a 40-minute IG run mid-flight. Stage 04 (scan) is
  resumable but stage 03 (attribute) is not -- lost work cannot be recovered.
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-type)   INSTANCE_TYPE="$2"; shift 2 ;;
        --region)          REGION="$2"; shift 2 ;;
        --key-name)        KEY_NAME="$2"; shift 2 ;;
        --security-group)  SECURITY_GROUP="$2"; shift 2 ;;
        --subnet)          SUBNET="$2"; shift 2 ;;
        --ami)             AMI="$2"; shift 2 ;;
        --ebs-size)        EBS_SIZE="$2"; shift 2 ;;
        --name)            NAME_TAG="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --terminate)       TERMINATE_ID="$2"; shift 2 ;;
        --help)            usage ;;
        *)                 echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Terminate mode
# ---------------------------------------------------------------------------
if [[ -n "$TERMINATE_ID" ]]; then
    echo "Terminating instance: ${TERMINATE_ID} in region ${REGION}"
    aws ec2 terminate-instances \
        --instance-ids "$TERMINATE_ID" \
        --region "$REGION"
    echo "Terminate request sent. Verify in the console:"
    echo "  https://${REGION}.console.aws.amazon.com/ec2/home?region=${REGION}#Instances:"
    exit 0
fi

# ---------------------------------------------------------------------------
# Validate required parameters for launch
# ---------------------------------------------------------------------------
if [[ -z "$KEY_NAME" ]]; then
    echo "ERROR: --key-name is required (or set AWS_KEY_NAME)." >&2
    echo "  Create one: aws ec2 create-key-pair --key-name igv --region ${REGION}" >&2
    exit 1
fi
if [[ -z "$SECURITY_GROUP" ]]; then
    echo "ERROR: --security-group is required (or set AWS_SECURITY_GROUP)." >&2
    echo "  Create one with SSH access from your IP:" >&2
    echo "    aws ec2 create-security-group --group-name igv-ssh --description 'SSH for IG' --region ${REGION}" >&2
    echo "    aws ec2 authorize-security-group-ingress --group-name igv-ssh --protocol tcp --port 22 --cidr \$(curl -s ifconfig.me)/32 --region ${REGION}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the launch command
# ---------------------------------------------------------------------------
BLOCK_DEVICE_MAPPINGS="[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
TAG_SPEC="ResourceType=instance,Tags=[{Key=Name,Value=${NAME_TAG}}]"

CMD=(
    aws ec2 run-instances
    --instance-type "$INSTANCE_TYPE"
    --image-id "$AMI"
    --key-name "$KEY_NAME"
    --security-group-ids "$SECURITY_GROUP"
    --region "$REGION"
    --block-device-mappings "$BLOCK_DEVICE_MAPPINGS"
    --tag-specifications "$TAG_SPEC"
    --count 1
)

if [[ -n "$SUBNET" ]]; then
    CMD+=(--subnet-id "$SUBNET")
fi

# ---------------------------------------------------------------------------
# Execute or dry-run
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN -- would execute:"
    echo ""
    # Print the command in a copy-pasteable format
    printf '%s' "${CMD[0]}"
    for arg in "${CMD[@]:1}"; do
        printf ' \\\n  %s' "$arg"
    done
    echo ""
    echo ""
    echo "(No AWS API call was made.)"
    exit 0
fi

echo "Launching ${INSTANCE_TYPE} in ${REGION}..."
echo ""

RESULT=$("${CMD[@]}" --output json)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")

echo "Instance launched: ${INSTANCE_ID}"
echo ""
echo "Waiting for instance to enter 'running' state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "============================================================"
echo "  INSTANCE READY"
echo "============================================================"
echo "  Instance ID:  ${INSTANCE_ID}"
echo "  Public IP:    ${PUBLIC_IP}"
echo "  Instance:     ${INSTANCE_TYPE}"
echo "  Region:       ${REGION}"
echo "============================================================"
echo ""
echo "Connect:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Next steps on the instance:"
echo "  1. git clone <your-repo> && cd IG"
echo "  2. bash scripts/cloud/bootstrap.sh"
echo "  3. tmux new -s igv   # detached execution -- SSH drops won't kill the run"
echo "  4. docker run --rm --gpus all --shm-size=32g --ipc=host \\"
echo "       -v \$(pwd):/app -w /app igv:latest bash scripts/run_all.sh"
echo ""
echo "IMPORTANT: When done, terminate to stop billing:"
echo "  bash scripts/cloud/aws_launch.sh --terminate ${INSTANCE_ID} --region ${REGION}"
echo ""
echo "  Or via the console:"
echo "  https://${REGION}.console.aws.amazon.com/ec2/home?region=${REGION}#Instances:"
