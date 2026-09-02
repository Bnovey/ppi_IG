#!/usr/bin/env bash
# bootstrap.sh -- set up a freshly launched GPU instance for the IG pipeline.
# Run ON the instance after SSH-ing in. Idempotent: safe to re-run.
#
# Usage:
#   bash scripts/cloud/bootstrap.sh          # full setup
#   bash scripts/cloud/bootstrap.sh --help
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-igv:latest}"
FORBIDDEN_PACKAGES=(boltzgen protenix chai_lab gnina cuequivariance)
REQUIRED_TORCH="2.7.1+cu126"
MIN_VRAM_GIB=80

# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bootstrap.sh [OPTIONS]

Sets up a freshly launched GPU instance:
  1. Installs NVIDIA drivers + container toolkit + Docker (if absent)
  2. Builds the Docker image from docker/Dockerfile
  3. Runs verification checks (Python, torch, CUDA, VRAM, forbidden packages)

Options:
  --image-tag TAG   Docker image tag (default: igv:latest)
  --skip-build      Skip Docker image build (use existing image)
  --verify-only     Run only the verification block
  --help            Show this help

Environment:
  IMAGE_TAG         Same as --image-tag
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_BUILD=false
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-tag)   IMAGE_TAG="$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --verify-only) VERIFY_ONLY=true; shift ;;
        --help)        usage ;;
        *)             echo "Unknown option: $1" >&2; usage ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Docker command prefix.  `usermod -aG docker` does not take effect in the
# session that runs it, so a bare `docker` call fails with a socket
# permission error on a freshly bootstrapped VM.  Probe instead of assuming.
DOCKER="docker"
set_docker_cmd() {
    if docker info &>/dev/null; then
        DOCKER="docker"
    elif sudo -n docker info &>/dev/null || sudo docker info &>/dev/null; then
        DOCKER="sudo docker"
        log "  Using 'sudo docker' (docker group not active in this session)."
    fi
}

# ---------------------------------------------------------------------------
# Step 1: NVIDIA drivers
# ---------------------------------------------------------------------------
install_nvidia_drivers() {
    log "Checking NVIDIA drivers..."
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        log "  NVIDIA drivers already installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        return 0
    fi

    log "  Installing NVIDIA drivers..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends nvidia-driver-560
    log "  Drivers installed. A reboot may be required."
    if ! nvidia-smi &>/dev/null; then
        echo "ERROR: nvidia-smi still failing after install. Reboot and re-run." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Step 2: Docker + NVIDIA container toolkit
# ---------------------------------------------------------------------------
install_docker() {
    log "Checking Docker..."
    if command -v docker &>/dev/null; then
        log "  Docker already installed: $(docker --version)"
    else
        log "  Installing Docker..."
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker "$USER"
        log "  Docker installed. You may need to log out and back in for group membership."
    fi
}

install_nvidia_container_toolkit() {
    log "Checking NVIDIA container toolkit..."
    if ${DOCKER} info 2>/dev/null | grep -qi "nvidia"; then
        log "  NVIDIA container runtime already configured."
        return 0
    fi

    log "  Installing NVIDIA container toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    log "  NVIDIA container toolkit installed and configured."
}

# ---------------------------------------------------------------------------
# Step 3: Build Docker image
# ---------------------------------------------------------------------------
build_image() {
    log "Building Docker image: ${IMAGE_TAG}"
    log "  Context: ${REPO_DIR} (dockerfile: docker/Dockerfile)"
    # Context must be the repo root: the Dockerfile COPYs pyproject.toml
    # and src/, which do not exist under docker/.
    ${DOCKER} build -t "${IMAGE_TAG}" -f "${REPO_DIR}/docker/Dockerfile" "${REPO_DIR}"
    log "  Image built successfully."
}

# ---------------------------------------------------------------------------
# Step 4: Verification
# ---------------------------------------------------------------------------
verify() {
    log "Running verification inside container..."

    # The verification script runs inside the container.
    # --gpus all: need GPU access for CUDA checks
    # --shm-size=32g --ipc=host: match production settings (validates they work)
    # --rm: clean up after
    local PASS=true

    VERIFY_OUTPUT=$(${DOCKER} run --rm --gpus all --shm-size=32g --ipc=host \
        "${IMAGE_TAG}" python3 -c "
import sys, json

results = {}

# 1. Python version: must be 3.11 final or 3.12+ (not rc)
v = sys.version_info
ver_str = f'{v.major}.{v.minor}.{v.micro}'
py_ok = (v.major == 3 and v.minor >= 11 and v.releaselevel == 'final') or (v.major == 3 and v.minor >= 12)
results['python_version'] = {'value': ver_str, 'ok': py_ok}

# 2. Torch version
import torch
torch_ver = torch.__version__
torch_ok = (torch_ver == '${REQUIRED_TORCH}')
results['torch_version'] = {'value': torch_ver, 'ok': torch_ok}

# 3. CUDA available
cuda_ok = torch.cuda.is_available()
results['cuda_available'] = {'value': str(cuda_ok), 'ok': cuda_ok}

# 4. GPU 0 VRAM >= ${MIN_VRAM_GIB} GiB
if cuda_ok:
    mem_bytes = torch.cuda.get_device_properties(0).total_memory
    mem_gib = mem_bytes / (1024**3)
    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_count = torch.cuda.device_count()
    vram_ok = mem_gib >= ${MIN_VRAM_GIB}
    results['gpu_0_vram_gib'] = {'value': f'{mem_gib:.1f}', 'ok': vram_ok}
    results['gpu_name'] = {'value': gpu_name, 'ok': True}
    results['gpu_count'] = {'value': str(gpu_count), 'ok': True}
else:
    results['gpu_0_vram_gib'] = {'value': 'N/A', 'ok': False}
    results['gpu_name'] = {'value': 'N/A', 'ok': False}
    results['gpu_count'] = {'value': '0', 'ok': False}

# 5. Forbidden packages must NOT be importable
forbidden = '${FORBIDDEN_PACKAGES[*]}'.split()
for pkg in forbidden:
    try:
        __import__(pkg)
        results[f'forbidden_{pkg}'] = {'value': 'INSTALLED', 'ok': False}
    except ImportError:
        results[f'forbidden_{pkg}'] = {'value': 'not found', 'ok': True}

# 6. boltz importable
try:
    import boltz
    boltz_ver = getattr(boltz, '__version__', 'unknown')
    results['boltz'] = {'value': boltz_ver, 'ok': True}
except ImportError:
    results['boltz'] = {'value': 'MISSING', 'ok': False}

print(json.dumps(results))
" 2>&1) || true

    # Parse the JSON output (last line)
    local JSON_LINE
    JSON_LINE=$(echo "$VERIFY_OUTPUT" | grep '^{' | tail -1)

    if [[ -z "$JSON_LINE" ]]; then
        echo "ERROR: Verification script produced no parseable output." >&2
        echo "Raw output:" >&2
        echo "$VERIFY_OUTPUT" >&2
        exit 1
    fi

    # Print summary table
    echo ""
    echo "============================================================"
    echo "  BOOTSTRAP VERIFICATION SUMMARY"
    echo "============================================================"
    printf "  %-25s %-25s %s\n" "CHECK" "VALUE" "STATUS"
    echo "  ---------------------------------------------------------"

    local ALL_OK=true
    while IFS= read -r key; do
        local val ok
        val=$(echo "$JSON_LINE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['$key']['value'])")
        ok=$(echo "$JSON_LINE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['$key']['ok'])")
        local status="PASS"
        if [[ "$ok" != "True" ]]; then
            status="FAIL"
            ALL_OK=false
        fi
        printf "  %-25s %-25s %s\n" "$key" "$val" "$status"
    done < <(echo "$JSON_LINE" | python3 -c "import sys,json; [print(k) for k in json.load(sys.stdin)]")

    echo "============================================================"

    if [[ "$ALL_OK" != "true" ]]; then
        echo ""
        echo "ERROR: One or more verification checks FAILED." >&2
        echo "Fix the issues above before running the pipeline." >&2
        exit 1
    fi

    echo ""
    log "All verification checks passed."
    echo ""
    echo "Next steps:"
    echo "  1. Clone / rsync the repo onto this instance"
    echo "  2. Run the pipeline inside Docker:"
    echo ""
    echo "     docker run --rm --gpus all --shm-size=32g --ipc=host \\"
    echo "       -v \$(pwd):/app -w /app ${IMAGE_TAG} \\"
    echo "       bash scripts/run_all.sh"
    echo ""
    echo "  IMPORTANT: Always use --shm-size=32g --ipc=host to avoid"
    echo "  /dev/shm exhaustion (silent hang at 0% GPU utilization)."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ "$VERIFY_ONLY" == "true" ]]; then
    verify
    exit 0
fi

install_nvidia_drivers
install_docker
set_docker_cmd
install_nvidia_container_toolkit
set_docker_cmd

if [[ "$SKIP_BUILD" == "false" ]]; then
    build_image
fi

verify
