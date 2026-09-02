#!/usr/bin/env bash
# fetch_weights.sh -- download Boltz-2 checkpoints into a PERSISTENT host cache.
#
# Why this exists:
#   boltz downloads its checkpoints to ~/.boltz on first use. The pipeline runs
#   inside `docker run --rm`, so that directory is destroyed with the container
#   and every stage would re-download several GB -- or, as actually happened,
#   fail outright with "No .ckpt files in /root/.boltz".
#
#   So we keep the cache on the host boot disk and bind-mount it at /root/.boltz.
#   The boot disk is preserved across VM stop/start, so this is a one-time cost.
#
# Usage:
#   bash scripts/cloud/fetch_weights.sh            # download if missing
#   bash scripts/cloud/fetch_weights.sh --force    # re-download even if present
#
# Every docker run in the pipeline must include:
#   -v "$BOLTZ_CACHE:/root/.boltz"
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-igv:latest}"
BOLTZ_CACHE="${BOLTZ_CACHE:-$HOME/boltz_cache}"

FORCE=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --help)  sed -n '2,20p' "$0"; exit 0 ;;
        *)       echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

DOCKER="docker"
docker info &>/dev/null || DOCKER="sudo docker"

mkdir -p "$BOLTZ_CACHE"

if [[ "$FORCE" == "false" ]] && compgen -G "$BOLTZ_CACHE/*.ckpt" > /dev/null; then
    log "Checkpoints already present in $BOLTZ_CACHE:"
    ls -lh "$BOLTZ_CACHE"/*.ckpt
    exit 0
fi

log "Downloading Boltz-2 weights into $BOLTZ_CACHE (several GB, one time)..."
${DOCKER} run --rm \
    -v "${REPO_DIR}:/app" -w /app \
    -v "${BOLTZ_CACHE}:/root/.boltz" \
    "${IMAGE_TAG}" \
    python3 -c '
from pathlib import Path
from boltz.main import download_boltz2
download_boltz2(Path("/root/.boltz"))
'

if ! compgen -G "$BOLTZ_CACHE/*.ckpt" > /dev/null; then
    echo "ERROR: download finished but no .ckpt in $BOLTZ_CACHE" >&2
    exit 1
fi

log "Done. Checkpoints:"
ls -lh "$BOLTZ_CACHE"/*.ckpt
