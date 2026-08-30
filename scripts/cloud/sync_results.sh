#!/usr/bin/env bash
# sync_results.sh -- Pull results and processed data from a remote GPU instance.
#
# Copies results/ and data/processed/ via rsync over SSH, then verifies that
# every artifact has its .prov.json provenance sidecar.
#
# Usage:
#   bash scripts/cloud/sync_results.sh --host ubuntu@1.2.3.4 --remote-dir /home/ubuntu/IG
#   bash scripts/cloud/sync_results.sh --host ubuntu@1.2.3.4 --remote-dir /home/ubuntu/IG --s3 s3://my-bucket/ig-results
#   bash scripts/cloud/sync_results.sh --host ubuntu@1.2.3.4 --remote-dir /home/ubuntu/IG --gcs gs://my-bucket/ig-results
#   bash scripts/cloud/sync_results.sh --help
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HOST=""
REMOTE_DIR=""
LOCAL_DIR="$(pwd)"
SSH_KEY=""
S3_DEST=""
GCS_DEST=""

# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: sync_results.sh [OPTIONS]

Pull results/ and data/processed/ from a remote instance via rsync.

Options:
  --host USER@IP         Remote host (REQUIRED), e.g. ubuntu@1.2.3.4
  --remote-dir PATH      Path to the repo root on the remote (REQUIRED)
  --local-dir PATH       Local destination directory (default: current directory)
  --ssh-key PATH         Path to SSH private key (optional, e.g. ~/.ssh/igv.pem)
  --s3 S3_URI            Also push to S3 (e.g. s3://bucket/prefix)
  --gcs GCS_URI          Also push to GCS (e.g. gs://bucket/prefix)
  --help                 Show this help

What gets synced:
  results/               All pipeline outputs
  data/processed/        Processed libraries and embeddings

Provenance sidecars (*.prov.json) are explicitly included. After transfer,
this script verifies every artifact has its sidecar and reports orphans.
Results without provenance are not interpretable.
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)        HOST="$2"; shift 2 ;;
        --remote-dir)  REMOTE_DIR="$2"; shift 2 ;;
        --local-dir)   LOCAL_DIR="$2"; shift 2 ;;
        --ssh-key)     SSH_KEY="$2"; shift 2 ;;
        --s3)          S3_DEST="$2"; shift 2 ;;
        --gcs)         GCS_DEST="$2"; shift 2 ;;
        --help)        usage ;;
        *)             echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [[ -z "$HOST" ]]; then
    echo "ERROR: --host is required." >&2
    exit 1
fi
if [[ -z "$REMOTE_DIR" ]]; then
    echo "ERROR: --remote-dir is required." >&2
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Build SSH options
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS+=(-i "$SSH_KEY")
fi

RSYNC_SSH="ssh ${SSH_OPTS[*]}"

# ---------------------------------------------------------------------------
# Sync results/ from remote
# ---------------------------------------------------------------------------
log "Syncing results/ from ${HOST}:${REMOTE_DIR}/results/"
mkdir -p "${LOCAL_DIR}/results"
rsync -avz --progress \
    -e "$RSYNC_SSH" \
    --include='*/' \
    --include='*.csv' \
    --include='*.npz' \
    --include='*.json' \
    --include='*.prov.json' \
    --exclude='__pycache__' \
    "${HOST}:${REMOTE_DIR}/results/" \
    "${LOCAL_DIR}/results/"

# ---------------------------------------------------------------------------
# Sync data/processed/ from remote
# ---------------------------------------------------------------------------
log "Syncing data/processed/ from ${HOST}:${REMOTE_DIR}/data/processed/"
mkdir -p "${LOCAL_DIR}/data/processed"
rsync -avz --progress \
    -e "$RSYNC_SSH" \
    --include='*/' \
    --include='*.parquet' \
    --include='*.json' \
    --include='*.npz' \
    --include='*.prov.json' \
    --exclude='__pycache__' \
    "${HOST}:${REMOTE_DIR}/data/processed/" \
    "${LOCAL_DIR}/data/processed/"

# ---------------------------------------------------------------------------
# Verify provenance sidecars
# ---------------------------------------------------------------------------
log "Verifying provenance sidecars..."

ORPHAN_COUNT=0
ARTIFACT_COUNT=0
SIDECAR_COUNT=0

check_provenance() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        return
    fi

    # Find all non-sidecar artifacts (csv, npz, parquet) and check for sidecars
    while IFS= read -r artifact; do
        ARTIFACT_COUNT=$((ARTIFACT_COUNT + 1))
        local sidecar="${artifact}.prov.json"
        if [[ -f "$sidecar" ]]; then
            SIDECAR_COUNT=$((SIDECAR_COUNT + 1))
        else
            echo "  ORPHAN (no sidecar): ${artifact}"
            ORPHAN_COUNT=$((ORPHAN_COUNT + 1))
        fi
    done < <(find "$dir" -type f \( -name '*.csv' -o -name '*.npz' -o -name '*.parquet' \) ! -name '*.prov.json')
}

check_provenance "${LOCAL_DIR}/results"
check_provenance "${LOCAL_DIR}/data/processed"

echo ""
echo "============================================================"
echo "  SYNC SUMMARY"
echo "============================================================"
echo "  Artifacts found:     ${ARTIFACT_COUNT}"
echo "  With provenance:     ${SIDECAR_COUNT}"
echo "  Orphans (no sidecar): ${ORPHAN_COUNT}"
echo "============================================================"

if [[ "$ORPHAN_COUNT" -gt 0 ]]; then
    echo ""
    echo "WARNING: ${ORPHAN_COUNT} artifact(s) lack provenance sidecars."
    echo "Results without provenance are not interpretable."
    echo "Check that the pipeline ran to completion on the remote."
fi

# ---------------------------------------------------------------------------
# Optional: push to S3
# ---------------------------------------------------------------------------
if [[ -n "$S3_DEST" ]]; then
    log "Pushing to S3: ${S3_DEST}"
    aws s3 sync "${LOCAL_DIR}/results/" "${S3_DEST}/results/" --exclude '__pycache__/*'
    aws s3 sync "${LOCAL_DIR}/data/processed/" "${S3_DEST}/data/processed/" --exclude '__pycache__/*'
    log "S3 sync complete."
fi

# ---------------------------------------------------------------------------
# Optional: push to GCS
# ---------------------------------------------------------------------------
if [[ -n "$GCS_DEST" ]]; then
    log "Pushing to GCS: ${GCS_DEST}"
    gsutil -m rsync -r "${LOCAL_DIR}/results/" "${GCS_DEST}/results/"
    gsutil -m rsync -r "${LOCAL_DIR}/data/processed/" "${GCS_DEST}/data/processed/"
    log "GCS sync complete."
fi

log "Done."
