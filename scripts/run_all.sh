#!/usr/bin/env bash
set -euo pipefail

# Full study driver for a GPU box.
# Usage:
#   bash scripts/run_all.sh                 # run with defaults
#   DATASET=4fqi_h3 bash scripts/run_all.sh # override dataset
#   DRY_RUN=1 bash scripts/run_all.sh       # print commands without executing
#   bash scripts/run_all.sh --dry-run       # also accepted

DATASET="${DATASET:-4fqi_h1}"
SCORE="${SCORE:-complex_pde}"
METHOD="${METHOD:-plain_grad}"
PY="${PY:-python3}"

DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    fi
done

LIBRARY="data/processed/${DATASET}_library.parquet"
DELTAS="data/processed/${DATASET}_deltas.npz"
GRAD="results/${DATASET}_${SCORE}_${METHOD}_grad.npz"
SCAN="results/${DATASET}_${SCORE}_scan.csv"
PRED="results/${DATASET}_${SCORE}_${METHOD}_pred.csv"
METRICS="results/metrics.csv"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run() {
    echo "+ $*"
    if [ "$DRY_RUN" = "1" ]; then
        return
    fi
    "$@"
}

log "Starting full pipeline: DATASET=${DATASET} SCORE=${SCORE} METHOD=${METHOD}"
echo ""

log "Stage 00: Fetch raw data"
run "$PY" scripts/00_fetch_data.py --datasets "$DATASET"
echo ""

log "Stage 01: Build mutant library"
run "$PY" scripts/01_build_library.py --dataset "$DATASET"
echo ""

log "Stage 07: Sanity checks (gate -- abort if this fails)"
run "$PY" scripts/07_sanity.py --dataset "$DATASET" --score "$SCORE"
echo ""

log "Stage 02: Compute embedding deltas [GPU]"
run "$PY" scripts/02_embed_deltas.py --dataset "$DATASET"
echo ""

log "Stage 03: Gradient attribution [GPU]"
run "$PY" scripts/03_attribute.py --dataset "$DATASET" --score "$SCORE" --method "$METHOD"
echo ""

log "Stage 04: Brute-force mutation scan [GPU]"
run "$PY" scripts/04_scan.py --dataset "$DATASET" --score "$SCORE"
echo ""

log "Stage 05: Predict mutant scores"
run "$PY" scripts/05_predict.py \
    --library "$LIBRARY" \
    --grad "$GRAD" \
    --deltas "$DELTAS" \
    --out "$PRED"
echo ""

log "Stage 06: Compute metrics"
run "$PY" scripts/06_metrics.py \
    --pred "$PRED" \
    --scan "$SCAN" \
    --dataset "$DATASET" \
    --method "$METHOD" \
    --out "$METRICS"
echo ""

log "Pipeline complete."
