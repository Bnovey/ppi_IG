#!/usr/bin/env bash
set -euo pipefail

# Pipeline driver for SKEMPI and DMS complexes.
# Skips stages 00/01 (AbBiBench-only) and resolves via the DMS/SKEMPI registry.
#
# SKEMPI (1JTG brute-force scan):
#   DATASET=1JTG CHAIN=B POSITIONS=skempi bash scripts/run_complex.sh
#
# DMS (6M0J saturation):
#   DATASET=spike_rbd CHAIN=E POSITIONS=dms_interface bash scripts/run_complex.sh

DATASET="${DATASET:-}"
CHAIN="${CHAIN:-}"
SCORE="${SCORE:-complex_pde}"
METHOD="${METHOD:-ig}"
M_STEPS="${M_STEPS:-32}"
BASELINE="${BASELINE:-mean_aa}"
POSITIONS="${POSITIONS:-}"
INTERFACE_CUTOFF="${INTERFACE_CUTOFF:-5.0}"
PY="${PY:-python3}"

DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    fi
done

if [ -z "$DATASET" ]; then
    echo "ERROR: DATASET must be set (e.g. DATASET=1JTG or DATASET=spike_rbd)" >&2
    exit 1
fi
if [ -z "$CHAIN" ]; then
    echo "ERROR: CHAIN must be set (e.g. CHAIN=B or CHAIN=E)" >&2
    exit 1
fi

DELTAS="data/processed/${DATASET}_deltas.npz"
GRAD="results/${DATASET}_${SCORE}_${METHOD}_grad.npz"
SCAN="results/${DATASET}_${SCORE}_scan.csv"
PRED="results/${DATASET}_${SCORE}_${METHOD}_pred.csv"
HOTSPOTS="results/${DATASET}_hotspots.json"

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

# Fail fast: verify DATASET is in the DMS or SKEMPI registry before any GPU work.
log "Checking registry for DATASET=${DATASET} CHAIN=${CHAIN}"
"$PY" -c "
import sys; sys.path.insert(0, 'src')
from igv.dms import resolve_pdb_complex
from pathlib import Path
resolve_pdb_complex('${DATASET}', '${CHAIN}', Path('data/raw'))
"
_RESOLVED=$("$PY" -c "
import sys; sys.path.insert(0, 'src')
from igv.dms import resolve_pdb_complex
from pathlib import Path
r = resolve_pdb_complex('${DATASET}', '${CHAIN}', Path('data/raw'))
print(r.data_source, r.struct_name.upper())
")
DATA_SOURCE="${_RESOLVED%% *}"
PDB_ID="${_RESOLVED##* }"
echo ""

POSITIONS_ARGS=""
if [ -n "$POSITIONS" ]; then
    POSITIONS_ARGS="--positions $POSITIONS --interface-cutoff $INTERFACE_CUTOFF"
fi

log "Starting complex pipeline: DATASET=${DATASET} CHAIN=${CHAIN} SCORE=${SCORE} METHOD=${METHOD} SOURCE=${DATA_SOURCE}"
echo ""

log "Stage 07: Sanity checks (gate -- abort if this fails)"
run "$PY" scripts/07_sanity.py --dataset "$DATASET" --chain "$CHAIN" --score "$SCORE"
echo ""

log "Stage 02: Compute embedding deltas [GPU]"
run "$PY" scripts/02_embed_deltas.py --dataset "$DATASET" --chain "$CHAIN" $POSITIONS_ARGS
echo ""

log "Stage 03: Gradient attribution [GPU]"
run "$PY" scripts/03_attribute.py --dataset "$DATASET" --chain "$CHAIN" --score "$SCORE" \
    --method "$METHOD" --m-steps "$M_STEPS" --baseline "$BASELINE"
echo ""

log "Stage 04: Brute-force mutation scan [GPU]"
run "$PY" scripts/04_scan.py --dataset "$DATASET" --chain "$CHAIN" --score "$SCORE" $POSITIONS_ARGS
echo ""

log "Stage 05: Predict mutant scores"
run "$PY" scripts/05_predict.py \
    --dataset "$DATASET" \
    --grad "$GRAD" \
    --deltas "$DELTAS" \
    --out "$PRED"
echo ""

if [ "$DATA_SOURCE" = "skempi" ]; then
    log "Stage 10: SKEMPI hot-spot analysis"
    run "$PY" scripts/10_skempi_hotspots.py \
        --complex "$PDB_ID" \
        --grad "$GRAD" \
        --chain "$CHAIN" \
        --cache-dir data/raw \
        --out "$HOTSPOTS"
    echo ""
fi

if [ "$DATA_SOURCE" = "dms" ]; then
    log "Stage 11: Within-position analysis (DMS only)"
    run "$PY" scripts/11_within_position.py \
        --pred "$PRED" \
        --dataset "$DATASET" \
        --cache-dir data/raw \
        --interface-cutoff "$INTERFACE_CUTOFF"
    echo ""
fi

log "Pipeline complete."
