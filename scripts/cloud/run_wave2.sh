#!/usr/bin/env bash
# Wave 2: single-point probes at L=730 only (MSAs are cached, ~2-3 min each).
#
# Hypothesis under test. ERRORS_LOG.md entry 12 found the pairformer group size
# to be a break-even: a LARGER group shrank the retained checkpoint-boundary z
# (34.30 -> 4.32 GiB from group=1 to group=8) while inflating the retained
# softmax transient (11.59 -> 25.22 GiB), and the two cancelled. That is why
# entry 12 concluded the knobs "only relocate the memory".
#
# IGV_TRI_ATTN_CKPT deletes the softmax term -- it is recomputed in backward
# rather than retained -- so the penalty that made large groups unattractive is
# gone and the boundary-z saving should no longer be cancelled. The ckpt ladder
# ran at the DEFAULT group size of 4, so this combination has never been tried.
#
# --pf-chunk 16 additionally shrinks the transient that is the failing
# allocation in every recorded OOM (1.0164 GiB at chunk=128 -> 0.1271 at 16).
set -uo pipefail
cd ~/IG
DOCKER="sudo docker run --rm --gpus all --shm-size=32g --ipc=host \
  -v $HOME/boltz_cache:/root/.boltz -v $HOME/IG:/app -w /app igv:latest"

run () {
  tag="$1"; shift
  echo "############ W2 ${tag} :: $(date -u +%H:%M:%S) :: $* ############"
  $DOCKER python3 scripts/08_memscale.py --dataset 4fqi_h1 --score complex_pde \
      --sizes 730 --out "results/w2_${tag}.csv" "$@"
  echo "############ W2 ${tag} exit=$? :: $(date -u +%H:%M:%S) ############"
}

run g8       --tri-attn-ckpt 1 --pf-group-size 8
run g16      --tri-attn-ckpt 1 --pf-group-size 16
run g8_c16   --tri-attn-ckpt 1 --pf-group-size 8  --pf-chunk 16
run g16_bf16 --tri-attn-ckpt 1 --pf-group-size 16 --pf-chunk 16 --autocast bf16
echo "WAVE2 DONE :: $(date -u +%H:%M:%S)"
