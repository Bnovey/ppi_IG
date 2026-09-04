#!/usr/bin/env bash
# Wave 3: the group knob, in the direction the data actually points.
#
# Wave 2 tested group 8 and 16 on the theory that a larger group would now be
# cheap, because IGV_TRI_ATTN_CKPT removes the retained softmax that a larger
# group used to inflate. That was wrong, and the measurement is unambiguous:
#
#   bf16, group 4  (default), chunk 128 -> 55.22 GiB COMPLETED
#   bf16, group 16,           chunk 16  -> 78.22 GiB OOM
#
# 23+ GiB worse. So the dominant term in the group knob is the RECOMPUTE
# TRANSIENT (a bigger group means more blocks recomputed simultaneously in the
# backward), not the retained boundary tensors. Smaller groups are better, which
# also matches ERRORS_LOG entry 12, where group=1 was the lowest of its nine
# rows (77.64) -- that table's boundary-z column pointed one way and its total
# pointed the other, and the total is what matters.
#
# fp32 needs to close only 0.37 GiB (fitted requirement 79.57 vs 79.20 usable),
# and an fp32 config is worth real effort: it needs no score re-derivation.
set -uo pipefail
cd ~/IG
DOCKER="sudo docker run --rm --gpus all --shm-size=32g --ipc=host \
  -v $HOME/boltz_cache:/root/.boltz -v $HOME/IG:/app -w /app igv:latest"

run () {
  tag="$1"; shift
  echo "############ W3 ${tag} :: $(date -u +%H:%M:%S) :: $* ############"
  $DOCKER python3 scripts/08_memscale.py --dataset 4fqi_h1 --score complex_pde \
      --sizes 730 --out "results/w3_${tag}.csv" "$@"
  echo "############ W3 ${tag} exit=$? :: $(date -u +%H:%M:%S) ############"
}

run f32_g1  --tri-attn-ckpt 1 --pf-group-size 1
run f32_g2  --tri-attn-ckpt 1 --pf-group-size 2
run bf16_g1 --tri-attn-ckpt 1 --pf-group-size 1 --autocast bf16
echo "WAVE3 DONE :: $(date -u +%H:%M:%S)"
