#!/usr/bin/env bash
# Wave 6: stop diagnosing, get a T3 number.
#
# Wave 4 finished the completeness question at L=554 (H+L+A), bf16,
# IGV_TRI_ATTN_CKPT=1:
#
#   m=8    sum(ig) = +21.602146   rel err 2.5781   FAIL
#   m=16   sum(ig) = -18.189129   rel err 1.1710   FAIL
#   m=32   sum(ig) =  -8.741840   rel err 0.0442   PASS   (expected -8.371882)
#   fp32 m=16                     rel err 1.7205   FAIL   (worse than bf16)
#
# Conclusions, in the order they matter:
#   - bf16 is exonerated: fp32 fails WORSE at the same L and m. The memory fix
#     is not what broke completeness, so IGV_AUTOCAST=bf16 stays.
#   - The integral converges, but slowly and non-monotonically. m=16 was simply
#     not enough. At L=730 (rel err 3.6387 at m=16) expect to need m>=64.
#   - Nothing here touches plain_grad, which has no path, no baseline and no
#     quadrature -- and which is the Makefile default and the method behind the
#     efficiency claim (~16 backward passes against ~615 GPU-hours). That claim
#     is one pass per dataset, not IG at m=16.
#
# So the gating question was never completeness. attrib.py:402-404 is explicit
# that per-substitution predictions use .grad and NOT the .ig product that
# completeness constrains. What actually gates the science is whether the
# gradient carries learned signal at all, and what T3 comes out at.
#
# 06_metrics takes --scan as OPTIONAL, so T3 (attribution against 184,500
# measured affinities) needs no brute-force scan. One backward pass covers every
# mutant. That is the whole efficiency argument, and it is cheap to test.
set -uo pipefail
cd ~/IG

DATASET=4fqi_h1
SCORE=complex_pde
METHOD=plain_grad
GPU="sudo docker run --rm --gpus all --shm-size=32g --ipc=host \
  -e IGV_TRI_ATTN_CKPT=1 -e IGV_AUTOCAST=bf16 \
  -v $HOME/boltz_cache:/root/.boltz -v $HOME/IG:/app -w /app igv:latest"

step () {
  tag="$1"; shift
  echo "############ W6 ${tag} :: $(date -u +%H:%M:%S) ############"
  $GPU "$@"
  echo "############ W6 ${tag} exit=$? :: $(date -u +%H:%M:%S) ############"
}

# 1. The check that can kill the project, and has never once run -- it OOM'd on
#    every previous attempt. If a randomly initialised Boltz-2 reproduces the
#    trained attribution, the gradient is reading input geometry rather than
#    anything the model learned, and no T3 number means a thing. Run at L=554
#    rather than 730 because check_random_weights deepcopies the model, which
#    doubles resident model VRAM inside the most memory-constrained stage there
#    is (noted in 07_sanity.py's own comment). 554 has ~40 GiB of headroom.
step random_weights \
  python3 scripts/07_sanity.py --dataset "$DATASET" --score "$SCORE" \
    --chain-subset H,L,A --checks random_weights,dead_target \
    --out results/w6_randweights_L554.json

# 2. Why m=16 was not enough, as a measurement rather than a story: D_analytic
#    across three decades of alpha. --skip-fd because out at alpha=0.001 the
#    finite difference is meaningless (h=0.1 exceeds the alphas of interest, and
#    shrinking it drops the window under the ~0.06 noise floor). The 21-point
#    F(alpha) profile it also writes lands on 0.05 spacing, which shares nodes
#    with the analytic alphas at 0.05/0.1/0.3/0.6/0.9 -- so autograd can be
#    checked against a central difference of the profile OFFLINE, for free, and
#    a chain-rule defect in the checkpoint nesting (which the fp32 control did
#    NOT rule out, since both arms checkpoint) would show up as a systematic
#    ratio away from 1.
step path_profile \
  python3 scripts/09_path_profile.py --dataset "$DATASET" --score "$SCORE" \
    --chain-subset H,L,A --profile-steps 21 --skip-fd \
    --grad-alphas 0.001,0.005,0.02,0.05,0.1,0.3,0.6,0.9 \
    --out results/path_profile_L554_bf16.csv

# 3-4. The actual pipeline, at the FULL complex (L=730, all four chains). No
#      --chain-subset: the subset flag exists for diagnostics, and the science
#      runs on the real thing. plain_grad is one backward pass.
step embed_deltas \
  python3 scripts/02_embed_deltas.py --dataset "$DATASET"

step attribute \
  python3 scripts/03_attribute.py --dataset "$DATASET" --score "$SCORE" \
    --method "$METHOD"

# 5-6. CPU, seconds. --scan is deliberately omitted: T1/T2 need the brute-force
#      scan, T3 does not, and T3 is the number that says whether one backward
#      pass tells a practitioner anything.
step predict \
  python3 scripts/05_predict.py \
    --library "data/processed/${DATASET}_library.parquet" \
    --grad "results/${DATASET}_${SCORE}_${METHOD}_grad.npz" \
    --deltas "data/processed/${DATASET}_deltas.npz" \
    --out "results/${DATASET}_${SCORE}_${METHOD}_pred.csv"

step metrics \
  python3 scripts/06_metrics.py \
    --pred "results/${DATASET}_${SCORE}_${METHOD}_pred.csv" \
    --dataset "$DATASET" --method "$METHOD" \
    --out results/metrics.csv

echo "WAVE6 DONE :: $(date -u +%H:%M:%S)"
