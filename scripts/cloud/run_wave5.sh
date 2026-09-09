#!/usr/bin/env bash
# Wave 5: is the zeros baseline a singularity?
#
# Wave 4 killed two of the three candidate causes from MEMSCALE_RESULTS 6a and
# left a much sharper hypothesis in their place. At L=554 (H+L+A), bf16 unless
# noted, IGV_TRI_ATTN_CKPT=1:
#
#   arm         f(x)      f(baseline)  expected   sum(ig)     rel err
#   m=16 bf16   3.649815  12.028320    -8.378505  -18.189129  1.1710
#   m=16 fp32   3.651461  11.749341    -8.097880  -22.029915  1.7205
#   m=8  bf16   3.648877  12.028320    -8.379443  +21.602146  2.5781
#
#   - fp32 fails WORSE than bf16, so precision is not the cause. bf16 stays.
#   - The SIGN flips between m=8 and m=16. A smooth integrand cannot do that,
#     so this is not an unconverged integral; the integrand itself is wrong or
#     singular, and node placement is deciding the answer.
#
# The mechanism that fits: the baseline is torch.zeros_like(s_inputs), an
# all-zeros embedding that lies on no data manifold (f(baseline)=12.03 against
# f(x)=3.65). Gauss-Legendre never evaluates the endpoints but approaches them
# as m grows -- the smallest node is alpha~0.0198 at m=8, ~0.0052 at m=16,
# ~0.0013 at m=32. If the gradient blows up as alpha->0, every increase in m
# reaches further into the singularity, and the integral wanders in magnitude
# and in sign rather than converging. That also explains the L-scaling (rel err
# 1.171 at L=554 against 3.6387 at L=730): a longer complex puts more tokens at
# the degenerate point.
#
# Runs 1 and 2 test it directly by starting the path at 0.1*x instead of the
# origin, excising the suspect neighbourhood. Two outcomes, both worth having:
#
#   completeness PASSES  -> the singularity is proven and localised, and the fix
#                           is a better baseline, not a rewrite of the
#                           attribution. m=8 and m=16 agreeing is the proof that
#                           the integrand went smooth.
#   completeness FAILS   -> the baseline is exonerated too, and what is left is
#                           the gradient itself being wrong -- which run 3
#                           measures without trusting autograd at all.
set -uo pipefail
cd ~/IG

DATASET=4fqi_h1
SCORE=complex_pde
SUBSET=H,L,A

sanity () {
  tag="$1"; subset="$2"; msteps="$3"; scale="$4"; shift 4
  echo "############ W5 ${tag} :: subset=${subset} m=${msteps} scale=${scale} :: $(date -u +%H:%M:%S) ############"
  sudo docker run --rm --gpus all --shm-size=32g --ipc=host \
    -e IGV_TRI_ATTN_CKPT=1 -e IGV_AUTOCAST=bf16 \
    -v "$HOME/boltz_cache:/root/.boltz" -v "$HOME/IG:/app" -w /app igv:latest \
    python3 scripts/07_sanity.py \
      --dataset "$DATASET" --score "$SCORE" \
      --chain-subset "$subset" --checks completeness \
      --m-steps "$msteps" --baseline-scale "$scale" \
      --out "results/c_${tag}.json"
  echo "############ W5 ${tag} exit=$? :: $(date -u +%H:%M:%S) ############"
  # A failed gate exits 1, which is the expected outcome of several of these.
  # set -e stays off; the exit code is logged, not obeyed.
}

# 1-2: the decisive pair. Same path, two node counts. Agreement between them is
# the claim, not just each one clearing the threshold.
sanity s010_m16 "$SUBSET" 16 0.1
sanity s010_m8  "$SUBSET"  8 0.1

# 3: the shape of the integrand as alpha -> 0, measured directly. --skip-fd
# because the finite difference is meaningless out here: h=0.1 is larger than
# the alphas of interest, and shrinking it drops the window under the ~0.06
# noise floor. One backward per alpha.
echo "############ W5 pathprofile :: $(date -u +%H:%M:%S) ############"
sudo docker run --rm --gpus all --shm-size=32g --ipc=host \
  -e IGV_TRI_ATTN_CKPT=1 -e IGV_AUTOCAST=bf16 \
  -v "$HOME/boltz_cache:/root/.boltz" -v "$HOME/IG:/app" -w /app igv:latest \
  python3 scripts/09_path_profile.py \
    --dataset "$DATASET" --score "$SCORE" --chain-subset "$SUBSET" \
    --profile-steps 21 --skip-fd \
    --grad-alphas 0.001,0.005,0.02,0.05,0.1,0.3,0.6,0.9 \
    --out results/path_profile_L554_bf16.csv
echo "############ W5 pathprofile exit=$? :: $(date -u +%H:%M:%S) ############"

# 4: a dose-response point. If 0.1 helps but does not clear the threshold, the
# question is whether the error keeps falling as more of the origin is excised.
sanity s025_m16 "$SUBSET" 16 0.25

# 5: the cheapest L-ladder point, to test the L-scaling claim. L=230 is H+L
# alone -- an antibody with no antigen, degenerate biologically but a perfectly
# well-posed IG identity. Extrapolating the two-point L^4.1 fit predicts ~0.03,
# i.e. a PASS; treat that as a falsifiable prediction, not an expectation.
sanity L230_m16 H,L 16 0.0

echo "WAVE5 DONE :: $(date -u +%H:%M:%S)"
