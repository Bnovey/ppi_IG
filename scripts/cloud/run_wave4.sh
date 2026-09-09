#!/usr/bin/env bash
# Wave 4: separate the three candidate causes of the completeness failure.
#
# MEMSCALE_RESULTS section 6a: at L=730 under IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16,
# `completeness` returned 3.6387 (threshold 0.05) -- sum(ig) = -39.87 against an
# expected -8.59, a 4.64x overshoot. It ran to completion with zero exceptions, so
# this is a threshold failure and not a crash. Three candidate causes were left
# unseparated:
#
#   (1) bf16 damaged the gradient. No fp32 control exists at 730 and none can:
#       fp32's fitted requirement there is 79.57 GiB against 79.20 usable.
#   (2) m=16 Gauss-Legendre quadrature is not converged.
#   (3) the zeros baseline is far off-manifold -- f(baseline)=12.449 against
#       f(x)=3.855 is a different regime entirely.
#
# L=554 (whole-chain subset H+L+A of 4fqi_hlab) is where the control becomes
# possible: fp32 measured 56.21 GiB there, bf16 39.06 GiB, both with headroom.
# Chunk profile is "large" on both sides of this comparison (auto selects it for
# any L > 384), so 554 and 730 are the same algorithm, not two.
#
# The four runs below decide it:
#   1. bf16 m=16 -- does the failure reproduce at 554 at all? Cheapest arm, so a
#      broken flag path costs the least. If completeness PASSES here, the failure
#      is L-dependent and neither dtype nor quadrature.
#   2. fp32 m=16 -- the dtype control that 730 cannot provide. If fp32 also fails
#      at ~3.6, bf16 is exonerated and cause (1) is dead: the flaw is
#      pre-existing and the OOMs were masking it all along.
#   3. bf16 m=8  \_ quadrature convergence. If the error falls toward zero across
#   4. bf16 m=32 /  8 -> 16 -> 32, cause (2) is live; if it plateaus near 3.6,
#                   quadrature is not the problem and cause (3) is what is left.
#
# Ordered so the decisive pair (1, 2) lands first: a preemption or a stop after
# ~45 min still answers the dtype question.
#
# check_m_sweep is deliberately NOT used here. It compares gradient-magnitude
# RANKINGS across m, which is a different question -- convergence of the ranking,
# not of the completeness identity.
set -uo pipefail
cd ~/IG

L_SUBSET=H,L,A          # 554 tokens: A=324 H=121 L=109 (chain B=176 dropped)
DATASET=4fqi_h1
SCORE=complex_pde

run () {
  tag="$1"; msteps="$2"; shift 2
  out="results/c554_${tag}.json"
  echo "############ W4 ${tag} :: m=${msteps} :: $(date -u +%H:%M:%S) ############"
  sudo docker run --rm --gpus all --shm-size=32g --ipc=host \
    -e IGV_TRI_ATTN_CKPT=1 "$@" \
    -v "$HOME/boltz_cache:/root/.boltz" -v "$HOME/IG:/app" -w /app igv:latest \
    python3 scripts/07_sanity.py \
      --dataset "$DATASET" --score "$SCORE" \
      --chain-subset "$L_SUBSET" \
      --checks completeness \
      --m-steps "$msteps" \
      --out "$out"
  echo "############ W4 ${tag} exit=$? :: $(date -u +%H:%M:%S) ############"
  # 07_sanity exits 1 on a failed gate. That is the EXPECTED outcome of three of
  # these four runs, so `set -e` is off and the exit code is logged, not obeyed.
}

# 1-2: the decisive dtype pair, same L, same m.
run bf16_m16 16 -e IGV_AUTOCAST=bf16
run f32_m16  16
# 3-4: quadrature convergence in the arm that has headroom to spare.
run bf16_m8   8 -e IGV_AUTOCAST=bf16
run bf16_m32 32 -e IGV_AUTOCAST=bf16

echo "WAVE4 DONE :: $(date -u +%H:%M:%S)"
grep -h '"name": "completeness"' -A 3 results/c554_*.json 2>/dev/null || true
