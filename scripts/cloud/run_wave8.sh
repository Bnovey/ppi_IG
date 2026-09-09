#!/usr/bin/env bash
# Wave 8: the T3 number.
#
# Everything upstream is now green. Wave 6 established:
#   random_weights  Spearman 0.0767 (threshold |rho| < 0.3) -- randomising 5035
#                   parameter tensors DESTROYS the attribution, so the gradient
#                   depends on learned weights and not on input geometry. This
#                   is the check that could have killed the project, and it had
#                   never once run before: it OOM'd on every prior attempt.
#   03_attribute    one backward pass at the FULL L=730 complex, 104 s,
#                   grad_chain (121, 384), no zero rows, per-residue norms
#                   spanning 0.0068 to 0.2056. That single pass is what is meant
#                   to replace ~615 GPU-hours of brute-force scanning.
#
# Stage 02 is the last blocker. It failed in wave 6 because it reused one
# cache_dir for every mutant and boltz handed back the wild-type features --
# caught by _check_featurised_sequences, not by review. Fixed, and the fix is
# now paired with wild-type MSA reuse, which is both ~5 hours cheaper and more
# correct: an MSA re-searched per mutant would make each delta reflect the
# substitution AND a different MSA, confounding exactly what the stage measures.
#
# The reference embedding is computed twice, once from the server MSA and once
# re-read from the written files, and the two are asserted equal to 1e-4. That
# runs after only two featurisations, so a broken reuse path fails in ~2 minutes
# rather than after 300.
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
  echo "############ W8 ${tag} :: $(date -u +%H:%M:%S) ############"
  $GPU "$@"
  echo "############ W8 ${tag} exit=$? :: $(date -u +%H:%M:%S) ############"
}

# ~300 substitutions over 16 variable positions, --force because wave 6 left no
# usable output and a partial file must not be mistaken for a complete one.
step embed_deltas \
  python3 scripts/02_embed_deltas.py --dataset "$DATASET" --force

# CPU, seconds each. The gradient from wave 6 is reused as-is: it is committed
# provenance-stamped output of an unchanged stage, and recomputing it would only
# add noise to a comparison.
step predict \
  python3 scripts/05_predict.py \
    --library "data/processed/${DATASET}_library.parquet" \
    --grad "results/${DATASET}_${SCORE}_${METHOD}_grad.npz" \
    --deltas "data/processed/${DATASET}_deltas.npz" \
    --out "results/${DATASET}_${SCORE}_${METHOD}_pred.csv"

# --scan omitted deliberately. T1 and T2 need the brute-force scan; T3 does not,
# and T3 is the number that says whether one backward pass tells a practitioner
# anything. Bar to beat on the AbBiBench leaderboard: Boltz-2 0.13, FoldX 0.12,
# AF3 -0.02, leaders ProteinMPNN 0.30 / ESM-IF1 0.28. 4fqi_h1 is the go/no-go
# dataset -- Boltz-2 scores 0.71 there, so a null result here is decisive.
step metrics \
  python3 scripts/06_metrics.py \
    --pred "results/${DATASET}_${SCORE}_${METHOD}_pred.csv" \
    --dataset "$DATASET" --method "$METHOD" \
    --out results/metrics.csv

echo "WAVE8 DONE :: $(date -u +%H:%M:%S)"
