#!/usr/bin/env bash
# Reproduce every number and figure in the README.
#
#   bash scripts/run_all_experiments.sh                 # full run
#   JOBS=6 bash scripts/run_all_experiments.sh          # more parallel jobs (needs ~0.4 GB RAM each)
#   STEPS=50000 SEEDS="0 1" bash scripts/run_all_experiments.sh   # quick check
#
# Trains 7 configurations x 5 seeds (one job per seed, JOBS at a time), then runs
# the evaluation suite. Re-running resumes: finished seeds are skipped.
set -euo pipefail

PY=${PY:-python}
STEPS=${STEPS:-300000}
SEEDS=${SEEDS:-"0 1 2 3 4"}
RUNS=${RUNS:-runs}
JOBS=${JOBS:-4}
mkdir -p "$RUNS/logs"

CONFIGS=(
  "ppo            --algo ppo --history-len 4"
  "ppo_no_memory  --algo ppo --history-len 1"
  "ppo_top1       --algo ppo --history-len 4 --obs-mode onehot"
  "ppo_privileged --algo ppo --history-len 4 --privileged"
  "recurrent_ppo  --algo recurrent_ppo --history-len 1"
  "a2c            --algo a2c --history-len 4"
  "dqn            --algo dqn --history-len 4"
)

for seed in $SEEDS; do
  for config in "${CONFIGS[@]}"; do
    echo "$seed $config"
  done
done | xargs -P "$JOBS" -L 1 bash -c '
  seed=$0; tag=$1; shift
  '"$PY"' -m emotion_rl.train --tag "$tag" --seeds "$seed" --timesteps '"$STEPS"' \
    --out '"$RUNS"' --skip-existing "$@" > '"$RUNS"'/logs/${tag}_${seed}.log 2>&1 \
    && echo "[done] $tag seed $seed" || echo "[FAILED] $tag seed $seed (see '"$RUNS"'/logs)"
'
echo "[Experiments] training finished"

$PY -m emotion_rl.evaluate --runs "$RUNS" --out results
