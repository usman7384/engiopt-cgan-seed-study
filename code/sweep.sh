#!/usr/bin/env bash
# 4 learning-rate arms x 10 seeds of the EngiOpt cgan_cnn_2d baseline on beams2d.
# Everything except the two learning rates is held at the published 11-seed configuration
# (batch 32, latent 32, 200 epochs), so the arms differ in one thing only.
# Seeds run in seed-major order: if the sweep is interrupted, every arm has the same seeds done.
# Run from a folder that holds this script, run_one.py and a clone of EngiOpt (./EngiOpt).
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 MPLBACKEND=Agg WANDB_MODE=disabled

# arm name | lr_gen | lr_disc
ARMS=(
  "default_g1e-4_d4e-4|0.0001|0.0004"
  "published11_g3e-4_d3e-4|0.0003|0.0003"
  "stable_g1e-3_d1e-4|0.001|0.0001"
  "discfast_g1e-4_d1e-3|0.0001|0.001"
)
SEEDS="1 2 3 4 5 6 7 8 9 10"

for s in $SEEDS; do
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r name lg ld <<< "$arm"
    $PY run_one.py --arm "$name" --seed "$s" --lr-gen "$lg" --lr-disc "$ld" \
      --n-epochs 200 --batch-size 32 --latent-dim 32
  done
done

for arm in "${ARMS[@]}"; do
  IFS='|' read -r name lg ld <<< "$arm"
  rm -f "results_$name.csv"
  $PY -m engiopt.evaluate --problem-id beams2d --generators cgan_cnn_2d --seeds $SEEDS \
    --model-source local --local-model-dir "$ROOT/runs/$name/seed_{seed}" \
    --output-csv "$ROOT/results_$name.csv"
done
