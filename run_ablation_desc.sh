#!/usr/bin/env bash
set -euo pipefail

python tune.py \
  --dataset rest \
  --search-space ablation_desc \
  --validation-seed-pairs 2024:2024,42:42,3407:3407 \
  --valid-ratio 0.15 \
  --final-seeds 2024,42,3407 \
  --epochs 12 \
  --final-epochs 15 \
  --patience 3 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --amp \
  --gradient-checkpointing \
  --run-final \
  --output-dir tuning_ablation_desc \
  --final-output-dir outputs_ablation_desc
