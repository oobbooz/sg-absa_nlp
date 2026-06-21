#!/usr/bin/env bash
set -euo pipefail

python tune.py \
  --dataset rest \
  --model microsoft/deberta-v3-base \
  --batch-size 8 \
  --search-space refined \
  --validation-seed-pairs 2024:2024,42:42,3407:3407 \
  --final-seeds 2024,42,3407 \
  --split-seed 2024 \
  --amp \
  --run-final
