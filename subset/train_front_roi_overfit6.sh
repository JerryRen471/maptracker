#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python tools/train.py \
    subset/debug_overfit6_stage1_front_roi.py \
    --gpus 4
