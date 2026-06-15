#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$repo_root/tools/train_waymo_asym.sh"

help_output="$(bash "$script" help)"
grep -q 'start stage1|stage2|stage3' <<<"$help_output"
grep -q 'status stage1|stage2|stage3' <<<"$help_output"
grep -q 'logs stage1|stage2|stage3' <<<"$help_output"
grep -q 'chain' <<<"$help_output"

dry_run_output="$(
    WAYMO_TRAIN_DRY_RUN=1 \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    bash "$script" start stage2
)"
grep -q 'maptracker_waymo_5cam_5frame_span10_stage2_warmup.py' \
    <<<"$dry_run_output"
grep -q 'maptracker_waymo_5cam_5frame_span10_stage2_warmup_asym_roi' \
    <<<"$dry_run_output"
grep -q 'PORT=29513' <<<"$dry_run_output"
grep -q 'BACKGROUND_PID=' <<<"$dry_run_output"

echo "train_waymo_asym interface tests passed"
