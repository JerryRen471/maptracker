#!/usr/bin/env bash
set -e

# Use a dedicated port to avoid conflict with other jobs (e.g. Waymo on 29513)
export PORT=29514

# Stage 1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
  plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage1_bev_pretrain.py 4

# Stage 2（Stage 1 完成后）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
  plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage2_warmup.py 4

# Stage 3（Stage 2 完成后）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
  plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py 4
