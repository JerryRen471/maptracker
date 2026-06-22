# Waymo 三阶段训练流水线脚本使用说明

本文说明如何使用 `tools/run_waymo_pipeline.sh` 自动运行 Waymo 数据转化、MapTracker 数据打包、GT track 生成、Stage1/Stage2/Stage3 训练、评测和可视化。

## 1. 脚本位置

在 `maptracker` 目录下运行：

```bash
cd /home/saimo/jerry/codex_workplace/maptr/maptracker
```

脚本路径：

```bash
tools/run_waymo_pipeline.sh
```

## 2. 最小运行命令

```bash
bash tools/run_waymo_pipeline.sh \
  --stage1-config plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py \
  --gpus 0,1,2,3 \
  --num-gpus 4
```

只需要显式指定 Stage1 config。脚本会按文件名自动推断：

```text
stage1_bev_pretrain -> stage2_warmup
stage1_bev_pretrain -> stage3_joint_finetune
```

也就是默认使用：

```text
plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage2_warmup.py
plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune.py
```

## 3. 推荐先 dry-run

第一次使用时建议先打印命令，不实际执行：

```bash
bash tools/run_waymo_pipeline.sh \
  --dry-run \
  --stage1-config plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py \
  --gpus 0,1,2,3 \
  --num-gpus 4
```

重点检查输出里的这些字段：

```text
STAGE1_CONFIG
STAGE2_CONFIG
STAGE3_CONFIG
PROCESSED_DIR
MAPTRACKER_DIR
WORK_ROOT
ROI
GPUS
NUM_GPUS
```

脚本不会自动改写 config 里的 `ann_file`。如果你修改了 `--maptracker-dir`，需要确认 config 中的数据路径也指向同一份 MapTracker pkl。

## 4. 默认数据路径和环境

脚本默认值如下：

```text
Waymo train tfrecord: /data8012/waymo/training
Waymo val tfrecord:   /data8012/waymo/validation
Converter output:     /data/waymo_processed_xm15_x45_y15
MapTracker pkl:       /data/waymo_maptracker_xm15_x45_y15
Work dir root:        /data/maptr_workspace/work_dirs
Conda home:           /root/miniconda3
Waymo converter env:  waymo_pack
Training env:         maptracker
ROI:                  x=[-15,45], y=[-15,15]
```

默认 ROI 是前向偏置的非对称 BEV 范围：

```text
x_min=-15, y_min=-15, x_max=45, y_max=15
```

## 5. 完整流程

脚本按顺序执行以下步骤：

```text
1. conda activate waymo_pack
   python tools/data_converter/waymo_map_converter.py ...

2. conda activate maptracker
   python pack_waymo_for_maptracker.py ...

3. python tools/tracking/prepare_gt_tracks.py ...

4. bash tools/dist_train.sh stage1_config ...

5. bash tools/dist_train.sh stage2_config ...
   --cfg-options load_from=<stage1_work_dir>/latest.pth

6. bash tools/dist_train.sh stage3_config ...
   --cfg-options load_from=<stage2_work_dir>/latest.pth

7. python tools/test.py stage3_config <stage3_work_dir>/latest.pth --eval ...

8. python tools/visualization/vis_global.py ... --option vis-pred

9. python tools/visualization/vis_global.py ... --option vis-gt
```

Stage2 会加载 Stage1 的 `latest.pth`，Stage3 会加载 Stage2 的 `latest.pth`。

## 6. 输出目录

假设 `--exp-tag asym_roi`，默认 work_dir 为：

```text
/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_asym_roi
/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage2_warmup_asym_roi
/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_asym_roi
```

评测结果默认输出到：

```text
<stage3_work_dir>/eval
```

预测可视化默认输出到：

```text
<stage3_work_dir>/visualization/pred
```

GT 可视化默认输出到：

```text
<stage3_work_dir>/visualization/gt
```

## 7. 常用参数

改 GPU：

```bash
--gpus 4,5,6,7 --num-gpus 4
```

改实验后缀：

```bash
--exp-tag xm15_x45_y15_v1
```

改 Waymo tfrecord 输入目录：

```bash
--data-dir /data8012/waymo/training \
--val-data-dir /data8012/waymo/validation
```

改数据输出目录：

```bash
--processed-dir /data/waymo_processed_xm15_x45_y15 \
--maptracker-dir /data/waymo_maptracker_xm15_x45_y15
```

改训练输出根目录：

```bash
--work-root /data/maptr_workspace/work_dirs
```

改 ROI：

```bash
--x-min -15 --y-min -15 --x-max 45 --y-max 15
```

指定 Stage2/Stage3 config：

```bash
--stage2-config plugin/configs/maptracker/waymo_5cam/xxx_stage2_warmup.py \
--stage3-config plugin/configs/maptracker/waymo_5cam/xxx_stage3_joint_finetune.py
```

只可视化指定 scene：

```bash
--scene-id segment-xxx \
--scene-id segment-yyy
```

## 8. 跳过部分步骤

如果数据已经转好，只重新训练：

```bash
bash tools/run_waymo_pipeline.sh \
  --stage1-config plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py \
  --skip-convert \
  --skip-pack \
  --skip-gt-tracks \
  --gpus 0,1,2,3 \
  --num-gpus 4
```

如果训练已经完成，只重新评测和可视化：

```bash
bash tools/run_waymo_pipeline.sh \
  --stage1-config plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py \
  --skip-convert \
  --skip-pack \
  --skip-gt-tracks \
  --skip-train \
  --gpus 0
```

如果只想跑到评测，不做可视化：

```bash
--skip-vis
```

如果只想准备数据，不训练：

```bash
--skip-train --skip-test --skip-vis
```

## 9. 环境变量覆盖

部分默认值也可以通过环境变量覆盖：

```bash
CONDA_HOME=/root/miniconda3 \
WAYMO_ENV=waymo_pack \
TRAIN_ENV=maptracker \
WAYMO_EXP_TAG=asym_roi \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/run_waymo_pipeline.sh \
  --stage1-config plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py \
  --num-gpus 4
```

如果同时设置了 `CUDA_VISIBLE_DEVICES` 和 `--gpus`，以 `--gpus` 为准。

## 10. 注意事项

1. `waymo_map_converter.py` 需要 TensorFlow 和 Waymo Open Dataset SDK，所以默认在 `waymo_pack` 环境中运行。
2. 打包、GT track、训练、评测、可视化默认在 `maptracker` 环境中运行。
3. 脚本是前台顺序执行。SSH 断开会影响运行，长任务建议配合 `tmux` 或 `screen`。
4. Stage2 依赖 Stage1 的 `latest.pth`，Stage3 依赖 Stage2 的 `latest.pth`。
5. `tools/test.py` 默认读取 Stage3 work_dir 下的 `latest.pth`。
6. 可视化预测结果默认读取 `<stage3_work_dir>/eval/pos_predictions.pkl`。
7. 可视化 GT 默认读取 `<maptracker_dir>/waymo_map_infos_val_gt_tracks.pkl`。
8. 脚本不会检查 pkl 中的 ROI metadata 是否和 config 完全一致，运行前需要人工确认数据目录和 config 对应。

