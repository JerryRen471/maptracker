# Waymo Asymmetric ROI

The Waymo configuration uses the vehicle coordinate convention `x=forward`
and `y=left`. Its training region is:

```text
x: [-15, 45] meters
y: [-15, 15] meters
```

This keeps the original 1,800 square meter area while allocating more of it
to the camera-visible region in front of the vehicle.

## 1. Convert TFRecords

Use `waymo_pack`, which contains TensorFlow and the Waymo Open Dataset SDK.
Existing JPEG files are reused, so this command only regenerates map labels
and PKL metadata.

```bash
conda activate waymo_pack

python tools/data_converter/waymo_map_converter.py \
  --data-dir /data8012/waymo/training \
  --val-data-dir /data8012/waymo/validation \
  --out-dir /data/waymo_processed_xm15_x45_y15 \
  --x-min -15 \
  --y-min -15 \
  --x-max 45 \
  --y-max 15 \
  --frame-stride 5 \
  --num-workers 8 \
  --reuse-images-from /data/waymo_processed_v3
```

## 2. Pack MapTracker Annotations

Use the `maptracker` environment for repository tooling and training.

```bash
conda activate maptracker

python pack_waymo_for_maptracker.py \
  --v3-dir /data/waymo_processed_xm15_x45_y15 \
  --out-dir /data/waymo_maptracker_xm15_x45_y15
```

The packer reads `metadata.image_root`, so `--img-root` is not required.

## 3. Generate GT Tracks

Both splits need tracking metadata because the training dataset enables
multi-frame matching. The command processes train and val in one run.

```bash
python tools/tracking/prepare_gt_tracks.py \
  plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py \
  --out-dir /data/waymo_maptracker_xm15_x45_y15/track_visualization
```

## 4. Train

Use the training controller to launch detached jobs. It validates predecessor
checkpoints, prevents duplicate runs, writes `stdout.log` in each work
directory, and returns the background PID:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash tools/train_waymo_asym.sh start stage1
```

Launch or inspect individual stages:

```bash
bash tools/train_waymo_asym.sh start stage2
bash tools/train_waymo_asym.sh status stage2
bash tools/train_waymo_asym.sh logs stage2
```

Run all missing stages sequentially in a detached controller:

```bash
bash tools/train_waymo_asym.sh chain
```

`chain` skips stages that already completed successfully. Stage 2 starts only
after the stage 1 checkpoint is valid, and stage 3 starts only after stage 2
finishes successfully. Disconnecting SSH or VPN does not stop these background
jobs.
