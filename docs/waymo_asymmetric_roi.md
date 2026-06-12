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

Use distinct work directories so checkpoints from the old symmetric ROI
cannot be loaded accidentally.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 \
bash tools/dist_train.sh \
  plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py \
  4 \
  --work-dir /data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_asym_roi
```

Stage 2 and stage 3 configs load checkpoints from the corresponding
`*_asym_roi` work directories.
