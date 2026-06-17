# Waymo 6-Scene Overfit Subset

This folder contains the isolated files for a small overfit experiment on the
asymmetric Waymo ROI `x=[-15, 45], y=[-15, 15]`.

Run from the `maptracker` repository root on the training machine.

## 1. Build the subset pkl files

```bash
python subset/make_waymo_overfit6_subset.py
```

Default input:

```text
/data/waymo_maptracker_xm15_x45_y15
```

Default output:

```text
/data/waymo_maptracker_xm15_x45_y15_overfit6
```

The script writes:

```text
waymo_map_infos_train.pkl
waymo_map_infos_val.pkl
selected_scenes.txt
subset_summary.pkl
```

`train` and `val` intentionally contain the same selected train scenes. This is
for overfit sanity checking, not for measuring generalization.

## 2. Generate GT tracks for the subset

```bash
python tools/tracking/prepare_gt_tracks.py \
  subset/debug_overfit6_stage1_asym_roi.py \
  --out-dir /data/waymo_maptracker_xm15_x45_y15_overfit6/track_visualization
```

This creates the required:

```text
/data/waymo_maptracker_xm15_x45_y15_overfit6/waymo_map_infos_train_gt_tracks.pkl
/data/waymo_maptracker_xm15_x45_y15_overfit6/waymo_map_infos_val_gt_tracks.pkl
```

## 3. Run the overfit training

```bash
python tools/train.py subset/debug_overfit6_stage1_asym_roi.py --gpus 1
```

The config writes to:

```text
/data/maptr_workspace/work_dirs/debug_overfit6_stage1_asym_roi
```

Use this experiment only to check whether the model can fit a tiny asymmetric
ROI subset. It is not a validation-quality metric.
