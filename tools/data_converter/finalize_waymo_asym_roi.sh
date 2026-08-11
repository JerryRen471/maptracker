#!/usr/bin/env bash
set -euo pipefail

CONVERTER_PID="${1:-803228}"
PROCESSED_DIR="/data/waymo_processed_xm15_x45_y15"
MAPTRACKER_DIR="/data/waymo_maptracker_xm15_x45_y15"
CONFIG="plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"

cd /root/maptracker

while kill -0 "${CONVERTER_PID}" 2>/dev/null; do
    echo "[$(date --iso-8601=seconds)] waiting for converter ${CONVERTER_PID}"
    sleep 60
done

test -s "${PROCESSED_DIR}/train_infos.pkl"
test -s "${PROCESSED_DIR}/val_infos.pkl"
test -s "${PROCESSED_DIR}/conversion_stats.json"

source /root/miniconda3/etc/profile.d/conda.sh
set +u
conda activate waymo_pack
set -u
python - <<'PY'
import json
import pickle

import numpy as np

processed_dir = "/data/waymo_processed_xm15_x45_y15"
expected_roi = (-15.0, -15.0, 45.0, 15.0)

with open(f"{processed_dir}/conversion_stats.json") as f:
    stats = json.load(f)
assert stats["num_segments"] == 1000, stats["num_segments"]
assert not stats["failures"], stats["failures"][:3]

for split in ("train", "val"):
    with open(f"{processed_dir}/{split}_infos.pkl", "rb") as f:
        data = pickle.load(f)
    assert tuple(data["metadata"]["roi_range"]) == expected_roi
    assert data["metadata"]["image_root"] == "/data/waymo_processed_v3"
    assert data["infos"], split
    for sample in data["infos"]:
        for polyline in sample["gt_polylines"]:
            points = np.asarray(polyline)
            assert np.all(points[:, 0] >= expected_roi[0] - 1e-4)
            assert np.all(points[:, 0] <= expected_roi[2] + 1e-4)
            assert np.all(points[:, 1] >= expected_roi[1] - 1e-4)
            assert np.all(points[:, 1] <= expected_roi[3] + 1e-4)
    print(split, len(data["infos"]))

print("FULL_CONVERSION_VERIFIED")
PY

rm -rf "${MAPTRACKER_DIR}"
set +u
conda activate maptracker
set -u
python pack_waymo_for_maptracker.py \
    --v3-dir "${PROCESSED_DIR}" \
    --out-dir "${MAPTRACKER_DIR}"

python tools/tracking/prepare_gt_tracks.py \
    "${CONFIG}" \
    --out-dir "${MAPTRACKER_DIR}/track_visualization"

python - <<'PY'
import os
import pickle

from mmcv import Config
from mmdet3d.datasets import build_dataset

import plugin  # noqa: F401

config_path = (
    "plugin/configs/maptracker/waymo_5cam/"
    "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"
)
cfg = Config.fromfile(config_path)

for split in ("train", "val"):
    ann_file = (
        f"/data/waymo_maptracker_xm15_x45_y15/"
        f"waymo_map_infos_{split}.pkl"
    )
    track_file = ann_file[:-4] + "_gt_tracks.pkl"
    assert os.path.isfile(ann_file)
    assert os.path.isfile(track_file)
    with open(track_file, "rb") as f:
        tracks = pickle.load(f)
    assert tracks

dataset = build_dataset(cfg.data.train)
sample = dataset[0]
assert "vectors" in sample
assert "semantic_mask" in sample
print("FINAL_DATASET_VERIFIED", len(dataset))
PY

echo "[$(date --iso-8601=seconds)] ASYMMETRIC_ROI_DATA_READY"
