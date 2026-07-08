#!/usr/bin/env bash
set -euo pipefail

WAYMO_DATA_DIR="${WAYMO_DATA_DIR:-/data/waymo}"
PROCESSED_DIR="${PROCESSED_DIR:-/data/waymo_processed_x0_x60_y15}"
MAPTRACKER_DIR="${MAPTRACKER_DIR:-/data/waymo_maptracker_x0_x60_y15}"
OVERFIT_DIR="${OVERFIT_DIR:-/data/waymo_maptracker_x0_x60_y15_overfit6}"
CONFIG="${CONFIG:-subset/debug_overfit6_stage1_front_roi.py}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LIMIT="${LIMIT:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-5}"
NUM_POINTS="${NUM_POINTS:-20}"
REUSE_IMAGES_FROM="${REUSE_IMAGES_FROM:-}"

export PROCESSED_DIR MAPTRACKER_DIR OVERFIT_DIR

cd "$(dirname "$0")/.."

CONVERT_ARGS=(
    --data-dir "${WAYMO_DATA_DIR}"
    --out-dir "${PROCESSED_DIR}"
    --x-min 0
    --y-min -15
    --x-max 60
    --y-max 15
    --num-workers "${NUM_WORKERS}"
    --frame-stride "${FRAME_STRIDE}"
    --num-points "${NUM_POINTS}"
)

if [[ "${LIMIT}" != "0" ]]; then
    CONVERT_ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${REUSE_IMAGES_FROM}" ]]; then
    CONVERT_ARGS+=(--reuse-images-from "${REUSE_IMAGES_FROM}")
fi

python tools/data_converter/waymo_map_converter.py "${CONVERT_ARGS[@]}"

python - <<'PY'
import os
import pickle
from pathlib import Path

import numpy as np

processed_dir = Path(os.environ["PROCESSED_DIR"])
expected_roi = (0.0, -15.0, 60.0, 15.0)

for split in ("train", "val"):
    with open(processed_dir / f"{split}_infos.pkl", "rb") as f:
        data = pickle.load(f)
    assert tuple(data["metadata"]["roi_range"]) == expected_roi
    if split == "train":
        assert data["infos"], split
    for sample in data["infos"]:
        for polyline in sample["gt_polylines"]:
            points = np.asarray(polyline)
            assert np.all(points[:, 0] >= expected_roi[0] - 1e-4)
            assert np.all(points[:, 0] <= expected_roi[2] + 1e-4)
            assert np.all(points[:, 1] >= expected_roi[1] - 1e-4)
            assert np.all(points[:, 1] <= expected_roi[3] + 1e-4)
    print(split, len(data["infos"]))

print("FRONT_ROI_CONVERSION_VERIFIED")
PY

rm -rf "${MAPTRACKER_DIR}" "${OVERFIT_DIR}"

python pack_waymo_for_maptracker.py \
    --v3-dir "${PROCESSED_DIR}" \
    --out-dir "${MAPTRACKER_DIR}"

python subset/make_waymo_overfit6_subset.py \
    --source-dir "${MAPTRACKER_DIR}" \
    --output-dir "${OVERFIT_DIR}" \
    --num-scenes 6

python tools/tracking/prepare_gt_tracks.py \
    "${CONFIG}" \
    --out-dir "${OVERFIT_DIR}/track_visualization"

python - <<'PY'
import os
import pickle

root = os.environ["OVERFIT_DIR"]
for split in ("train", "val"):
    ann_file = f"{root}/waymo_map_infos_{split}.pkl"
    track_file = ann_file[:-4] + "_gt_tracks.pkl"
    assert os.path.isfile(ann_file), ann_file
    assert os.path.isfile(track_file), track_file
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    assert tuple(data["metadata"]["roi_range"]) == (0, -15, 60, 15)
    assert data["samples"], split
    with open(track_file, "rb") as f:
        tracks = pickle.load(f)
    assert tracks, split
    print(split, len(data["samples"]), len(tracks))

print("FRONT_ROI_OVERFIT6_DATA_READY")
PY
