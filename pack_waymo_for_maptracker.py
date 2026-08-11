"""
pack_waymo_for_maptracker.py

把 /data/waymo_processed_v3/{train,val}_infos.pkl 转为 MapTracker 格式:
    /data/waymo_maptracker/waymo_map_infos_{train,val}.pkl

关键转换:
  1. dict(infos=[...])          → dict(samples=[...], id2map={})
  2. cams[name].sensor2ego_*    → cams[name].extrinsics (4x4 ego2cam,
                                  ego → pinhole camera)
  3. label 重映射               v3 {0:divider, 1:boundary, 2:ped_crossing}
                              → MT {0:ped_crossing, 1:divider, 2:boundary}
  4. 新增 prev 字段             每段第一帧 prev=-1,其他帧 prev=上一帧 token
  5. 新增 sample_idx            全局递增,segment 内 frame_idx 顺序排列

用法:
    # Prefer the ROI-correct processed dir (e.g. xm15_x45_y15), not legacy v3:
    python pack_waymo_for_maptracker.py \
        --v3-dir /data/waymo_processed_xm15_x45_y15 \
        --out-dir /data/waymo_maptracker_xm15_x45_y15_pinhole \
        --expect-roi-range -15 -15 45 15
"""
import argparse
import os
import os.path as osp
import pickle
from collections import defaultdict

import numpy as np

# v3 label index -> category name (from metadata.class_names)
V3_LABEL_TO_NAME = {0: 'divider', 1: 'boundary', 2: 'ped_crossing'}

# MapTracker / AV2 convention (from av2 oldsplit stage3 config)
MT_CAT2ID = {'ped_crossing': 0, 'divider': 1, 'boundary': 2}

# Remap: v3 label int -> MapTracker label int
V3_TO_MT_LABEL = {v3_idx: MT_CAT2ID[name]
                  for v3_idx, name in V3_LABEL_TO_NAME.items()}
# Resulting map: {0: 1, 1: 2, 2: 0}

# Waymo camera sensor coordinates use x-forward, y-left, z-up. The pinhole
# projection used by MapTracker/BEVFormer expects x-right, y-down, z-forward.
WAYMO_CAM_TO_PINHOLE = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def cam2ego_to_ego2cam(rotation_3x3, translation_3):
    """Invert a Waymo cam->ego transform to get ego->pinhole-cam.

    sensor2ego_rotation/translation describe how to map a point expressed in
    the Waymo camera sensor frame into the ego frame. MapTracker's
    `extrinsics` field is the inverse convention, but downstream projection
    also expects pinhole camera axes where depth is the third coordinate.

    inv([[R, t], [0, 1]]) == [[R^T, -R^T @ t], [0, 1]]
    """
    R = np.asarray(rotation_3x3, dtype=np.float64)
    t = np.asarray(translation_3, dtype=np.float64).reshape(3)

    cam2ego = np.eye(4, dtype=np.float64)
    cam2ego[:3, :3] = R
    cam2ego[:3, 3] = t

    ego2waymo_cam = np.linalg.inv(cam2ego)
    return WAYMO_CAM_TO_PINHOLE @ ego2waymo_cam


def convert_cams(v3_cams, img_root):
    """Build MapTracker-style cams dict from our v3 cams dict.

    MapTracker expects each cam to have: img_fpath (str), intrinsics (3x3),
    extrinsics (4x4 ego->cam).
    """
    out = {}
    for cam_name, c in v3_cams.items():
        intr = np.asarray(c['cam_intrinsic'], dtype=np.float64)
        ext = cam2ego_to_ego2cam(c['sensor2ego_rotation'],
                                 c['sensor2ego_translation'])
        img_fpath = osp.join(img_root, c['data_path']) if img_root else c['data_path']
        out[cam_name] = dict(
            img_fpath=img_fpath,
            intrinsics=intr,
            extrinsics=ext,
            img_shape=tuple(c['img_shape']) if 'img_shape' in c else None,
        )
    return out


def validate_roi(metadata, samples, expect_roi_range, atol=1e-3):
    """Ensure packed metadata/GT match the expected ego-frame ROI.

    Raises ValueError when metadata disagrees or any GT point falls outside.
    """
    expect = tuple(float(v) for v in expect_roi_range)
    if len(expect) != 4:
        raise ValueError(f'expect_roi_range must be length 4, got {expect}')
    x_min, y_min, x_max, y_max = expect

    meta_roi = metadata.get('roi_range')
    if meta_roi is not None:
        got = tuple(float(v) for v in meta_roi)
        if got != expect:
            raise ValueError(
                f"metadata.roi_range {got} != --expect-roi-range {expect}. "
                f"Did you pack from the wrong --v3-dir?"
            )

    # Legacy packs only store half-extents; catch swapped / wrong symmetric ROI.
    if 'bev_x' in metadata and 'bev_y' in metadata:
        bev_x = float(metadata['bev_x'])
        bev_y = float(metadata['bev_y'])
        expect_bev_x = (x_max - x_min) / 2.0
        expect_bev_y = (y_max - y_min) / 2.0
        if abs(bev_x - expect_bev_x) > atol or abs(bev_y - expect_bev_y) > atol:
            raise ValueError(
                f"metadata bev_x/bev_y=({bev_x}, {bev_y}) incompatible with "
                f"expect ROI {expect} (half-extents "
                f"{expect_bev_x}, {expect_bev_y}). Wrong source ROI?"
            )

    for sample in samples:
        for poly in sample.get('gt_polylines', []):
            pts = np.asarray(poly, dtype=np.float64)
            if pts.size == 0:
                continue
            if (pts[:, 0].min() < x_min - atol or pts[:, 0].max() > x_max + atol or
                    pts[:, 1].min() < y_min - atol or pts[:, 1].max() > y_max + atol):
                raise ValueError(
                    f"GT point outside expect ROI {expect} in token="
                    f"{sample.get('token')}: "
                    f"x[{pts[:, 0].min()}, {pts[:, 0].max()}] "
                    f"y[{pts[:, 1].min()}, {pts[:, 1].max()}]"
                )


def convert_split(v3_infos, img_root, split_name):
    """Convert one split's list of frame dicts to a MapTracker samples list."""

    # Group frames by segment, sort each segment by frame_idx
    by_segment = defaultdict(list)
    for f in v3_infos:
        by_segment[f['segment_name']].append(f)
    for seg in by_segment:
        by_segment[seg].sort(key=lambda x: int(x['frame_idx']))

    # Stable segment ordering (for reproducible sample_idx)
    segments = sorted(by_segment.keys())

    samples = []
    sample_idx_global = 0
    skipped_polys = 0

    for seg_name in segments:
        frames = by_segment[seg_name]
        prev_token = -1
        for f in frames:
            polys_out = []
            labels_out = []
            for p, lbl in zip(f['gt_polylines'], f['gt_polyline_labels']):
                arr = np.asarray(p, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
                    skipped_polys += 1
                    continue
                polys_out.append(arr)
                labels_out.append(V3_TO_MT_LABEL[int(lbl)])

            sample = dict(
                # ---- MapTracker AV2 standard fields ----
                token=str(f['token']),
                log_id=seg_name,
                scene_name=seg_name,
                e2g_translation=np.asarray(f['ego2global_translation'],
                                           dtype=np.float64),
                e2g_rotation=np.asarray(f['ego2global_rotation'],
                                        dtype=np.float64),
                cams=convert_cams(f['cams'], img_root),
                lidar_fpath='',  # no lidar in our v3
                prev=prev_token,
                # ---- Bookkeeping / our additions ----
                timestamp=int(f['timestamp']),
                frame_idx=int(f['frame_idx']),
                sample_idx=sample_idx_global,
                # ---- KEY: per-frame polylines replace AV2's id2map lookup ----
                gt_polylines=polys_out,
                gt_polyline_labels=labels_out,
            )
            samples.append(sample)
            prev_token = str(f['token'])
            sample_idx_global += 1

    print(f'[{split_name}] segments={len(segments)} samples={len(samples)} '
          f'skipped_polys={skipped_polys}')
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--v3-dir',
        default='/data/waymo_processed_xm15_x45_y15',
        help='Processed infos dir (must already have the desired ROI GT).')
    parser.add_argument('--out-dir', default='/data/waymo_maptracker')
    parser.add_argument(
        '--img-root',
        default=None,
        help='Joined to cams[*].data_path. Defaults to metadata.image_root '
             'from each input pkl.')
    parser.add_argument('--info-prefix', default='waymo')
    parser.add_argument(
        '--expect-roi-range',
        nargs=4,
        type=float,
        default=None,
        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
        help='If set, fail when source metadata/GT do not match this ROI. '
             'Use this to avoid packing from a wrong processed dir.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ['train', 'val']:
        in_path = osp.join(args.v3_dir, f'{split}_infos.pkl')
        out_path = osp.join(args.out_dir,
                            f'{args.info_prefix}_map_infos_{split}.pkl')
        print(f'\n=== {split} ===')
        print(f'reading {in_path}')
        with open(in_path, 'rb') as f:
            v3 = pickle.load(f)

        metadata = dict(v3.get('metadata', {}))
        img_root = args.img_root
        if img_root is None:
            img_root = metadata.get('image_root', args.v3_dir)
        print(f'img_root     : {img_root}')
        print(f'src metadata : {metadata}')
        samples = convert_split(v3['infos'], img_root, split)

        if args.expect_roi_range is not None:
            print(f'validating ROI {tuple(args.expect_roi_range)} ...')
            validate_roi(metadata, samples, args.expect_roi_range)
            # Persist explicit roi_range even if source only had bev half-extents.
            metadata['roi_range'] = tuple(float(v) for v in args.expect_roi_range)

        # id2map kept as empty dict — WaymoDataset overrides get_sample()
        # to bypass map_extractor and read polylines directly from each sample.
        out = dict(samples=samples, id2map={}, metadata=metadata)
        print(f'writing {out_path}')
        with open(out_path, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = osp.getsize(out_path) / 1024 / 1024
        print(f'OK: {out_path} ({mb:.1f} MB)')

    # Quick sanity printout
    print('\n=== sanity (first sample of train) ===')
    with open(osp.join(args.out_dir,
                       f'{args.info_prefix}_map_infos_train.pkl'), 'rb') as f:
        chk = pickle.load(f)
    s0 = chk['samples'][0]
    print(f'  token        : {s0["token"]}')
    print(f'  log_id       : {s0["log_id"]}')
    print(f'  prev         : {s0["prev"]}')
    print(f'  sample_idx   : {s0["sample_idx"]}')
    print(f'  e2g_t shape  : {s0["e2g_translation"].shape}')
    print(f'  e2g_R shape  : {s0["e2g_rotation"].shape}')
    print(f'  cams         : {list(s0["cams"].keys())}')
    print(f'  cam FRONT extrinsics shape: '
          f'{s0["cams"]["FRONT"]["extrinsics"].shape}')
    print(f'  polylines    : {len(s0["gt_polylines"])} polys')
    print(f'  labels       : {s0["gt_polyline_labels"]}')

    print('\nDone.')


if __name__ == '__main__':
    main()
