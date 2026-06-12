#!/usr/bin/env python3
"""
waymo_converter.py - Waymo Perception v1 tfrecord → MapTR-compatible pkl

输出结构:
  out_dir/
    train_infos.pkl           # list of info dict, train split
    val_infos.pkl             # list of info dict, val split
    images/<segment>/<frame_idx:04d>_<cam>.jpg
    conversion_stats.json     # 完整性统计

每个 info dict 字段:
  token, segment_name, frame_idx, timestamp,
  cams[5路]: {data_path, cam_intrinsic(3,3), sensor2ego_rotation(3,3),
             sensor2ego_translation(3,), img_shape(H,W)},
  ego2global_translation(3,), ego2global_rotation(3,3),
  gt_polylines: list of (N_resample, 2) in vehicle frame,
  gt_polyline_labels: list of int (0=divider, 1=boundary, 2=ped_crossing).

用法:
  # 测试转 5 个 segment,看流程是否跑通
  python waymo_converter.py --limit 5 --num-workers 1

  # 全量转换
  python waymo_converter.py --data-dir /data/waymo --out-dir /data/waymo_processed \\
      --num-workers 8

  # 跑完用 verify_pkl.py 校验(下一步给)
"""

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict, Counter

import numpy as np
from shapely.geometry import LineString, Polygon, box
from tqdm import tqdm


# ============================================================
# 常量 / 配置
# ============================================================

CAM_ENUM_TO_NAME = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT",
                    4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

# Waymo map_features 的 oneof type → MapTR 类别 + label
# 注意:lane / driveway / speed_bump / stop_sign 在 baseline 中**忽略**
CLASS_MAP = {
    "road_line": ("divider",      0),
    "road_edge": ("boundary",     1),
    "crosswalk": ("ped_crossing", 2),
}

# polyline 短于这个长度(米)直接丢
MIN_POLYLINE_LENGTH_M = 1.0

# 每个 GT polyline 重采样到的点数(MapTR 标准 20)
DEFAULT_NUM_POINTS = 20

# 抽帧:Waymo 10Hz,每 N 帧取一帧
DEFAULT_FRAME_STRIDE = 5


# ============================================================
# 坐标变换工具
# ============================================================

def mat4(transform_field):
    """proto 中 16 个 float (row-major) → 4x4 numpy。"""
    return np.array(transform_field, dtype=np.float64).reshape(4, 4)


def transform_points_3d(pts_xyz, T):
    """(N,3) 点用 4x4 矩阵 T 变换;返回 (N,3)。"""
    n = pts_xyz.shape[0]
    homo = np.concatenate([pts_xyz, np.ones((n, 1))], axis=1)  # (N,4)
    out = homo @ T.T
    return out[:, :3]


# ============================================================
# Polyline / Polygon 处理
# ============================================================

def extract_map_polylines_world(map_features):
    """
    从 frame.map_features(只在 frame 0 里)提取所需的 3 类多边形/线。
    返回 list of dict:
      {label, points_world (N,3), is_polygon}
    """
    out = []
    for mf in map_features:
        kind = mf.WhichOneof("feature_data")
        if kind not in CLASS_MAP:
            continue
        _, label = CLASS_MAP[kind]

        if kind == "road_line":
            pts = mf.road_line.polyline
        elif kind == "road_edge":
            pts = mf.road_edge.polyline
        elif kind == "crosswalk":
            pts = mf.crosswalk.polygon
        else:
            continue

        if len(pts) < 2:
            continue

        arr = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float64)
        out.append(dict(
            label=label,
            points_world=arr,
            is_polygon=(kind == "crosswalk"),
        ))
    return out


def crop_polyline_to_roi(pts_2d, roi_range,
                         min_length=MIN_POLYLINE_LENGTH_M):
    """Clip a polyline to (x_min, y_min, x_max, y_max)."""
    if len(pts_2d) < 2:
        return []
    bev_poly = box(*roi_range)
    try:
        line = LineString(pts_2d)
        if not line.is_valid or line.is_empty:
            return []
        clipped = line.intersection(bev_poly)
    except Exception:
        return []

    if clipped.is_empty:
        return []

    result = []
    if clipped.geom_type == "LineString":
        if clipped.length >= min_length:
            result.append(np.array(clipped.coords))
    elif clipped.geom_type == "MultiLineString":
        for sub in clipped.geoms:
            if sub.length >= min_length:
                result.append(np.array(sub.coords))
    return result


def crop_polyline_to_bev(pts_2d, bev_x, bev_y,
                         min_length=MIN_POLYLINE_LENGTH_M):
    """
    用 shapely 把一条 polyline 裁到 BEV 框 [-bev_x, bev_x] × [-bev_y, bev_y]。
    polyline 可能进出 BEV 多次,返回多段子 polyline。

    pts_2d: (N, 2) 自车系 (x=前, y=左)
    返回: list of (M, 2) numpy
    """
    return crop_polyline_to_roi(
        pts_2d, (-bev_x, -bev_y, bev_x, bev_y), min_length)


def crop_polygon_to_roi(poly_pts_2d, roi_range,
                        min_length=MIN_POLYLINE_LENGTH_M):
    """
    裁剪 polygon(crosswalk)到 BEV 框,返回外轮廓 polyline 列表。
    poly_pts_2d: (N, 2) 自车系,polygon 顶点
    """
    if len(poly_pts_2d) < 3:
        return []
    bev_poly = box(*roi_range)
    try:
        poly = Polygon(poly_pts_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return []
        clipped = poly.intersection(bev_poly)
    except Exception:
        return []

    if clipped.is_empty:
        return []

    result = []
    geoms = [clipped] if clipped.geom_type == "Polygon" else list(getattr(clipped, "geoms", []))
    for g in geoms:
        if g.geom_type != "Polygon":
            continue
        if g.exterior is None:
            continue
        coords = np.array(g.exterior.coords)  # 闭合,首尾相同
        if g.exterior.length >= min_length and len(coords) >= 3:
            result.append(coords)
    return result


def crop_polygon_to_bev(poly_pts_2d, bev_x, bev_y,
                        min_length=MIN_POLYLINE_LENGTH_M):
    return crop_polygon_to_roi(
        poly_pts_2d, (-bev_x, -bev_y, bev_x, bev_y), min_length)


def resample_polyline(pts_2d, n_points=DEFAULT_NUM_POINTS):
    """
    按弧长均匀重采样到 n_points 点。
    pts_2d: (N, 2), N >= 2
    返回 (n_points, 2) 或 None(退化)
    """
    if len(pts_2d) < 2:
        return None
    diffs = np.diff(pts_2d, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = seg_lens.sum()
    if total_len < 1e-6:
        return None

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    targets = np.linspace(0.0, total_len, n_points)

    out = np.zeros((n_points, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        if t >= total_len:
            out[i] = pts_2d[-1]
            continue
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = max(0, min(idx, len(seg_lens) - 1))
        seg_t = (t - cum[idx]) / seg_lens[idx] if seg_lens[idx] > 1e-9 else 0.0
        out[i] = pts_2d[idx] * (1.0 - seg_t) + pts_2d[idx + 1] * seg_t
    return out


def build_gt_for_frame(polylines_world, vehicle_T_world, roi_range,
                       n_resample):
    """
    输入世界系下的所有 polyline + 当前帧的 vehicle←world 变换,
    输出该帧的 GT(自车系、已裁剪、已重采样)。

    返回 (list of (n_resample, 2) arrays, list of int labels)
    """
    gt_lines = []
    gt_labels = []

    for pl in polylines_world:
        pts_v = transform_points_3d(pl["points_world"], vehicle_T_world)
        pts_2d = pts_v[:, :2]  # 忽略 z

        if pl["is_polygon"]:
            sub_lines = crop_polygon_to_roi(pts_2d, roi_range)
        else:
            sub_lines = crop_polyline_to_roi(pts_2d, roi_range)

        for sub in sub_lines:
            resampled = resample_polyline(sub, n_resample)
            if resampled is None:
                continue
            gt_lines.append(resampled.astype(np.float32))
            gt_labels.append(pl["label"])

    return gt_lines, gt_labels


# ============================================================
# 相机 / 标定 / 图像保存
# ============================================================

def extract_calibration(cam_calib):
    """
    Waymo CameraCalibration → MapTR 风格字段
    intrinsic = [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    extrinsic = 4x4: camera(sensor) → vehicle
    """
    intr = cam_calib.intrinsic
    K = np.array([
        [intr[0], 0.0,      intr[2]],
        [0.0,     intr[1],  intr[3]],
        [0.0,     0.0,      1.0    ],
    ], dtype=np.float32)
    T_cam_to_vehicle = mat4(cam_calib.extrinsic.transform).astype(np.float32)
    R = T_cam_to_vehicle[:3, :3]      # sensor2ego rotation
    t = T_cam_to_vehicle[:3, 3]       # sensor2ego translation
    return K, R, t, int(cam_calib.width), int(cam_calib.height)


def save_camera_jpegs(frame, segment_img_dir, frame_idx):
    """
    Waymo 的相机图像本身就是 JPEG bytes,直接写盘不做重编码,极快。
    返回 dict: {cam_name: relative_path_str}
    """
    out = {}
    for img in frame.images:
        cam_name = CAM_ENUM_TO_NAME.get(img.name)
        if cam_name is None:
            continue
        fname = f"{frame_idx:04d}_{cam_name}.jpg"
        fpath = os.path.join(segment_img_dir, fname)
        with open(fpath, "wb") as f:
            f.write(img.image)
        out[cam_name] = fname  # 相对路径,后面 info 里拼绝对/相对都行
    return out


# ============================================================
# 单帧 → info dict
# ============================================================

def frame_to_info(frame_target, polylines_world, segment_name, frame_idx,
                  segment_img_dir, roi_range, n_resample, save_images,
                  reference_images):
    """构造 info dict。"""
    # GT
    vehicle_pose = mat4(frame_target.pose.transform)   # vehicle → world
    vehicle_T_world = np.linalg.inv(vehicle_pose)
    gt_lines, gt_labels = build_gt_for_frame(
        polylines_world, vehicle_T_world, roi_range, n_resample)

    # 相机:标定 + 图像
    cam_info = {}
    calibs = {c.name: c for c in frame_target.context.camera_calibrations}
    saved_paths = save_camera_jpegs(frame_target, segment_img_dir, frame_idx) \
                  if save_images else {}
    if reference_images:
        saved_paths = {
            cam_name: f"{frame_idx:04d}_{cam_name}.jpg"
            for cam_name in CAM_ENUM_TO_NAME.values()
        }

    for cam_enum, cam_name in CAM_ENUM_TO_NAME.items():
        if cam_enum not in calibs:
            continue
        K, R, t, W, H = extract_calibration(calibs[cam_enum])
        rel_path = os.path.join(
            "images", segment_name, saved_paths.get(cam_name, "")
        ) if (save_images or reference_images) else ""
        cam_info[cam_name] = dict(
            data_path=rel_path,
            cam_intrinsic=K,
            sensor2ego_rotation=R,
            sensor2ego_translation=t,
            img_shape=(H, W),
        )

    info = dict(
        token=f"{segment_name}_frame_{frame_idx:04d}",
        segment_name=segment_name,
        frame_idx=frame_idx,
        timestamp=int(frame_target.timestamp_micros),
        cams=cam_info,
        ego2global_translation=vehicle_pose[:3, 3].astype(np.float32),
        ego2global_rotation=vehicle_pose[:3, :3].astype(np.float32),
        gt_polylines=gt_lines,
        gt_polyline_labels=gt_labels,
    )
    return info


# ============================================================
# 单 segment → list of infos
# ============================================================

def process_segment(args_tuple):
    """
    worker 函数,处理一个 segment。
    args_tuple: (segment_path, out_dir, roi_range, n_resample,
                 frame_stride, save_images, reference_images)
    返回 (segment_name, infos, stats_dict, error_str_or_None)
    """
    (segment_path, out_dir, roi_range, n_resample,
     frame_stride, save_images, reference_images) = args_tuple

    # 子进程内才 import TF / proto,避免 fork 后的 TF 状态问题
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    segment_name = os.path.basename(segment_path).replace(".tfrecord", "")
    segment_img_dir = os.path.join(out_dir, "images", segment_name)
    if save_images:
        os.makedirs(segment_img_dir, exist_ok=True)

    try:
        ds = tf.data.TFRecordDataset(segment_path, compression_type="")
        polylines_world = None
        infos = []
        gt_count_per_frame = []
        label_counter = Counter()
        zero_gt_frames = 0

        for i, data in enumerate(ds):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # 地图只在 frame 0
            if i == 0:
                polylines_world = extract_map_polylines_world(frame.map_features)

            # 抽帧
            if i % frame_stride != 0:
                continue

            info = frame_to_info(
                frame, polylines_world or [], segment_name, i,
                segment_img_dir, roi_range, n_resample, save_images,
                reference_images
            )
            infos.append(info)
            gt_count_per_frame.append(len(info["gt_polylines"]))
            for lab in info["gt_polyline_labels"]:
                label_counter[lab] += 1
            if len(info["gt_polylines"]) == 0:
                zero_gt_frames += 1

        stats = dict(
            segment_name=segment_name,
            num_frames_processed=len(infos),
            num_map_polylines_world=len(polylines_world or []),
            mean_gt_per_frame=(float(np.mean(gt_count_per_frame))
                               if gt_count_per_frame else 0.0),
            zero_gt_frames=zero_gt_frames,
            label_distribution={int(k): int(v) for k, v in label_counter.items()},
        )
        return segment_name, infos, stats, None

    except Exception:
        return segment_name, [], {}, traceback.format_exc()


# ============================================================
# Train/Val 划分(稳定哈希,按 segment)
# ============================================================

def assign_split(segment_name, val_ratio, seed=42):
    """
    用稳定哈希决定 segment 进 train 还是 val,保证不同次运行结果一致。
    """
    h = hashlib.md5(f"{seed}_{segment_name}".encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF   # [0, 1)
    return "val" if bucket < val_ratio else "train"


# ============================================================
# Main / CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/waymo",
                   help="训练 segments 目录。若给 --val-data-dir 则此目录全进 train 集")
    p.add_argument("--val-data-dir", default=None,
                   help="可选:此目录所有 segment 进 val 集,--data-dir 全进 train 集(忽略 --val-ratio)")
    p.add_argument("--out-dir",  default="/data/waymo_processed")
    p.add_argument("--bev-x", type=float, default=60.0,
                   help="BEV 前后范围 ±x 米(默认 60)")
    p.add_argument("--bev-y", type=float, default=30.0,
                   help="BEV 左右范围 ±y 米(默认 30)")
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--y-min", type=float, default=None)
    p.add_argument("--x-max", type=float, default=None)
    p.add_argument("--y-max", type=float, default=None)
    p.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS,
                   help="GT polyline 重采样点数(默认 20)")
    p.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE,
                   help="抽帧间隔,10Hz/stride = 输出帧率(默认 5 → 2Hz)")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="验证集 segment 比例(默认 0.2)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="并行进程数(默认 8)")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 时只处理前 N 个 segment,用于测试")
    p.add_argument("--no-images", action="store_true",
                   help="不写出图像,只产 pkl(用于快速跑 GT 流水线测试)")
    p.add_argument(
        "--reuse-images-from",
        default=None,
        help="复用已有 images/ 目录，只重算 GT 和 pkl")
    args = p.parse_args()

    explicit_bounds = [args.x_min, args.y_min, args.x_max, args.y_max]
    if any(v is not None for v in explicit_bounds):
        if not all(v is not None for v in explicit_bounds):
            p.error("--x-min/--y-min/--x-max/--y-max must be used together")
        roi_range = tuple(explicit_bounds)
    else:
        roi_range = (-args.bev_x, -args.bev_y, args.bev_x, args.bev_y)
    if roi_range[0] >= roi_range[2] or roi_range[1] >= roi_range[3]:
        p.error(f"invalid ROI range: {roi_range}")

    train_only_segs = sorted(glob.glob(os.path.join(args.data_dir, "segment-*.tfrecord")))
    val_only_segs = []
    if args.val_data_dir:
        val_only_segs = sorted(glob.glob(os.path.join(args.val_data_dir, "segment-*.tfrecord")))
        if not val_only_segs:
            print(f"❌ --val-data-dir 无 segment-*.tfrecord: {args.val_data_dir}")
            return 1
        print(f"✅ 显式 split: train={len(train_only_segs)}, val={len(val_only_segs)}")
    segments = train_only_segs + val_only_segs
    if not segments:
        print(f"❌ 数据目录无 segment-*.tfrecord: {args.data_dir}")
        return 1
    # 显式 val 集的 segment 名(用于覆盖 hash split)
    val_seg_names = set(
        os.path.basename(s)[:-len(".tfrecord")] for s in val_only_segs
    )
    if args.limit > 0:
        segments = segments[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    save_images = not args.no_images and args.reuse_images_from is None
    reference_images = args.reuse_images_from is not None
    if reference_images:
        image_dir = os.path.join(args.reuse_images_from, "images")
        if not os.path.isdir(image_dir):
            p.error(f"missing reused image directory: {image_dir}")

    print(f"找到 {len(segments)} 个 segment 待处理")
    print(f"输出目录: {args.out_dir}")
    print(f"BEV 范围: x=[{roi_range[0]}, {roi_range[2]}], "
          f"y=[{roi_range[1]}, {roi_range[3]}], "
          f"重采样: {args.num_points} 点, 抽帧 1/{args.frame_stride}")
    print(f"并行 workers: {args.num_workers}, 写图: {save_images}, "
          f"复用图像: {args.reuse_images_from}\n")

    task_args = [
        (seg, args.out_dir, roi_range, args.num_points,
         args.frame_stride, save_images, reference_images)
        for seg in segments
    ]

    train_infos, val_infos = [], []
    all_stats = []
    t_start = time.time()

    if args.num_workers <= 1:
        results = (process_segment(t) for t in task_args)
        results_iter = tqdm(results, total=len(task_args), desc="转换")
    else:
        # spawn 比 fork 与 TF 兼容性更好
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=args.num_workers)
        results_iter = tqdm(
            pool.imap_unordered(process_segment, task_args),
            total=len(task_args), desc="转换"
        )

    failures = []
    for seg_name, infos, stats, err in results_iter:
        if err is not None:
            failures.append((seg_name, err))
            continue
        all_stats.append(stats)
        split = ("val" if (args.val_data_dir and seg_name in val_seg_names)
                  else assign_split(seg_name, args.val_ratio))
        if split == "train":
            train_infos.extend(infos)
        else:
            val_infos.extend(infos)

    if args.num_workers > 1:
        pool.close(); pool.join()

    # ----- 落盘 -----
    train_pkl = os.path.join(args.out_dir, "train_infos.pkl")
    val_pkl = os.path.join(args.out_dir, "val_infos.pkl")
    with open(train_pkl, "wb") as f:
        pickle.dump({"infos": train_infos, "metadata": {
            "roi_range": roi_range,
            "bev_x": (roi_range[2] - roi_range[0]) / 2,
            "bev_y": (roi_range[3] - roi_range[1]) / 2,
            "num_points": args.num_points, "frame_stride": args.frame_stride,
            "class_names": ["divider", "boundary", "ped_crossing"],
            "image_root": args.reuse_images_from or args.out_dir,
        }}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_pkl, "wb") as f:
        pickle.dump({"infos": val_infos, "metadata": {
            "roi_range": roi_range,
            "bev_x": (roi_range[2] - roi_range[0]) / 2,
            "bev_y": (roi_range[3] - roi_range[1]) / 2,
            "num_points": args.num_points, "frame_stride": args.frame_stride,
            "class_names": ["divider", "boundary", "ped_crossing"],
            "image_root": args.reuse_images_from or args.out_dir,
        }}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ----- 统计 -----
    elapsed = time.time() - t_start
    label_total = Counter()
    for s in all_stats:
        for k, v in s["label_distribution"].items():
            label_total[int(k)] += int(v)

    total_frames = sum(s["num_frames_processed"] for s in all_stats)
    total_gt = sum(s["num_frames_processed"] * s["mean_gt_per_frame"]
                   for s in all_stats)
    mean_gt = total_gt / max(total_frames, 1)
    zero_total = sum(s["zero_gt_frames"] for s in all_stats)

    stats_json = dict(
        elapsed_sec=round(elapsed, 1),
        num_segments=len(all_stats),
        num_train_frames=len(train_infos),
        num_val_frames=len(val_infos),
        mean_gt_per_frame=round(mean_gt, 2),
        zero_gt_frames=zero_total,
        label_total={
            "divider(0)":      label_total[0],
            "boundary(1)":     label_total[1],
            "ped_crossing(2)": label_total[2],
        },
        failures=[{"segment": s, "trace": e[:500]} for s, e in failures],
        per_segment=all_stats,
    )
    with open(os.path.join(args.out_dir, "conversion_stats.json"), "w") as f:
        json.dump(stats_json, f, indent=2)

    # ----- 终端摘要 -----
    print()
    print("=" * 60)
    print(f"耗时              : {elapsed:.1f} 秒")
    print(f"成功 segment      : {len(all_stats)} / {len(segments)}")
    print(f"  失败            : {len(failures)}")
    print(f"训练帧 / 验证帧   : {len(train_infos)} / {len(val_infos)}")
    print(f"平均 GT / 帧      : {mean_gt:.2f}")
    print(f"GT 为 0 的帧      : {zero_total}")
    print(f"类别分布:")
    print(f"  divider      (0): {label_total[0]}")
    print(f"  boundary     (1): {label_total[1]}")
    print(f"  ped_crossing (2): {label_total[2]}")
    if failures:
        print(f"\n⚠️ 失败 segment(前 3):")
        for s, e in failures[:3]:
            print(f"  - {s}")
            print(f"    {e.splitlines()[-1] if e else ''}")
    print(f"\n输出:")
    print(f"  {train_pkl}")
    print(f"  {val_pkl}")
    print(f"  {os.path.join(args.out_dir, 'conversion_stats.json')}")
    print("=" * 60)
    print("➡️  下一步: 用 verify_pkl.py 回读一帧的 GT 画 BEV,跟 Phase 2 对比")
    return 0


if __name__ == "__main__":
    sys.exit(main())
