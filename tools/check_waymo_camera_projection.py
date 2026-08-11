#!/usr/bin/env python3
"""Check Waymo MapTracker camera projection statistics from packed pkl data."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path


DEFAULT_ROI_RANGE = (-15.0, -15.0, 45.0, 15.0)
DEFAULT_PROBE_X = (5.0, 10.0, 20.0, 30.0, 40.0)


def as_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def matmul(a, b):
    a = as_list(a)
    b = as_list(b)
    out = []
    for row in a:
        out_row = []
        for col_idx in range(len(b[0])):
            out_row.append(sum(float(row[k]) * float(b[k][col_idx])
                               for k in range(len(b))))
        out.append(out_row)
    return out


def matvec(a, vec):
    a = as_list(a)
    vec = as_list(vec)
    return [
        sum(float(row[k]) * float(vec[k]) for k in range(len(vec)))
        for row in a
    ]


def viewpad_from_intrinsic(intrinsic):
    intrinsic = as_list(intrinsic)
    viewpad = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for row in range(min(3, len(intrinsic))):
        for col in range(min(3, len(intrinsic[row]))):
            viewpad[row][col] = float(intrinsic[row][col])
    return viewpad


def linspace(start, stop, num):
    if num <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(num - 1)
    return [float(start) + step * idx for idx in range(num)]


def make_bev_grid(roi_range=DEFAULT_ROI_RANGE, bev_h=50, bev_w=100, z=0.0):
    xmin, ymin, xmax, ymax = [float(v) for v in roi_range]
    xs = linspace(xmin, xmax, int(bev_w))
    ys = linspace(ymax, ymin, int(bev_h))
    return [[x, y, float(z), 1.0] for y in ys for x in xs]


def make_probe_points(probe_x=DEFAULT_PROBE_X, probe_y=0.0, z=0.0):
    return [[float(x), float(probe_y), float(z), 1.0] for x in probe_x]


def project_points(points, intrinsic, extrinsic):
    ego2img = matmul(viewpad_from_intrinsic(intrinsic), extrinsic)
    depth, u_values, v_values = [], [], []
    for point in points:
        proj = matvec(ego2img, point)
        d = float(proj[2])
        depth.append(d)
        if d > 0.0:
            u_values.append(float(proj[0]) / d)
            v_values.append(float(proj[1]) / d)
        else:
            u_values.append(float("nan"))
            v_values.append(float("nan"))
    return {"depth": depth, "u": u_values, "v": v_values}


def finite_min(values):
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return min(finite) if finite else None


def finite_max(values):
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return max(finite) if finite else None


def get_img_shape(cam_info, fallback_img_shape=None):
    shape = cam_info.get("img_shape")
    if shape is None:
        shape = fallback_img_shape
    if shape is None:
        return None
    shape = as_list(shape)
    return int(shape[0]), int(shape[1])


def count_in_image(projected, img_shape, eps=1e-5):
    if img_shape is None:
        return None
    img_h, img_w = img_shape
    count = 0
    for depth, u, v in zip(projected["depth"], projected["u"], projected["v"]):
        if (
            depth > eps
            and math.isfinite(u)
            and math.isfinite(v)
            and 0.0 < u < float(img_w)
            and 0.0 < v < float(img_h)
        ):
            count += 1
    return count


def projection_summary(projected, img_shape=None, eps=1e-5):
    total = len(projected["depth"])
    depth_positive = sum(1 for depth in projected["depth"] if depth > eps)
    in_image = count_in_image(projected, img_shape, eps=eps)
    return {
        "total": total,
        "depth_positive": depth_positive,
        "depth_positive_ratio": depth_positive / total if total else 0.0,
        "in_image": in_image,
        "in_image_ratio": (
            in_image / total if total and in_image is not None else None),
        "depth_min": finite_min(projected["depth"]),
        "depth_max": finite_max(projected["depth"]),
        "u_min": finite_min(projected["u"]),
        "u_max": finite_max(projected["u"]),
        "v_min": finite_min(projected["v"]),
        "v_max": finite_max(projected["v"]),
    }


def camera_projection_stats(sample, roi_range=DEFAULT_ROI_RANGE, bev_h=50,
                            bev_w=100, probe_x=DEFAULT_PROBE_X,
                            probe_y=0.0, cameras=None,
                            fallback_img_shape=None):
    grid_points = make_bev_grid(roi_range=roi_range, bev_h=bev_h, bev_w=bev_w)
    probe_points = make_probe_points(probe_x=probe_x, probe_y=probe_y)
    selected = set(cameras or [])
    stats = {}

    for cam_name, cam_info in sample.get("cams", {}).items():
        if selected and cam_name not in selected:
            continue

        intrinsic = cam_info["intrinsics"]
        extrinsic = cam_info["extrinsics"]
        img_shape = get_img_shape(cam_info, fallback_img_shape)

        grid_projected = project_points(grid_points, intrinsic, extrinsic)
        probe_projected = project_points(probe_points, intrinsic, extrinsic)
        grid_summary = projection_summary(grid_projected, img_shape)
        probe_summary = projection_summary(probe_projected, img_shape)

        stats[cam_name] = {
            "img_shape": img_shape,
            "grid_total": grid_summary["total"],
            "grid_depth_positive": grid_summary["depth_positive"],
            "grid_depth_positive_ratio": grid_summary["depth_positive_ratio"],
            "grid_in_image": grid_summary["in_image"],
            "grid_in_image_ratio": grid_summary["in_image_ratio"],
            "grid_depth_min": grid_summary["depth_min"],
            "grid_depth_max": grid_summary["depth_max"],
            "grid_u_min": grid_summary["u_min"],
            "grid_u_max": grid_summary["u_max"],
            "grid_v_min": grid_summary["v_min"],
            "grid_v_max": grid_summary["v_max"],
            "probe_total": probe_summary["total"],
            "probe_depth_positive": probe_summary["depth_positive"],
            "probe_in_image": probe_summary["in_image"],
            "probe_depth": probe_projected["depth"],
            "probe_u": probe_projected["u"],
            "probe_v": probe_projected["v"],
        }
    return stats


def load_samples(ann_file):
    with Path(ann_file).open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        "annotation file must be a list of samples or a dict containing "
        f"'samples', got {type(payload).__name__}")


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_stats_line(cam_name, stats):
    return (
        f"  {cam_name}: "
        f"grid depth>0 {stats['grid_depth_positive']}/{stats['grid_total']} "
        f"({stats['grid_depth_positive_ratio']:.4f}), "
        f"in_image {fmt(stats['grid_in_image'])}/{stats['grid_total']} "
        f"({fmt(stats['grid_in_image_ratio'])}), "
        f"depth_range=[{fmt(stats['grid_depth_min'])}, "
        f"{fmt(stats['grid_depth_max'])}], "
        f"u_range=[{fmt(stats['grid_u_min'])}, {fmt(stats['grid_u_max'])}], "
        f"v_range=[{fmt(stats['grid_v_min'])}, {fmt(stats['grid_v_max'])}], "
        f"probe_depth={stats['probe_depth']}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check camera projection validity in packed Waymo MapTracker pkl files.")
    parser.add_argument("--ann-file", required=True,
                        help="Packed waymo_map_infos_{train,val}.pkl")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Maximum samples to inspect")
    parser.add_argument("--roi-range", nargs=4, type=float,
                        default=DEFAULT_ROI_RANGE,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="BEV ROI range in ego coordinates")
    parser.add_argument("--bev-h", type=int, default=50,
                        help="BEV grid height used for projection sampling")
    parser.add_argument("--bev-w", type=int, default=100,
                        help="BEV grid width used for projection sampling")
    parser.add_argument("--probe-x", nargs="+", type=float,
                        default=list(DEFAULT_PROBE_X),
                        help="Forward ego x positions for probe points")
    parser.add_argument("--probe-y", type=float, default=0.0,
                        help="Lateral ego y for probe points")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Optional camera names to inspect, e.g. FRONT FRONT_LEFT")
    parser.add_argument("--fallback-img-shape", nargs=2, type=int, default=None,
                        metavar=("H", "W"),
                        help="Use this image shape if cam img_shape is absent")
    parser.add_argument("--fail-on-empty-depth", type=int, default=1,
                        help="Exit with code 2 if all inspected grid depths are non-positive")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_samples(args.ann_file)
    max_frames = min(len(samples), args.max_frames)
    total_grid = 0
    total_depth_positive = 0

    print("ann_file:", args.ann_file)
    print("samples:", len(samples))
    print("checked_frames:", max_frames)
    print("roi_range:", tuple(args.roi_range))
    print("bev_size:", (args.bev_h, args.bev_w))
    if args.fallback_img_shape is None:
        print("fallback_img_shape: none; in_image is n/a when pkl lacks cam img_shape")
    else:
        print("fallback_img_shape:", tuple(args.fallback_img_shape))

    for idx, sample in enumerate(samples[:max_frames]):
        token = sample.get("token", "")
        scene = sample.get("scene_name", sample.get("log_id", ""))
        frame_idx = sample.get("frame_idx", idx)
        print(f"\n[{idx}] scene={scene} frame={frame_idx} token={token}")
        stats = camera_projection_stats(
            sample,
            roi_range=args.roi_range,
            bev_h=args.bev_h,
            bev_w=args.bev_w,
            probe_x=args.probe_x,
            probe_y=args.probe_y,
            cameras=args.cameras,
            fallback_img_shape=args.fallback_img_shape,
        )
        if not stats:
            print("  no cameras matched")
        for cam_name, cam_stats in stats.items():
            print(format_stats_line(cam_name, cam_stats))
            total_grid += cam_stats["grid_total"]
            total_depth_positive += cam_stats["grid_depth_positive"]

    print("\nSUMMARY")
    print("grid_total:", total_grid)
    print("grid_depth_positive:", total_depth_positive)
    ratio = total_depth_positive / total_grid if total_grid else 0.0
    print("grid_depth_positive_ratio:", f"{ratio:.6f}")

    if args.fail_on_empty_depth and total_grid > 0 and total_depth_positive == 0:
        print("ERROR: all inspected BEV grid points have non-positive camera depth")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
