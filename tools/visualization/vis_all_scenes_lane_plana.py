#!/usr/bin/env python3
"""Batch lane visualization for all val scenes (planA-style).

Adapted from docs/single_scene_lane_vis_0708_planA.md for the
xm15_x45_y15_pinhole_roi_fixed full-val submission (correct asym ROI).

Per scene outputs:
  - check_vector_score_hist.png
  - vis_bev_score_{thr}/frame_*_{gt,pred,gt_pred}.png + gt_pred.mp4
  - cam_proj_pred_score_{thr}/ + cam_proj_gt/ (stride / max_frames)
"""

from __future__ import annotations

import argparse
import json
import pickle
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import av2.geometry.interpolate as interp_utils

    HAS_AV2 = True
except Exception:
    HAS_AV2 = False

ID2CAT = {0: "ped_crossing", 1: "divider", 2: "boundary"}
COLOR_BGR = {
    "divider": (0, 0, 255),
    "boundary": (0, 255, 0),
    "ped_crossing": (255, 0, 0),
}
BEV_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
BEV_NAMES = {0: "ped", 1: "divider", 2: "boundary"}
CAM_ORDER = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ann",
        default="/data/waymo_maptracker_xm15_x45_y15_pinhole/waymo_map_infos_val.pkl",
    )
    p.add_argument(
        "--submission",
        default=(
            "/data/maptr_workspace/work_dirs/"
            "maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole_roi_fixed/"
            "eval_full_val/submission_vector.json"
        ),
    )
    p.add_argument(
        "--out-root",
        default=(
            "/data/maptr_workspace/work_dirs/"
            "maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole_roi_fixed/"
            "eval_full_val/vis_lane_planA_all"
        ),
    )
    p.add_argument("--score-thr", type=float, default=0.30)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--max-cam-frames", type=int, default=8)
    p.add_argument("--xlim", type=float, nargs=2, default=[-15.0, 45.0])
    p.add_argument("--ylim", type=float, nargs=2, default=[-15.0, 15.0])
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--scenes", nargs="*", default=None, help="Optional scene filter")
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--skip-cam", type=int, default=0)
    p.add_argument("--skip-bev", type=int, default=0)
    p.add_argument("--skip-score-hist", type=int, default=0)
    return p.parse_args()


def scene_done(scene_out: Path, score_thr: float, skip_cam: bool, skip_bev: bool) -> bool:
    ok = True
    if not skip_bev:
        ok = ok and (scene_out / f"vis_bev_score_{score_thr:.2f}" / "gt_pred.mp4").exists()
    if not skip_cam:
        pred_panels = list((scene_out / f"cam_proj_pred_score_{score_thr:.2f}").glob("*_PRED_panel.jpg"))
        ok = ok and len(pred_panels) > 0
    return ok


def save_score_hist(scores, out_path: Path, title: str, thr: float):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    scores = np.asarray(scores, dtype=float) if len(scores) else np.array([])
    if len(scores):
        ax.hist(scores, bins=50, range=(0, 1), color="#4c78a8", alpha=0.9)
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.5, label=f"score_thr={thr:.2f}")
        ax.legend()
    ax.set_title(f"{title}\nn={len(scores)}")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def draw_bev(vecs, path, title, xlim, ylim, dpi):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.25, alpha=0.25)
    car = np.array([[-2.2, -1], [2.2, -1], [2.2, 1], [-2.2, 1], [-2.2, -1]])
    ax.plot(car[:, 0], car[:, 1], color="orange", lw=1.5)
    counts = []
    for lab in [0, 1, 2]:
        vs = vecs.get(lab, [])
        counts.append(f"{BEV_NAMES[lab]}={len(vs)}")
        for v in vs:
            a = np.asarray(v, float).reshape(-1, 2)
            if len(a) >= 2:
                ax.plot(a[:, 0], a[:, 1], "-", color=BEV_COLORS[lab], lw=1.4, alpha=0.9)
    ax.set_title(title + "  " + ", ".join(counts), fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def remove_nan_values(uv):
    valid = np.logical_and(~np.isnan(uv[:, 0]), ~np.isnan(uv[:, 1]))
    return uv[valid]


def points_ego2img(pts_ego, ego2cam, intrinsics):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = ego2cam @ pts_ego_4d.T
    uv = (intrinsics @ pts_cam_4d[:3, :]).T
    uv = remove_nan_values(uv)
    if len(uv) == 0:
        return uv, np.array([])
    depth = uv[:, 2]
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    return uv, depth


def interp_poly(polyline_ego):
    if HAS_AV2:
        try:
            return interp_utils.interp_arc(t=500, points=polyline_ego.astype(np.float64))
        except Exception:
            pass
    pts = []
    for i in range(len(polyline_ego) - 1):
        pts.append(np.linspace(polyline_ego[i], polyline_ego[i + 1], 20))
    return np.concatenate(pts, axis=0) if pts else polyline_ego


def draw_polyline_ego_on_img(polyline_ego, img_bgr, ego2cam, intrinsics, color_bgr, thickness=3):
    polyline_ego = np.asarray(polyline_ego, dtype=np.float64)
    if polyline_ego.shape[1] == 2:
        polyline_ego = np.concatenate(
            [polyline_ego, np.zeros((polyline_ego.shape[0], 1))], axis=1
        )
    if len(polyline_ego) < 2:
        return
    polyline_ego = interp_poly(polyline_ego)
    uv, depth = points_ego2img(polyline_ego, ego2cam, intrinsics)
    if len(uv) == 0:
        return
    h, w = img_bgr.shape[:2]
    valid = (
        (0 <= uv[:, 0])
        & (uv[:, 0] < w - 1)
        & (0 <= uv[:, 1])
        & (uv[:, 1] < h - 1)
        & (depth > 0)
    )
    if valid.sum() == 0:
        return
    uv = np.round(uv[valid]).astype(np.int32)
    for i in range(len(uv) - 1):
        if np.linalg.norm(uv[i + 1] - uv[i].astype(float)) > 200:
            continue
        cv2.line(img_bgr, tuple(uv[i]), tuple(uv[i + 1]), color_bgr, thickness, cv2.LINE_AA)


def collect_pred(pred, score_thr):
    by_label = defaultdict(list)
    for vec, lab, score in zip(
        pred.get("vectors", []), pred.get("labels", []), pred.get("scores", [])
    ):
        if float(score) < score_thr:
            continue
        arr = np.asarray(vec, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] >= 2:
            by_label[int(lab)].append(arr)
    return by_label


def collect_gt(sample):
    by_label = defaultdict(list)
    for vec, lab in zip(sample.get("gt_polylines", []), sample.get("gt_polyline_labels", [])):
        arr = np.asarray(vec, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] >= 2:
            by_label[int(lab)].append(arr)
    return by_label


def make_panel(cam_imgs, cam_names, out_path, title):
    target_h = 360
    resized = []
    for name, img in zip(cam_names, cam_imgs):
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        im = cv2.resize(img, (new_w, target_h))
        cv2.putText(im, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        resized.append(im)
    order_map = {n: i for i, n in enumerate(cam_names)}
    row1 = [n for n in ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"] if n in order_map]
    row2 = [n for n in ["SIDE_LEFT", "SIDE_RIGHT"] if n in order_map]

    def concat_row(names):
        return np.concatenate([resized[order_map[n]] for n in names], axis=1)

    panel = concat_row(row1)
    if row2:
        r2 = concat_row(row2)
        if r2.shape[1] < panel.shape[1]:
            pad_w = panel.shape[1] - r2.shape[1]
            pad = np.zeros((r2.shape[0], pad_w, 3), dtype=np.uint8)
            r2 = np.concatenate([pad[:, : pad_w // 2], r2, pad[:, pad_w // 2 :]], axis=1)
            if r2.shape[1] < panel.shape[1]:
                r2 = np.concatenate(
                    [r2, np.zeros((r2.shape[0], panel.shape[1] - r2.shape[1], 3), dtype=np.uint8)],
                    axis=1,
                )
            r2 = r2[:, : panel.shape[1]]
        panel = np.concatenate([panel, r2], axis=0)
    bar = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title[:120], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    panel = np.concatenate([bar, panel], axis=0)
    cv2.imwrite(str(out_path), panel)


def render_cam(sample, vectors, out_root, tag, score_thr, scene):
    frame_idx = int(sample.get("frame_idx", 0))
    cams = sample["cams"]
    cam_names = [n for n in CAM_ORDER if n in cams]
    frame_dir = out_root / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for cam_name in cam_names:
        c = cams[cam_name]
        img = cv2.imread(c["img_fpath"])
        if img is None:
            continue
        ego2cam = np.asarray(c["extrinsics"], dtype=np.float64)
        intr = np.asarray(c["intrinsics"], dtype=np.float64)
        img_draw = img.copy()
        for label, vecs in vectors.items():
            color = COLOR_BGR[ID2CAT[label]]
            for vec in vecs:
                draw_polyline_ego_on_img(vec, img_draw, ego2cam, intr, color, thickness=3)
        y0 = 30
        for cat, col in [
            ("ped_crossing", COLOR_BGR["ped_crossing"]),
            ("divider", COLOR_BGR["divider"]),
            ("boundary", COLOR_BGR["boundary"]),
        ]:
            cv2.putText(img_draw, cat, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
            y0 += 36
        cv2.putText(
            img_draw,
            f"{tag} thr={score_thr:.2f} frame={frame_idx}",
            (20, y0 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(frame_dir / f"{cam_name}.jpg"), img_draw)
        rendered.append(img_draw)
    if rendered:
        make_panel(
            rendered,
            cam_names,
            out_root / f"frame_{frame_idx:04d}_{tag}_panel.jpg",
            f"[{tag}] {scene} frame={frame_idx} score>={score_thr:.2f}",
        )


def process_scene(scene, samples, preds, args, out_root: Path):
    scene_out = out_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)
    thr = args.score_thr

    if (not args.overwrite) and scene_done(scene_out, thr, bool(args.skip_cam), bool(args.skip_bev)):
        return "skip"

    # score hist for scene
    if not args.skip_score_hist:
        scores = []
        for s in samples:
            pr = preds.get(s["token"])
            if pr:
                scores.extend(pr.get("scores", []))
        save_score_hist(
            scores,
            scene_out / "check_vector_score_hist.png",
            f"roi_fixed pred scores — {scene}",
            thr,
        )

    # BEV
    if not args.skip_bev:
        out_dir = scene_out / f"vis_bev_score_{thr:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        missing = 0
        samples_sorted = sorted(samples, key=lambda s: int(s.get("frame_idx", 0)))
        for sample in samples_sorted:
            gt = defaultdict(list)
            for lab, poly in zip(sample["gt_polyline_labels"], sample["gt_polylines"]):
                gt[int(lab)].append(np.asarray(poly, dtype=float))

            pred = preds.get(sample["token"])
            if pred is None:
                missing += 1
                pred = {}
            pb = defaultdict(list)
            for v, lab, sc in zip(
                pred.get("vectors", []), pred.get("labels", []), pred.get("scores", [])
            ):
                if float(sc) >= thr:
                    pb[int(lab)].append(v)

            fi = int(sample.get("frame_idx", 0))
            gt_p = out_dir / f"frame_{fi:04d}_gt.png"
            pr_p = out_dir / f"frame_{fi:04d}_pred.png"
            gp_p = out_dir / f"frame_{fi:04d}_gt_pred.png"
            draw_bev(gt, gt_p, f"[roi_fixed] GT frame={fi}", args.xlim, args.ylim, args.dpi)
            draw_bev(
                pb,
                pr_p,
                f"[roi_fixed] Pred score>={thr} frame={fi}",
                args.xlim,
                args.ylim,
                args.dpi,
            )

            g = np.asarray(Image.open(gt_p))
            p = np.asarray(Image.open(pr_p))
            h = max(g.shape[0], p.shape[0])

            def pad(im, hh):
                if im.shape[0] == hh:
                    return im
                out = np.zeros((hh, im.shape[1], 3), dtype=im.dtype)
                out[: im.shape[0]] = im
                return out

            panel = np.concatenate([pad(g, h), pad(p, h)], axis=1)
            Image.fromarray(panel).save(gp_p)
            panels.append(panel)

        if panels:
            imageio.mimsave(out_dir / "gt_pred.mp4", panels, fps=2)
        (out_dir / "missing_pred_tokens.txt").write_text(str(missing) + "\n")

    # Cam projection
    if not args.skip_cam:
        samples_sorted = sorted(samples, key=lambda s: int(s.get("frame_idx", 0)))
        sel = samples_sorted[:: args.frame_stride][: args.max_cam_frames]
        out_pred = scene_out / f"cam_proj_pred_score_{thr:.2f}"
        out_gt = scene_out / "cam_proj_gt"
        out_pred.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)
        for sample in sel:
            render_cam(sample, collect_gt(sample), out_gt, "GT", thr, scene)
            pred = preds.get(sample["token"])
            if pred is None:
                continue
            render_cam(
                sample, collect_pred(pred, thr), out_pred, "PRED", thr, scene
            )

    return "ok"


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "PROGRESS.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"loading ann: {args.ann}")
    with open(args.ann, "rb") as f:
        ann = pickle.load(f)
    samples = ann["samples"]
    by_scene = defaultdict(list)
    for s in samples:
        by_scene[s["scene_name"]].append(s)
    scenes = sorted(by_scene.keys())
    if args.scenes:
        scenes = [s for s in scenes if s in set(args.scenes)]
    log(f"scenes={len(scenes)} samples={len(samples)}")

    log(f"loading submission: {args.submission}")
    with open(args.submission) as f:
        preds_all = json.load(f)["results"]
    log(f"pred tokens={len(preds_all)}")

    # global score hist
    if not args.skip_score_hist:
        all_scores = []
        for pr in preds_all.values():
            all_scores.extend(pr.get("scores", []))
        save_score_hist(
            all_scores,
            out_root / "check_vector_score_hist.png",
            "roi_fixed full val pred scores",
            args.score_thr,
        )
        log(f"wrote global score hist, n_scores={len(all_scores)}")

    # INDEX header
    index_path = out_root / "INDEX.md"
    if not index_path.exists() or args.overwrite:
        index_path.write_text(
            "# Lane vis (planA) for all val scenes — roi_fixed full val\n\n"
            f"- submission: `{args.submission}`\n"
            f"- ann (correct asym ROI): `{args.ann}`\n"
            f"- ROI xlim={args.xlim} ylim={args.ylim}\n"
            f"- score_thr={args.score_thr}, cam stride={args.frame_stride}, max_cam_frames={args.max_cam_frames}\n"
            f"- scenes: {len(scenes)}\n\n"
            "Follows `docs/single_scene_lane_vis_0708_planA.md` steps 2–4 "
            "(GT from correct pinhole pack, not legacy v3 ROI).\n"
        )

    done = skip = fail = 0
    for i, scene in enumerate(scenes, 1):
        try:
            # filter preds for this scene to keep lookups local
            scene_preds = {
                t: preds_all[t]
                for t in preds_all
                if t.startswith(scene + "_frame_")
            }
            # also allow exact token matches from samples
            for s in by_scene[scene]:
                tok = s["token"]
                if tok in preds_all:
                    scene_preds[tok] = preds_all[tok]
            status = process_scene(scene, by_scene[scene], scene_preds, args, out_root)
            if status == "skip":
                skip += 1
                log(f"[{i}/{len(scenes)}] SKIP {scene}")
            else:
                done += 1
                log(f"[{i}/{len(scenes)}] OK   {scene} frames={len(by_scene[scene])} preds={len(scene_preds)}")
        except Exception as e:
            fail += 1
            log(f"[{i}/{len(scenes)}] FAIL {scene}: {e}")
            log(traceback.format_exc())

    log(f"FINISHED done={done} skip={skip} fail={fail} total={len(scenes)}")


if __name__ == "__main__":
    main()
