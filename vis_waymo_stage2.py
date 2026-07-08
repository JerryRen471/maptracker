"""BEV vector visualization for Waymo stage2 checkpoint."""
from pathlib import Path
from collections import defaultdict
import json, os, sys, importlib
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet3d.datasets import build_dataset

work_dir = Path(
    "/data/maptr_workspace/work_dirs/"
    "maptracker_waymo_5cam_5frame_span10_stage2_warmup_xm15_x45_y15_pretrain_init"
)
cfg_path = work_dir / "maptracker_waymo_5cam_5frame_span10_stage2_warmup_xm15_x45_y15_pretrain_init.py"
sub_path = work_dir / "submission_vector.json"
out_root = work_dir / "bev_vis_iter6351"

SCORE_THR = 0.30
MAX_SCENES = 3

colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
names = {0: "ped", 1: "divider", 2: "boundary"}

cfg = Config.fromfile(str(cfg_path))
sys.path.append(os.path.abspath("."))

if getattr(cfg, "plugin", False):
    plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
    for plugin_dir in plugin_dirs:
        parts = os.path.dirname(plugin_dir).split("/")
        mod = parts[0]
        for p in parts[1:]:
            mod += "." + p
        importlib.import_module(mod)

dataset = build_dataset(cfg.match_config)

roi_size = np.array(cfg.roi_size, dtype=float)
origin = np.array(cfg.pc_range[:2], dtype=float)
xlim = (origin[0], origin[0] + roi_size[0])
ylim = (origin[1], origin[1] + roi_size[1])

print("roi_size:", roi_size.tolist())
print("origin:", origin.tolist())
print("xlim:", xlim, "ylim:", ylim)
print("dataset len:", len(dataset))

with open(sub_path) as f:
    pred_results = json.load(f)["results"]
print("pred tokens:", len(pred_results))

scene_indices = defaultdict(list)
for idx, sample in enumerate(dataset.samples):
    scene_indices[sample["scene_name"]].append(idx)

for scene in scene_indices:
    scene_indices[scene].sort(key=lambda i: dataset.samples[i].get("frame_idx", i))

sorted_scenes = sorted(scene_indices.keys())[:MAX_SCENES]
print(f"visualizing {len(sorted_scenes)} scenes")


def draw_vectors(vectors_by_label, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.25)
    ax.axvline(0, color="0.6", linewidth=0.5, alpha=0.5)
    ax.axhline(0, color="0.6", linewidth=0.5, alpha=0.5)

    car = np.array([[-2.2, -1.0], [2.2, -1.0], [2.2, 1.0], [-2.2, 1.0], [-2.2, -1.0]])
    ax.plot(car[:, 0], car[:, 1], color="orange", linewidth=1.5)

    counts = []
    for label in [0, 1, 2]:
        vecs = vectors_by_label.get(label, [])
        counts.append(f"{names[label]}={len(vecs)}")
        for vec in vecs:
            arr = np.asarray(vec, dtype=float).reshape(-1, 2)
            if arr.shape[0] >= 2:
                ax.plot(arr[:, 0], arr[:, 1], "-", color=colors[label], linewidth=1.4, alpha=0.9)

    ax.set_title(title + "  " + ", ".join(counts), fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def save_scene(scene):
    out_dir = out_root / f"score_thr_{SCORE_THR:.2f}" / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    pred_counts = []
    missing_tokens = 0

    for local_idx, ds_idx in enumerate(scene_indices[scene]):
        sample = dataset.samples[ds_idx]
        item = dataset[ds_idx]

        gt_by_label = {}
        for label, vecs in item["vectors"].data.items():
            gt_by_label[int(label)] = [np.asarray(v, dtype=float) * roi_size + origin for v in vecs]

        pred = pred_results.get(sample["token"])
        if pred is None:
            missing_tokens += 1
            pred = {}

        pred_by_label = defaultdict(list)
        for vec, lab, score in zip(pred.get("vectors", []), pred.get("labels", []), pred.get("scores", [])):
            if float(score) < SCORE_THR:
                continue
            arr = np.asarray(vec, dtype=float).reshape(-1, 2)
            if arr.size and np.abs(arr).max() <= 1.5:
                arr = arr * roi_size + origin
            pred_by_label[int(lab)].append(arr)

        pred_counts.append(sum(len(v) for v in pred_by_label.values()))

        gt_path = out_dir / f"frame_{local_idx:04d}_gt.png"
        pred_path = out_dir / f"frame_{local_idx:04d}_pred.png"
        pair_path = out_dir / f"frame_{local_idx:04d}_gt_pred.png"

        draw_vectors(gt_by_label, gt_path, f"GT {local_idx:04d}")
        draw_vectors(pred_by_label, pred_path, f"Pred score>={SCORE_THR:.2f} {local_idx:04d}")

        gt_img = Image.open(gt_path).convert("RGB")
        pred_img = Image.open(pred_path).convert("RGB")
        w = max(gt_img.width, pred_img.width)
        h = max(gt_img.height, pred_img.height)

        pair = Image.new("RGB", (w * 2, h), "white")
        pair.paste(gt_img, ((w - gt_img.width) // 2, (h - gt_img.height) // 2))
        pair.paste(pred_img, (w + (w - pred_img.width) // 2, (h - pred_img.height) // 2))
        pair.save(pair_path)
        frames.append(np.asarray(pair))

    if frames:
        imageio.mimsave(out_dir / "gt_pred.mp4", frames, format="MP4", fps=10, macro_block_size=1)

        sample_ids = sorted(set([0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]))
        tiles = [Image.fromarray(frames[i]) for i in sample_ids]
        for t in tiles:
            t.thumbnail((1100, 620))
        tw = max(t.width for t in tiles)
        th = max(t.height for t in tiles)
        sheet = Image.new("RGB", (tw, th * len(tiles)), "white")
        for row, tile in enumerate(tiles):
            sheet.paste(tile, ((tw - tile.width) // 2, row * th + (th - tile.height) // 2))
        sheet.save(out_dir / "contact_sheet.png")

    return {
        "frames": len(frames),
        "missing_tokens": missing_tokens,
        "pred_min": int(min(pred_counts)) if pred_counts else 0,
        "pred_med": float(np.median(pred_counts)) if pred_counts else 0.0,
        "pred_max": int(max(pred_counts)) if pred_counts else 0,
    }


for scene in sorted_scenes:
    stat = save_scene(scene)
    print(scene, stat)

print("\noutput:", out_root)
