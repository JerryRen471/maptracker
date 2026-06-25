#!/usr/bin/env python3
"""Visualize MapTracker BEV semantic predictions and score heatmaps."""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGETS = {
    "ped_crossing": 0,
    "divider": 1,
    "boundary": 2,
}

PALETTE = {
    0: (255, 255, 255),
    1: (31, 119, 180),
    2: (214, 39, 40),
    3: (44, 160, 44),
}


def parse_args() -> argparse.Namespace:
    from mmcv import DictAction

    parser = argparse.ArgumentParser(
        description="Render BEV semantic GT, predicted masks, and per-class score heatmaps."
    )
    parser.add_argument("--config", required=True, help="Config file used to build the model and dataset")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--out-dir", required=True, help="Directory for BEV semantic visualizations")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split whose ann_file should be visualized",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to render")
    parser.add_argument("--scene-id", nargs="+", default=None, help="Optional scene_name values to render")
    parser.add_argument("--workers-per-gpu", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id inside CUDA_VISIBLE_DEVICES")
    parser.add_argument("--score-dpi", type=int, default=140, help="DPI for score heatmap figures")
    parser.add_argument("--draw-bev-range", type=int, default=1, help="Draw the BEV ROI range border")
    parser.add_argument(
        "--no-score-heatmaps",
        action="store_true",
        help="Only render GT/predicted hard masks and side-by-side summaries",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="Override config options, same format as tools/train.py.",
    )
    return parser.parse_args()


def import_plugin(cfg) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if getattr(cfg, "plugin", False):
        plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
        for plugin_dir in plugin_dirs:
            parts = os.path.dirname(plugin_dir).split("/")
            mod = parts[0]
            for part in parts[1:]:
                mod += "." + part
            importlib.import_module(mod)


def colorize_label(label_map):
    import numpy as np

    label_map = np.asarray(label_map, dtype=np.uint8)
    h, w = label_map.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for key, color in PALETTE.items():
        img[label_map == key] = color
    return img


def bev_border_box(width: int, height: int):
    return (0, 0, max(0, width - 1), max(0, height - 1))


def draw_bev_border(image, enabled: bool = True):
    if not enabled:
        return image
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle(bev_border_box(width, height), outline=(242, 201, 76), width=3)
    return image


def save_label_image(label_map, out_path: Path, draw_bev_range: bool) -> None:
    from PIL import Image

    image = Image.fromarray(colorize_label(label_map))
    draw_bev_border(image, enabled=draw_bev_range)
    image.save(out_path)


def gt_onehot_to_label(gt_semantic):
    import numpy as np

    gt = np.asarray(gt_semantic)
    label = np.zeros(gt.shape[1:], dtype=np.uint8)
    for channel in range(gt.shape[0]):
        label[gt[channel] > 0] = channel + 1
    return label


def save_score_heatmap(score_map, out_path: Path, title: str, dpi: int, draw_bev_range: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    im = ax.imshow(score_map, cmap="magma", vmin=0.0, vmax=1.0)
    if draw_bev_range:
        height, width = score_map.shape[:2]
        ax.add_patch(
            patches.Rectangle(
                (-0.5, -0.5),
                width,
                height,
                fill=False,
                edgecolor="#f2c94c",
                linewidth=2.0,
                linestyle="--",
            )
        )
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def make_side_by_side(paths: list[Path], out_path: Path) -> None:
    from PIL import Image

    imgs = [Image.open(path).convert("RGB") for path in paths]
    target_h = max(img.height for img in imgs)
    resized = []
    for img in imgs:
        if img.height != target_h:
            new_w = int(img.width * target_h / img.height)
            img = img.resize((new_w, target_h))
        resized.append(img)

    width = sum(img.width for img in resized)
    canvas = Image.new("RGB", (width, target_h), "white")
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width
    canvas.save(out_path)


def select_dataset_cfg(cfg, split: str):
    if split == "train" and cfg.data.train.get("type", None) == "RepeatDataset":
        return copy.deepcopy(cfg.data.train.dataset)
    return copy.deepcopy(cfg.data[split])


def build_vis_pipeline(cfg):
    pipeline = getattr(cfg, "train_pipeline", None)
    if pipeline is None:
        raise ValueError("config must define train_pipeline so semantic_mask can be rendered")

    vis_pipeline = []
    for step in pipeline:
        step = copy.deepcopy(step)
        if step["type"] == "PhotoMetricDistortionMultiViewImage":
            continue
        vis_pipeline.append(step)
    return vis_pipeline


def render(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    from plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    import_plugin(cfg)

    dataset_cfg = select_dataset_cfg(cfg, args.split)
    dataset_cfg.pipeline = build_vis_pipeline(cfg)
    dataset_cfg.test_mode = True
    dataset_cfg.multi_frame = False
    dataset_cfg.matching = False

    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
        shuffler_sampler=cfg.data.get("shuffler_sampler", None),
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(args.device_id), device_ids=[args.device_id])
    model.eval()

    captured = {}

    def hook_seg_decoder(module, inputs, output):
        captured["seg_preds"] = output[0].detach().cpu()

    handle = model.module.seg_decoder.register_forward_hook(hook_seg_decoder)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print("dataset length:", len(dataset))
    print("config:", args.config)
    print("checkpoint:", args.checkpoint)
    print("output:", out_root)
    print("split:", args.split)

    scene_filter = set(args.scene_id or [])
    summary = []
    rendered = 0
    try:
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                sample = dataset.samples[idx]
                scene = sample["scene_name"]
                if scene_filter and scene not in scene_filter:
                    continue
                if args.max_frames is not None and rendered >= args.max_frames:
                    break

                captured.clear()
                result = model(return_loss=False, rescale=True, **data)
                result0 = result[0] if isinstance(result, list) else result

                if "seg_preds" not in captured:
                    raise RuntimeError("seg_decoder hook did not capture seg_preds")

                seg_logits = captured["seg_preds"][0].float().numpy()
                seg_scores = 1.0 / (1.0 + np.exp(-seg_logits))
                pred_label = np.asarray(result0["semantic_mask"], dtype=np.uint8)

                gt_dc = dataset[idx]["semantic_mask"]
                gt_semantic = gt_dc.data
                if hasattr(gt_semantic, "numpy"):
                    gt_semantic = gt_semantic.numpy()
                gt_semantic = np.asarray(gt_semantic)

                # Match training-time orientation in MapTracker.forward_train.
                gt_semantic = np.flip(gt_semantic, axis=1)
                gt_label = gt_onehot_to_label(gt_semantic)

                frame_idx = int(sample.get("frame_idx", idx))
                frame_dir = out_root / scene / f"frame_{frame_idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)

                gt_path = frame_dir / "gt_semantic_mask.png"
                pred_path = frame_dir / "pred_hard_thr_0.4.png"
                draw_bev_range = bool(args.draw_bev_range)
                save_label_image(gt_label, gt_path, draw_bev_range)
                save_label_image(pred_label, pred_path, draw_bev_range)

                panel_paths = [gt_path, pred_path]
                score_stats = {}
                if not args.no_score_heatmaps:
                    for target_name, target_idx in TARGETS.items():
                        score = seg_scores[target_idx]
                        heatmap_path = frame_dir / f"pred_score_{target_name}.png"
                        save_score_heatmap(
                            score,
                            heatmap_path,
                            f"{scene} frame={frame_idx} {target_name} score",
                            args.score_dpi,
                            draw_bev_range,
                        )
                        panel_paths.append(heatmap_path)
                        score_stats[f"{target_name}_score_max"] = float(score.max())
                        score_stats[f"{target_name}_score_p99"] = float(np.percentile(score, 99))
                        score_stats[f"{target_name}_score_mean"] = float(score.mean())

                make_side_by_side(panel_paths, frame_dir / "gt_pred_side_by_side.png")

                stat = {
                    "idx": idx,
                    "scene": scene,
                    "frame_idx": frame_idx,
                    "gt_ped": int((gt_label == 1).sum()),
                    "gt_divider": int((gt_label == 2).sum()),
                    "gt_boundary": int((gt_label == 3).sum()),
                    "pred_ped": int((pred_label == 1).sum()),
                    "pred_divider": int((pred_label == 2).sum()),
                    "pred_boundary": int((pred_label == 3).sum()),
                }
                stat.update(score_stats)
                summary.append(stat)
                rendered += 1
                print(stat)
    finally:
        handle.remove()

    summary_path = out_root / "summary.txt"
    with summary_path.open("w") as f:
        for row in summary:
            f.write(str(row) + "\n")

    print("DONE")
    print("rendered:", rendered)
    print("summary:", summary_path)


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
