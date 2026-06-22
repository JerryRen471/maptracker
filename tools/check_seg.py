from pathlib import Path
import os
import sys
import importlib
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from plugin.datasets.builder import build_dataloader

cfg_path = "/root/maptracker/plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune.py"
work_dir = Path("/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_asym_roi/")
ckpt_path = str(work_dir / "latest.pth")
# "/root/maptracker/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage1_bev_pretrain/latest.pth"# str(work_dir / "latest.pth")
out_root = work_dir / "seg_vis_all_classes"

hard_thr = 0.4
max_frames = None

targets = {
    "ped_crossing": 0,
    "divider": 1,
    "boundary": 2,
}

palette = {
    0: (255, 255, 255),
    1: (31, 119, 180),
    2: (214, 39, 40),
    3: (44, 160, 44),
}

def import_plugin(cfg):
    sys.path.append(os.path.abspath("."))
    if getattr(cfg, "plugin", False):
        plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
        for plugin_dir in plugin_dirs:
            parts = os.path.dirname(plugin_dir).split("/")
            mod = parts[0]
            for p in parts[1:]:
                mod += "." + p
            importlib.import_module(mod)

def colorize_label(label_map):
    label_map = np.asarray(label_map, dtype=np.uint8)
    h, w = label_map.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in palette.items():
        img[label_map == k] = color
    return img

def gt_onehot_to_label(gt_semantic):
    gt = np.asarray(gt_semantic)
    label = np.zeros(gt.shape[1:], dtype=np.uint8)
    for c in range(gt.shape[0]):
        label[gt[c] > 0] = c + 1
    return label

def save_score_heatmap(score_map, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    im = ax.imshow(score_map, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)

def make_side_by_side(paths, out_path):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    target_h = max(img.height for img in imgs)
    resized = []
    for img in imgs:
        if img.height != target_h:
            new_w = int(img.width * target_h / img.height)
            img = img.resize((new_w, target_h))
        resized.append(img)

    w = sum(img.width for img in resized)
    canvas = Image.new("RGB", (w, target_h), "white")
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width
    canvas.save(out_path)

cfg = Config.fromfile(cfg_path)
import_plugin(cfg)

if cfg.data.train.get("type", None) == "RepeatDataset":
    dataset_cfg = copy.deepcopy(cfg.data.train.dataset)
else:
    dataset_cfg = copy.deepcopy(cfg.data.train)

vis_pipeline = []
for step in cfg.train_pipeline:
    step = copy.deepcopy(step)
    if step["type"] == "PhotoMetricDistortionMultiViewImage":
        continue
    vis_pipeline.append(step)

dataset_cfg.pipeline = vis_pipeline
dataset_cfg.test_mode = True
dataset_cfg.multi_frame = False
dataset_cfg.matching = False
dataset = build_dataset(dataset_cfg)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False,
    shuffler_sampler=cfg.data.get("shuffler_sampler", None),
    nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
)

cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, ckpt_path, map_location="cpu")
model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()

captured = {}

def hook_seg_decoder(module, inputs, output):
    captured["seg_preds"] = output[0].detach().cpu()

handle = model.module.seg_decoder.register_forward_hook(hook_seg_decoder)

out_root.mkdir(parents=True, exist_ok=True)
print("dataset length:", len(dataset))
print("checkpoint:", ckpt_path)
print("output:", out_root)

summary = []
with torch.no_grad():
    for idx, data in enumerate(data_loader):
        if max_frames is not None and idx >= max_frames:
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

        sample = dataset.samples[idx]
        scene = sample["scene_name"]
        frame_idx = int(sample.get("frame_idx", idx))
        frame_dir = out_root / scene / f"frame_{frame_idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        gt_path = frame_dir / "gt_semantic_mask.png"
        pred_path = frame_dir / "pred_hard_thr_0.4.png"

        gt_img = colorize_label(gt_label)
        pred_img = colorize_label(pred_label)

        Image.fromarray(gt_img).save(gt_path)
        Image.fromarray(pred_img).save(pred_path)

        heatmap_paths = []
        score_stats = {}
        for target_name, target_idx in targets.items():
            score = seg_scores[target_idx]
            heatmap_path = frame_dir / f"pred_score_{target_name}.png"
            save_score_heatmap(
                score,
                heatmap_path,
                f"{scene} frame={frame_idx} {target_name} score"
            )
            heatmap_paths.append(heatmap_path)
            score_stats[f"{target_name}_score_max"] = float(score.max())
            score_stats[f"{target_name}_score_p99"] = float(np.percentile(score, 99))
            score_stats[f"{target_name}_score_mean"] = float(score.mean())

        make_side_by_side(
            [gt_path, pred_path] + heatmap_paths,
            frame_dir / "gt_pred_side_by_side.png"
        )

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
        print(stat)

handle.remove()

summary_path = out_root / "summary.txt"
with open(summary_path, "w") as f:
    for row in summary:
        f.write(str(row) + "\n")

print("DONE")
print("summary:", summary_path)
