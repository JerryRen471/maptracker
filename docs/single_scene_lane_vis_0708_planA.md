# 单场景车道线可视化（0708 实验）

用 `0708_pinhole_stage123.out` 对应训练产生的**已有全量 val 预测**，对单个场景做可视化。**不重新推理**。

流程包括：
1. 准备与 0708 匹配的旧 ROI 单场景 GT
2. **预测向量分数检查**（`tools/check_vector.py` 风格）
3. **BEV** 车道线 GT / Pred 对比图 + mp4
4. **摄像头反投影**：把预测（及 GT）车道线画回 5 路原图 + panel 拼图

> 说明：`tools/check_vector.py` 本身只统计 `submission_vector.json` 的 score 分布，不画图。  
> 车道线几何可视化在后面的 BEV / 摄像头步骤；本流程把 check_vector 的检查接到 0708 的 JSON 上，并可选输出分数直方图。


## 用到的路径

| 项 | 路径 |
|---|---|
| 训练日志 | `/root/maptracker/0708_pinhole_stage123.out` |
| Stage3 work_dir | `/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole/` |
| 已有预测 | `.../eval_full_val/submission_vector.json` |
| 配置 | `/data/maptr_workspace/work_dirs/generated_configs/xm15_x45_y15_pinhole/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole.py` |
| 旧 ROI 数据源 | `/data/waymo_processed_v3`（`bev_x=15, bev_y=30`，约 `x∈[-15,15], y∈[-30,30]`） |

**注意：** 当前主目录 `/data/waymo_maptracker_xm15_x45_y15_pinhole` 已换成 asym ROI，**不要**用它当本流程的 GT。Pred 用 `eval_full_val/submission_vector.json`；GT 从 `waymo_processed_v3` pack 后再切场景。

---

## 0. 环境变量

```bash
cd /root/maptracker
source /root/miniconda3/etc/profile.d/conda.sh
conda activate maptracker
export PYTHONPATH=/root/maptracker:${PYTHONPATH:-}

WORK=/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole
SUB_JSON=$WORK/eval_full_val/submission_vector.json
CFG=/data/maptr_workspace/work_dirs/generated_configs/xm15_x45_y15_pinhole/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole.py

# 改成要看的 val 场景
SCENE=segment-10023947602400723454_1120_000_1140_000_with_camera_labels
SCORE_THR=0.30
OUT=$WORK/vis_one_scene_0708/$SCENE

# 本流程生成的旧 ROI 单场景子集
SUB=/data/waymo_maptracker_xm15_x45_y15_pinhole_scene1_val_legacyroi_0708
TMP=/data/waymo_maptracker_from_v3_legacyroi

export WORK SUB_JSON CFG SCENE SCORE_THR OUT SUB TMP
```

确认预测文件存在：

```bash
ls -lh "$SUB_JSON"
```

---

## 1. 准备单场景 GT（旧 ROI）

从 `waymo_processed_v3` pack（**不要**加 `--expect-roi-range`），再切出目标场景：

```bash
# 1a. 全量 pack 一次即可（若 $TMP 已有且 metadata 为 bev_x=15, bev_y=30 可跳过）
python pack_waymo_for_maptracker.py \
  --v3-dir /data/waymo_processed_v3 \
  --out-dir "$TMP" \
  --img-root /data/waymo_processed_v3

# 1b. 切单场景
python - <<'PY'
import os, pickle
from pathlib import Path

scene = os.environ["SCENE"]
tmp = Path(os.environ["TMP"])
sub = Path(os.environ["SUB"])
sub.mkdir(parents=True, exist_ok=True)

with open(tmp / "waymo_map_infos_val.pkl", "rb") as f:
    val = pickle.load(f)

samples = [s for s in val["samples"] if s["scene_name"] == scene]
assert samples, f"scene not in val: {scene}"
print("frames:", len(samples), "meta:", val.get("metadata"))

payload = dict(val)
payload["samples"] = samples
for split in ("train", "val"):
    out = sub / f"waymo_map_infos_{split}.pkl"
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote", out)

(sub / "selected_scenes.txt").write_text(scene + "\n")
print("OK", scene)
PY
```

可选校验（坐标应接近 `x[-15,15], y[-30,30]`）：

```bash
python - <<'PY'
import os, pickle, numpy as np
with open(os.environ["SUB"] + "/waymo_map_infos_val.pkl", "rb") as f:
    d = pickle.load(f)
print("metadata", d["metadata"])
xs, ys = [], []
for s in d["samples"]:
    for p in s["gt_polylines"]:
        a = np.asarray(p)
        xs += [a[:, 0].min(), a[:, 0].max()]
        ys += [a[:, 1].min(), a[:, 1].max()]
print("x", min(xs), max(xs), "y", min(ys), max(ys))
PY
```

---

## 2. 预测向量分数检查（`tools/check_vector.py`）

仓库里的 `tools/check_vector.py` 默认写死了另一个 `submission_vector.json` 路径。对本流程，直接对 0708 的 `$SUB_JSON` 做同样统计（全量 + 当前 `SCENE`）。

### 2.1 命令行统计

```bash
# 方式 A：临时改路径后跑原脚本
python - <<'PY'
# 等价于 tools/check_vector.py，但指向 0708 预测
import json, os, numpy as np

p = os.environ["SUB_JSON"]
with open(p) as f:
    r = json.load(f)["results"]

scores, counts = [], []
for token, pred in r.items():
    s = pred.get("scores", [])
    scores.extend(s)
    counts.append(len(s))

print("=== full val submission ===")
print("file:", p)
print("tokens:", len(r))
print("pred count per frame min/med/max:", min(counts), float(np.median(counts)), max(counts))
if scores:
    scores = np.asarray(scores, dtype=float)
    print(
        "score min/p50/p90/p99/max:",
        float(np.min(scores)),
        float(np.percentile(scores, 50)),
        float(np.percentile(scores, 90)),
        float(np.percentile(scores, 99)),
        float(np.max(scores)),
    )
    for thr in (0.01, 0.05, 0.30, 0.50):
        print(f"num >= {thr:.2f}:", int((scores >= thr).sum()))
else:
    print("NO SCORES")
PY
```

只统计当前场景（需已完成第 1 步，用 `$SUB` 的 token 过滤）：

```bash
python - <<'PY'
import json, os, pickle, numpy as np

scene = os.environ["SCENE"]
sub = os.environ["SUB"]
with open(f"{sub}/waymo_map_infos_val.pkl", "rb") as f:
    tokens = {s["token"] for s in pickle.load(f)["samples"] if s["scene_name"] == scene}

with open(os.environ["SUB_JSON"]) as f:
    all_pred = json.load(f)["results"]
r = {t: all_pred[t] for t in tokens if t in all_pred}

scores, counts = [], []
for pred in r.values():
    s = pred.get("scores", [])
    scores.extend(s)
    counts.append(len(s))

print(f"=== scene: {scene} ===")
print("tokens:", len(r), "(missing)", len(tokens) - len(r))
if counts:
    print("pred count per frame min/med/max:", min(counts), float(np.median(counts)), max(counts))
if scores:
    scores = np.asarray(scores, dtype=float)
    print(
        "score min/p50/p90/p99/max:",
        float(np.min(scores)),
        float(np.percentile(scores, 50)),
        float(np.percentile(scores, 90)),
        float(np.percentile(scores, 99)),
        float(np.max(scores)),
    )
    thr = float(os.environ["SCORE_THR"])
    print(f"num >= {thr:.2f}:", int((scores >= thr).sum()))
else:
    print("NO SCORES")
PY
```

### 2.2 分数直方图（可选可视化）

```bash
mkdir -p "$OUT"
python - <<'PY'
import json, os, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

scene = os.environ["SCENE"]
out = Path(os.environ["OUT"])
thr = float(os.environ["SCORE_THR"])

with open(f"{os.environ['SUB']}/waymo_map_infos_val.pkl", "rb") as f:
    tokens = {s["token"] for s in pickle.load(f)["samples"] if s["scene_name"] == scene}
with open(os.environ["SUB_JSON"]) as f:
    all_pred = json.load(f)["results"]

scores = []
for t in tokens:
    if t in all_pred:
        scores.extend(all_pred[t].get("scores", []))
scores = np.asarray(scores, dtype=float) if scores else np.array([])

fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
if len(scores):
    ax.hist(scores, bins=50, range=(0, 1), color="#4c78a8", alpha=0.9)
    ax.axvline(thr, color="red", linestyle="--", linewidth=1.5, label=f"score_thr={thr:.2f}")
    ax.legend()
ax.set_title(f"0708 pred scores — {scene}\nn={len(scores)}")
ax.set_xlabel("score")
ax.set_ylabel("count")
fig.tight_layout()
out_path = out / "check_vector_score_hist.png"
fig.savefig(out_path)
plt.close(fig)
print("wrote", out_path)
PY
```

后续 BEV / 摄像头步骤里的 `SCORE_THR`（默认 `0.30`）应参考这里的分数分布再决定。

---

## 3. BEV 车道线可视化（GT / Pred）

Pred 读 `$SUB_JSON`，GT 读 `$SUB` 里该场景的 `gt_polylines`。  
（这一步才是车道线矢量的几何可视化；`check_vector.py` 不画线。）

```bash
mkdir -p "$OUT"

python - <<'PY'
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

CFG = os.environ["CFG"]
SUB = os.environ["SUB"]
SUB_JSON = os.environ["SUB_JSON"]
OUT = Path(os.environ["OUT"])
SCENE = os.environ["SCENE"]
SCORE_THR = float(os.environ["SCORE_THR"])

cfg = Config.fromfile(CFG)
sys.path.append(os.path.abspath("."))
if getattr(cfg, "plugin", False):
    plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
    for plugin_dir in plugin_dirs:
        parts = os.path.dirname(plugin_dir).split("/")
        mod = parts[0]
        for p in parts[1:]:
            mod += "." + p
        importlib.import_module(mod)

cfg.match_config.ann_file = f"{SUB}/waymo_map_infos_val.pkl"
cfg.match_config.interval = 1
dataset = build_dataset(cfg.match_config)

# 画布按 0708 实际数据范围
xlim = (-15.0, 15.0)
ylim = (-30.0, 30.0)

with open(SUB_JSON) as f:
    preds = json.load(f)["results"]

colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
names = {0: "ped", 1: "divider", 2: "boundary"}
out_dir = OUT / f"vis_bev_score_{SCORE_THR:.2f}"
out_dir.mkdir(parents=True, exist_ok=True)


def draw(vecs, path, title):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.25, alpha=0.25)
    car = np.array([[-2.2, -1], [2.2, -1], [2.2, 1], [-2.2, 1], [-2.2, -1]])
    ax.plot(car[:, 0], car[:, 1], color="orange", lw=1.5)
    counts = []
    for lab in [0, 1, 2]:
        vs = vecs.get(lab, [])
        counts.append(f"{names[lab]}={len(vs)}")
        for v in vs:
            a = np.asarray(v, float).reshape(-1, 2)
            if len(a) >= 2:
                ax.plot(a[:, 0], a[:, 1], "-", color=colors[lab], lw=1.4, alpha=0.9)
    ax.set_title(title + "  " + ", ".join(counts), fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


idxs = [i for i, s in enumerate(dataset.samples) if s["scene_name"] == SCENE]
idxs.sort(key=lambda i: dataset.samples[i].get("frame_idx", i))
print("frames", len(idxs), "pred_tokens", len(preds))

panels = []
missing = 0
for i in idxs:
    sample = dataset.samples[i]

    # GT：pkl 里已是 ego 米制
    gt = defaultdict(list)
    for lab, poly in zip(sample["gt_polyline_labels"], sample["gt_polylines"]):
        gt[int(lab)].append(np.asarray(poly, dtype=float))

    pred = preds.get(sample["token"])
    if pred is None:
        missing += 1
        pred = {}
    pb = defaultdict(list)
    for v, lab, sc in zip(pred.get("vectors", []), pred.get("labels", []), pred.get("scores", [])):
        if float(sc) >= SCORE_THR:
            pb[int(lab)].append(v)

    fi = int(sample.get("frame_idx", i))
    gt_p = out_dir / f"frame_{fi:04d}_gt.png"
    pr_p = out_dir / f"frame_{fi:04d}_pred.png"
    gp_p = out_dir / f"frame_{fi:04d}_gt_pred.png"
    draw(gt, gt_p, f"[0708] GT frame={fi}")
    draw(pb, pr_p, f"[0708] Pred score>={SCORE_THR} frame={fi}")

    g = np.asarray(Image.open(gt_p))
    p = np.asarray(Image.open(pr_p))
    h = max(g.shape[0], p.shape[0])

    def pad(im, h):
        if im.shape[0] == h:
            return im
        out = np.zeros((h, im.shape[1], 3), dtype=im.dtype)
        out[: im.shape[0]] = im
        return out

    panel = np.concatenate([pad(g, h), pad(p, h)], axis=1)
    Image.fromarray(panel).save(gp_p)
    panels.append(panel)

if panels:
    imageio.mimsave(out_dir / "gt_pred.mp4", panels, fps=2)
print("missing_pred_tokens", missing)
print("DONE", out_dir)
PY
```

---

## 4. 摄像头反投影（车道线预测画回 5 路原图）

把 `$SUB_JSON` 里该场景的预测 polyline（以及可选 GT）投影到 Waymo 5 相机原图上。  
依赖第 1 步生成的 `$SUB`（含 `cams[*].img_fpath` / `extrinsics` / `intrinsics`）。

外参约定：pinhole pack 后 `extrinsics` 为 **ego → cam**；地面假设 `z=0`。

```bash
# 每隔几帧画一张，避免单场景 40 帧过多；改成 1 即全画
FRAME_STRIDE=5
MAX_FRAMES=8
export FRAME_STRIDE MAX_FRAMES

python - <<'PY'
from pathlib import Path
from collections import defaultdict
import json, os
import numpy as np
import cv2
import av2.geometry.interpolate as interp_utils

SUB = Path(os.environ["SUB"])
SUB_JSON = os.environ["SUB_JSON"]
OUT = Path(os.environ["OUT"])
SCENE = os.environ["SCENE"]
SCORE_THR = float(os.environ["SCORE_THR"])
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "5"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "8"))

import pickle
with open(SUB / "waymo_map_infos_val.pkl", "rb") as f:
    samples = pickle.load(f)["samples"]
samples = [s for s in samples if s["scene_name"] == SCENE]
samples.sort(key=lambda s: int(s.get("frame_idx", 0)))
sel = samples[::FRAME_STRIDE][:MAX_FRAMES]
print("project frames:", len(sel))

with open(SUB_JSON) as f:
    preds = json.load(f)["results"]

ID2CAT = {0: "ped_crossing", 1: "divider", 2: "boundary"}
COLOR_BGR = {
    "divider": (0, 0, 255),
    "boundary": (0, 255, 0),
    "ped_crossing": (255, 0, 0),
}
CAM_ORDER = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]


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


def draw_polyline_ego_on_img(polyline_ego, img_bgr, ego2cam, intrinsics, color_bgr, thickness=3):
    if polyline_ego.shape[1] == 2:
        polyline_ego = np.concatenate(
            [polyline_ego, np.zeros((polyline_ego.shape[0], 1))], axis=1
        )
    if len(polyline_ego) < 2:
        return
    try:
        polyline_ego = interp_utils.interp_arc(t=500, points=polyline_ego.astype(np.float64))
    except Exception:
        pts = []
        for i in range(len(polyline_ego) - 1):
            pts.append(np.linspace(polyline_ego[i], polyline_ego[i + 1], 20))
        polyline_ego = np.concatenate(pts, axis=0)

    uv, depth = points_ego2img(polyline_ego, ego2cam, intrinsics)
    if len(uv) == 0:
        return
    h, w = img_bgr.shape[:2]
    valid = (
        (0 <= uv[:, 0]) & (uv[:, 0] < w - 1) &
        (0 <= uv[:, 1]) & (uv[:, 1] < h - 1) &
        (depth > 0)
    )
    if valid.sum() == 0:
        return
    uv = np.round(uv[valid]).astype(np.int32)
    for i in range(len(uv) - 1):
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


def render(sample, vectors, out_root, tag):
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
            print("missing image", c["img_fpath"])
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
            f"{tag} thr={SCORE_THR:.2f} frame={frame_idx}",
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
        panel_name = f"frame_{frame_idx:04d}_{tag}_panel.jpg"
        make_panel(
            rendered,
            cam_names,
            out_root / panel_name,
            f"[{tag}] {SCENE} frame={frame_idx} score>={SCORE_THR:.2f}",
        )


out_pred = OUT / f"cam_proj_pred_score_{SCORE_THR:.2f}"
out_gt = OUT / f"cam_proj_gt"
out_pred.mkdir(parents=True, exist_ok=True)
out_gt.mkdir(parents=True, exist_ok=True)

for sample in sel:
    token = sample["token"]
    render(sample, collect_gt(sample), out_gt, "GT")
    pred = preds.get(token)
    if pred is None:
        print("missing pred", token)
        continue
    render(sample, collect_pred(pred, SCORE_THR), out_pred, "PRED")

print("DONE pred ->", out_pred)
print("DONE gt   ->", out_gt)
PY
```

颜色约定（BGR）：行人蓝 / divider 红 / boundary 绿。

---

## 5. 输出与检查

**分数检查（第 2 步）**

```text
$OUT/check_vector_score_hist.png
```

**BEV（第 3 步）**

```text
$OUT/vis_bev_score_0.30/
  frame_XXXX_gt.png
  frame_XXXX_pred.png
  frame_XXXX_gt_pred.png
  gt_pred.mp4
```

**摄像头反投影（第 4 步）**

```text
$OUT/cam_proj_pred_score_0.30/
  frame_XXXX/{FRONT,FRONT_LEFT,...}.jpg
  frame_XXXX_PRED_panel.jpg

$OUT/cam_proj_gt/
  frame_XXXX/{FRONT,...}.jpg
  frame_XXXX_GT_panel.jpg
```

```bash
ls -lh "$OUT/check_vector_score_hist.png"
ls -lh "$OUT/vis_bev_score_${SCORE_THR}/" | head
ls -lh "$OUT/cam_proj_pred_score_${SCORE_THR}/" | head
ls "$OUT/cam_proj_pred_score_${SCORE_THR}/"*panel.jpg | head
```

---

## 6. 脚本与文件一览

| 用途 | 路径 |
|---|---|
| 训练日志 | `0708_pinhole_stage123.out` |
| 预测 JSON | `$WORK/eval_full_val/submission_vector.json` |
| 配置（加载 dataset） | `$CFG` |
| 旧 ROI 源 | `/data/waymo_processed_v3` |
| pack | `pack_waymo_for_maptracker.py` |
| 预测分数统计 | `tools/check_vector.py`（本流程用第 2 步等价脚本指向 0708 JSON） |
| BEV 可视化参考 | `vis_waymo_stage2.py` |
| 投影参考 | `plugin/datasets/visualize/renderer.py`（`draw_polyline_ego_on_img`） |
