# 单场景全局可视化（vis_global）

用已有 val 预测（`submission_vector*.json`）对**单个场景**做全局坐标系下的向量地图可视化。**不重新推理**。

流程：
1. 准备 / 切出单场景 submission JSON
2. 转成 `vis_global` 需要的 pickle（含 `global_ids` + `meta`）
3. 跑 `tools/visualization/vis_global.py`（`vis-pred`）

> 相关但不相同：逐帧 BEV / 摄像头反投影见 [`single_scene_lane_vis_0708_planA.md`](./single_scene_lane_vis_0708_planA.md)。  
> 本文只覆盖 **全局拼接**（merged / unmerged / per-frame / mp4）。


## 为什么不能直接喂 JSON？

`vis_global.py --option vis-pred` 需要的是 `pos_predictions.pkl` 风格的 **list[dict]**，每帧至少包含：

| 字段 | 作用 |
|---|---|
| `vectors` / `labels` / `scores` | 过滤后的预测实例 |
| `global_ids` | 跨帧同一实例的跟踪 ID（用于叠图 / merge） |
| `scene_name` / `local_idx` | 场景与帧序 |
| `meta.ego2global_translation` / `meta.ego2global_rotation` | 投到最后一帧坐标系 |

而 `submission_vector*.json` 是：

```text
{ "meta": ..., "results": { token: {vectors, scores, labels, ...}, ... } }
```

通常是每帧 100 个 query、无 `global_ids`、无 ego2global。因此必须先转换。


## 用到的路径（0708 pinhole 示例）

| 项 | 路径 |
|---|---|
| Stage3 work_dir | `/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole` |
| 全量 val 预测 | `.../eval_full_val/submission_vector.json` |
| 单场景 JSON（可选） | `.../eval_full_val/submission_vector_<SCENE>.json` |
| Stage3 配置 | `/data/maptr_workspace/work_dirs/generated_configs/xm15_x45_y15_pinhole/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole.py` |
| 转换脚本逻辑 | `tools/tracking/prepare_pred_tracks.py` |
| 可视化脚本 | `tools/visualization/vis_global.py` |

---

## 0. 环境变量

```bash
cd /root/maptracker
source /root/miniconda3/etc/profile.d/conda.sh
conda activate maptracker
export PYTHONPATH=/root/maptracker:${PYTHONPATH:-}

WORK=/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole
CFG=/data/maptr_workspace/work_dirs/generated_configs/xm15_x45_y15_pinhole/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_xm15_x45_y15_pinhole.py
EVAL=$WORK/eval_full_val

# 改成要看的 val 场景
SCENE=segment-10203656353524179475_7625_000_7645_000_with_camera_labels
SCORE_THR=0.30
CONS_FRAMES=5

SCENE_JSON=$EVAL/submission_vector_${SCENE}.json
SCENE_PKL=$EVAL/pos_predictions_${SCENE}.pkl
OUT=$EVAL/vis_global/${SCENE}

export WORK CFG EVAL SCENE SCORE_THR CONS_FRAMES SCENE_JSON SCENE_PKL OUT
```

---

## 1. 准备单场景 submission JSON

### 已有单场景文件

若已有 `$SCENE_JSON`，跳过本节。

### 从全量 JSON 切出

```bash
python - <<'PY'
import json, os
from pathlib import Path

scene = os.environ["SCENE"]
src = Path(os.environ["EVAL"]) / "submission_vector.json"
dst = Path(os.environ["SCENE_JSON"])

with open(src) as f:
    data = json.load(f)

results = data["results"]
kept = {tok: v for tok, v in results.items() if tok.startswith(scene + "_frame_")}
assert kept, f"no frames for {scene}"

out = {
    "meta": data.get("meta", {}),
    "results": kept,
    "scene_name": scene,
    "num_frames": len(kept),
}
dst.parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w") as f:
    json.dump(out, f)
print(f"wrote {dst} frames={len(kept)}")
PY
```

---

## 2. JSON → pickle（track matching）

复用 `prepare_pred_tracks.py` 里的匹配逻辑：按 `SCORE_THR` 过滤，再在连续帧间做 IoU matching，生成 `global_ids`，并从 dataset 取 `ego2global` meta。

```bash
python - <<'PY'
import os, sys, json, pickle
import numpy as np
import torch
from pathlib import Path
from mmcv import Config
from mmdet3d.datasets import build_dataset

sys.path.insert(0, "tools/tracking")
from prepare_pred_tracks import get_scene_matching_result, generate_results

SCENE = os.environ["SCENE"]
JSON_PATH = Path(os.environ["SCENE_JSON"])
CFG = os.environ["CFG"]
OUT_PKL = Path(os.environ["SCENE_PKL"])
THR = float(os.environ["SCORE_THR"])
CONS = int(os.environ["CONS_FRAMES"])

cfg = Config.fromfile(CFG)
if hasattr(cfg.match_config, "interval"):
    cfg.match_config.interval = 1

sys.path.append(os.path.abspath("."))
if getattr(cfg, "plugin", False):
    import importlib
    plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
    for plugin_dir in plugin_dirs:
        parts = os.path.dirname(plugin_dir).split("/")
        mod = parts[0]
        for p in parts[1:]:
            mod += "." + p
        importlib.import_module(mod)

dataset = build_dataset(cfg.match_config)
scene_name2idx = {}
for idx, sample in enumerate(dataset.samples):
    scene_name2idx.setdefault(sample["scene_name"], []).append(idx)
assert SCENE in scene_name2idx, f"{SCENE} not in dataset"

with open(JSON_PATH) as f:
    results = json.load(f)["results"]

class Args:
    thr = THR
    cons_frames = CONS

roi_size = torch.tensor(cfg.roi_size).numpy()
origin = torch.tensor(cfg.pc_range[:2]).numpy()

ids_info, vectors_seq, scores_seq, meta_list = get_scene_matching_result(
    Args(), cfg, results, dataset, origin, roi_size, scene_name2idx[SCENE]
)
gen_result = generate_results(ids_info, vectors_seq, scores_seq, meta_list, SCENE)

for item in gen_result:
    item["vectors"] = (
        np.array(item["vectors"], dtype=np.float64)
        if len(item["vectors"])
        else np.zeros((0, 40), dtype=np.float64)
    )
    item["labels"] = np.array(item["labels"], dtype=np.int64)
    item["scores"] = np.array(item["scores"], dtype=np.float32)
    item["global_ids"] = np.array(item["global_ids"], dtype=np.int64)
    if not isinstance(item["meta"], dict):
        item["meta"] = dict(item["meta"])

with open(OUT_PKL, "wb") as f:
    pickle.dump(gen_result, f, protocol=pickle.HIGHEST_PROTOCOL)

nvec = [len(x["vectors"]) for x in gen_result]
print("saved", OUT_PKL)
print("frames", len(gen_result), "vecs/frame min/med/max",
      min(nvec), sorted(nvec)[len(nvec) // 2], max(nvec))
PY
```

**说明：**
- `SCORE_THR=0.30` 与近期 val 可视化一致；需要更稀/更密可改 thr。
- 若已有 `eval_full_val/pos_predictions.pkl`（推理时写的、带模型 `global_ids`），也可直接从中筛该 scene，跳过本节匹配。模型 ID 通常比 IoU rematch 更准。

---

## 3. 跑 vis_global

```bash
mkdir -p "$OUT"

python tools/visualization/vis_global.py \
  "$CFG" \
  --data_path "$SCENE_PKL" \
  --out_dir "$OUT" \
  --option vis-pred \
  --per_frame_result 1 \
  --overwrite 1 \
  --draw_bev_range 1 \
  --scene_id "$SCENE"
```

可选参数：
- `--per_frame_result 0`：只要整段 merged/unmerged，不做 per-frame / mp4（更快）
- `--simplify` / `--line_opacity` / `--dpi`：线简化与画质

> `vis_global` 会在 `--out_dir` 下再建一层 `$SCENE/` 目录。若 `--out_dir` 已含场景名，会出现嵌套路径，属正常。

---

## 4. 输出说明

主目录（嵌套时）：

```text
$OUT/$SCENE/
  pred_comb.png              # merged | unmerged 并排（主结果）
  pred_merged.png            # 跨帧合并后的全局地图
  pred_unmerged.png          # 同 instance 多帧观测叠图（未合并）
  pred_merged_per_frame/     # 逐帧累积 merged + vis.mp4
  pred_unmerged_per_frame/   # 逐帧累积 unmerged + vis.mp4
```

坐标：各帧 ego 向量经 `ego2global` 统一投到**场景最后一帧**坐标系，并画出车辆轨迹。


## 5. 已跑通的例子

| 项 | 值 |
|---|---|
| Scene | `segment-10203656353524179475_7625_000_7645_000_with_camera_labels` |
| 输入 JSON | `$EVAL/submission_vector_<SCENE>.json`（40 frames） |
| 中间 pickle | `$EVAL/pos_predictions_<SCENE>.pkl`（thr=0.30，约 10–16 vecs/frame） |
| 输出 | `$EVAL/vis_global/<SCENE>/<SCENE>/pred_comb.png` |
| 日志 | `/root/maptracker/0713_vis_global_one_scene.out` |

---

## 可选：顺带画 GT

需要 GT 全局图时，用 val tracks：

```bash
GT_PKL=/data/waymo_maptracker_xm15_x45_y15_pinhole/waymo_map_infos_val_gt_tracks.pkl
GT_OUT=$EVAL/vis_global_gt/${SCENE}

python tools/visualization/vis_global.py \
  "$CFG" \
  --data_path "$GT_PKL" \
  --out_dir "$GT_OUT" \
  --option vis-gt \
  --per_frame_result 1 \
  --overwrite 1 \
  --draw_bev_range 1 \
  --scene_id "$SCENE"
```

注意：GT ROI 必须与训练/评测配置一致（本例为 pinhole asym ROI `x∈[-15,45], y∈[-15,15]`）。

---

## 常见问题

1. **`[vis-pred] skip scene ...: no predicted vectors`**  
   pickle 里该 scene 为空，或 thr 过高。检查转换步骤的 `vecs/frame`。

2. **缺少 `ego2global_*`**  
   转换时未从 dataset 取 meta。确认 `match_config.interval=1` 且 token 能对上。

3. **整段 val 太慢**  
   只传 `--scene_id`，且可先 `--per_frame_result 0`。

4. **与 `pos_predictions.pkl` 的差异**  
   - 推理写出的 pickle：模型内部 tracking ID  
   - 本文 JSON rematch：IoU + `SCORE_THR` 后处理 ID  
   两者分数过滤后的几何大体一致，ID 可能不同。
