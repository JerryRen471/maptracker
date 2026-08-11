# Waymo Mask 与相机投影修复总结

本文总结 `feature/mask-train` 分支上，将 MapTracker 适配到 Waymo 时的主要改动：非对称 ROI、BEV visibility mask（含时序扩展），以及最关键的 **Waymo 相机坐标系 → pinhole 投影修复**。

相关实验图位于 [`fig/waymo_summary/`](fig/waymo_summary/)。

---

## 1. 背景与目标

原始 MapTracker 面向 nuScenes（环视相机 + 对称 BEV 范围）。迁移到 Waymo 5 相机时遇到两类问题：

1. **感知范围**：前方可见区域远大于后方，对称 `±30m` 浪费大量相机看不到的后方面积。
2. **可见性监督**：BEV 语义分割若在整块 ROI 上算 loss，会强迫模型在相机看不到的格子上也拟合 map GT。
3. **外参约定不一致（最严重）**：Waymo 相机坐标系与 MapTracker/BEVFormer 期望的 pinhole 轴不一致，导致 `ego2img` 投影错误，进而让 visibility mask、BEV 特征对齐全部失效。

---

## 2. 非对称 ROI

车辆坐标系：`x` 向前，`y` 向左。训练 ROI 改为：

```text
x ∈ [-15, 45] m   （后方 15m，前方 45m）
y ∈ [-15, 15] m
```

面积仍约 1800 m²，但把更多格子分配给前方相机可见区域。

实现要点：

- `plugin/roi.py`：统一 `roi_range` / `roi_size` / `pc_range` 换算
- `tools/data_converter/waymo_map_converter.py`：按非对称范围裁剪 map 标注
- 三阶段 config 中 `roi_range = (-15, -15, 45, 15)`，`roi_size = (60, 30)`

GT track 可视化（标注 `ROI x=[-15, 45] y=[-15, 15]`）：

![非对称 ROI 的 GT tracks](fig/waymo_summary/01_asym_roi_gt_tracks.png)

---

## 3. BEV Visibility Mask

### 3.1 怎么加的

在 `MapTracker.get_bev_visibility_mask` 中，对 BEV 地面网格（`z=0`）用每帧的 `ego2img` 做投影：

1. 取 BEV plane 上每个格子的齐次坐标
2. `proj = ego2img @ plane`，深度取第三维 `depth = proj[..., 2]`
3. `u = proj_x / depth`，`v = proj_y / depth`
4. 若 `depth > 0` 且 `(u, v)` 落在对应相机图像范围内，则该相机可见
5. **任意一台相机可见** → 该 BEV 格子为 visible

训练时：

- `MapSegHead` 用 `visible_mask` 做 **masked CE / masked Dice**，不可见格子不计入分割损失
- 可见区域会与 GT 前景做 `maximum`，避免漏掉标注前景
- 时序扩展（`plugin/temporal_visibility.py`）：
  - 把历史帧 visibility warp 到当前帧
  - `supervision_mask = max(current_visible, warped_history_union)`
  - 历史 BEV 特征按 warped visibility gating，避免用看不见的历史特征

### 3.2 Mask 长什么样

可视化约定（`tools/check_seg.py`）：**白色 = visible，黑色 = invisible**。

修复外参后的 mask（已放大便于查看）：五路相机 FOV 并集在 BEV 上形成一块连续的白色可见区；车后/侧后相机覆盖不到的区域为黑色。

![修复后 visibility mask（白=可见）](fig/waymo_summary/02_visibility_mask_large.png)

对应帧的 GT 语义图与预测热力图：

![GT semantic mask](fig/waymo_summary/06_gt_semantic_mask_large.png)

![Seg GT / Pred / score 拼图](fig/waymo_summary/03_seg_gt_pred_side_by_side.png)

拼图从左到右通常为：GT 语义、硬阈值预测、visibility mask、各类别 score 热力图。可见预测主要落在 mask 覆盖区域内。

### 3.3 代码：加 mask

分三层：生成 → 接入 loss → 时序扩展。

#### （1）生成 BEV visibility mask

`MapTracker.get_bev_visibility_mask`：地面网格用 `ego2img` 投影，任一台相机可见即为 1。

```python
# plugin/models/mapers/MapTracker.py
def get_bev_visibility_mask(self, img_metas, device):
    """Return BEV cells visible in at least one camera.

    The mask is defined on the model BEV grid and uses ground-plane
    points because the semantic map labels are rasterized on z=0.
    """
    masks = []
    plane = self.plane.to(device=device, dtype=torch.float32)
    eps = 1e-5

    for img_meta in img_metas:
        ego2img = torch.as_tensor(
            np.asarray(img_meta['ego2img']),
            dtype=torch.float32,
            device=device)
        proj = torch.einsum('nij,hwj->nhwi', ego2img, plane)
        depth = proj[..., 2]
        denom = torch.maximum(depth, torch.ones_like(depth) * eps)
        u = proj[..., 0] / denom
        v = proj[..., 1] / denom

        img_shapes = img_meta['img_shape']
        cam_masks = []
        for cam_idx in range(ego2img.shape[0]):
            shape = img_shapes[cam_idx] if isinstance(img_shapes, list) else img_shapes
            img_h = float(shape[0])
            img_w = float(shape[1])
            cam_visible = (
                (depth[cam_idx] > eps)
                & (u[cam_idx] > 0.0)
                & (u[cam_idx] < img_w)
                & (v[cam_idx] > 0.0)
                & (v[cam_idx] < img_h)
            )
            cam_masks.append(cam_visible)
        masks.append(torch.stack(cam_masks, dim=0).any(dim=0))

    return torch.stack(masks, dim=0).to(dtype=torch.float32)
```

#### （2）训练时接入 seg loss

`forward_train` 里算 mask，再和历史可见并集组成 `supervision_mask`，传给 `seg_decoder`：

```python
# plugin/models/mapers/MapTracker.py  (forward_train 片段)
visible_mask_prev = self.get_bev_visibility_mask(
    img_metas_prev, bev_feats.device)
supervision_mask_prev = build_temporal_supervision_mask(
    visible_mask_prev, warped_history_visible_union)
seg_preds, seg_feats, seg_loss, seg_dice_loss = self.seg_decoder(
    bev_feats, gts_semantic_prev,
    all_history_coord, visible_mask=supervision_mask_prev,
    return_loss=True)
```

`MapSegHead`：用 mask 做加权 CE + masked Dice；并与 GT 前景做 `maximum`，避免漏掉标注：

```python
# plugin/models/heads/MapSegHead.py
def _prepare_visible_mask(self, visible_mask, preds, gts):
    if visible_mask is None:
        return None, None
    if visible_mask.dim() == 3:
        visible_mask = visible_mask[:, None]
    ...
    visible_mask = (visible_mask > 0).to(dtype=preds.dtype, device=preds.device)
    gt_mask = (gts.sum(dim=1, keepdim=True) > 0).to(dtype=preds.dtype)
    visible_mask = torch.maximum(visible_mask, gt_mask)  # 可见 ∪ GT 前景
    pixel_weight = visible_mask[:, 0]
    avg_factor = pixel_weight.sum().clamp_min(1.0)
    return visible_mask, (pixel_weight, avg_factor)

def _masked_dice_loss(self, preds, gts, visible_mask):
    pred = preds.sigmoid() * visible_mask
    target = gts * visible_mask
    ...

def forward_train(self, bev_features, gts, history_coords, visible_mask=None):
    ...
    visible_mask, seg_weight = self._prepare_visible_mask(visible_mask, preds, gts)
    if visible_mask is None:
        seg_loss = self.loss_seg(preds, gts)
        dice_loss = self.loss_dice(preds, gts)
    else:
        pixel_weight, avg_factor = seg_weight
        seg_loss = self.loss_seg(
            preds, gts, weight=pixel_weight, avg_factor=avg_factor)
        dice_loss = self._masked_dice_loss(preds, gts, visible_mask)
```

#### （3）时序 mask

新建 `plugin/temporal_visibility.py`：

| 函数 | 作用 |
|------|------|
| `warp_history_visibility_masks` | 用与历史 BEV 相同的 grid，把历史 mask warp 到当前帧 |
| `build_temporal_supervision_mask` | `max(current_visible, warped_history_union)` |
| `gate_history_bev_features` | 不可见历史 BEV 特征置零 |

```python
# plugin/temporal_visibility.py
def build_temporal_supervision_mask(current_visible_mask,
                                    warped_history_visible_union):
    """Return current-or-history visibility for segmentation supervision."""
    current_visible_mask = _as_bhw(current_visible_mask)
    if warped_history_visible_union is None:
        return current_visible_mask
    warped_history_visible_union = _as_bhw(warped_history_visible_union)
    warped_history_visible_union = warped_history_visible_union.to(
        device=current_visible_mask.device, dtype=current_visible_mask.dtype)
    return torch.maximum(current_visible_mask, warped_history_visible_union)
```

---

## 4. 最严重的问题：相机投影错误与修复

### 4.1 根因

Waymo 相机传感器坐标：

```text
x = forward,  y = left,  z = up
```

MapTracker / BEVFormer 的 pinhole 投影约定：

```text
x = right,  y = down,  z = forward（深度轴）
```

原先 `pack_waymo_for_maptracker.py` 只做了 `cam2ego → ego2cam` 求逆，**没有做轴变换**。于是：

- `ego2img` 的「深度」不是前方距离
- 图像平面 `(u, v)` 错位
- `get_bev_visibility_mask` 基于错误投影，几乎判不出可见格子
- BEVFormer 的相机–BEV 注意力同样错位

### 4.2 代码前后对比（`pack_waymo_for_maptracker.py`）

改动集中在 `cam2ego_to_ego2cam`：原先只做求逆，修复后求逆再乘轴变换。

#### 修复前：只做 `cam2ego → ego2cam` 求逆

```python
def cam2ego_to_ego2cam(rotation_3x3, translation_3):
    """Invert a cam->ego transform to get ego->cam."""
    R = np.asarray(rotation_3x3, dtype=np.float64)
    t = np.asarray(translation_3, dtype=np.float64).reshape(3)

    cam2ego = np.eye(4, dtype=np.float64)
    cam2ego[:3, :3] = R
    cam2ego[:3, 3] = t

    return np.linalg.inv(cam2ego)  # 仍是 Waymo 相机轴，不是 pinhole
```

#### 修复后：求逆后再乘 `WAYMO_CAM_TO_PINHOLE`

```python
# Waymo camera: x-forward, y-left, z-up
# Pinhole (MapTracker/BEVFormer): x-right, y-down, z-forward
WAYMO_CAM_TO_PINHOLE = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [1.0,  0.0,  0.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float64)


def cam2ego_to_ego2cam(rotation_3x3, translation_3):
    """Invert a Waymo cam->ego transform to get ego->pinhole-cam."""
    R = np.asarray(rotation_3x3, dtype=np.float64)
    t = np.asarray(translation_3, dtype=np.float64).reshape(3)

    cam2ego = np.eye(4, dtype=np.float64)
    cam2ego[:3, :3] = R
    cam2ego[:3, 3] = t

    ego2waymo_cam = np.linalg.inv(cam2ego)
    return WAYMO_CAM_TO_PINHOLE @ ego2waymo_cam  # ego → pinhole cam
```

**唯一实质改动**：`return inv(cam2ego)` → `return WAYMO_CAM_TO_PINHOLE @ inv(cam2ego)`。  
这样写入 pkl 的 `extrinsics` / 下游 `ego2img` 第三维才是深度，visibility mask 与 BEV 投影才能算对。

对应提交：`90128c1 Fix Waymo camera transformation and add unit tests`。  
修复后 pack 输出：`/data/waymo_maptracker_xm15_x45_y15_pinhole`；流水线：`tools/waymo_pipeline_xm15_x45_y15_pinhole.yml`。

### 4.3 修复前后：visibility mask 对比

同一场景帧上的 mask：

| | 修复前（错误外参 pack） | 修复后（pinhole pack） |
|--|--|--|
| 形态 | 几乎全黑，仅残留边界线 | 清晰的多相机 FOV 并集 |
| 含义 | 投影失效，几乎没有格子被判为可见 | 白色区域与真实相机覆盖一致 |

![修复前 visibility mask](fig/waymo_summary/02b_visibility_mask_pre_pinhole_pack_large.png)

![修复后 visibility mask](fig/waymo_summary/02_visibility_mask_large.png)

### 4.4 修复前后：BEV 热力图对比

同一场景（`segment-100239...`）后几帧（`frame_0035` / `0040` / `0045`）上，Stage1 分割 head 输出的 score 热力图对比。  
左：错误外参 pack（`waymo_stage1_seg`）；右：pinhole pack 修复后（`vis_iter11638`）。每张图从上到下为三帧。

| | 修复前 | 修复后 |
|--|--|--|
| 形态 | 响应呈横向条带，铺满整块 ROI，几何不可解释 | 高响应集中在可见 FOV 内，与车道结构对齐 |
| 与 mask 关系 | mask 近乎全黑，热力图却在全图乱响应 | 热力图激活区域与白色可见区一致 |

**divider score（frame 35 / 40 / 45）：**

![divider 热力图对比](fig/waymo_summary/10_bev_heatmap_compare_divider.png)

**boundary score（frame 35 / 40 / 45）：**

![boundary 热力图对比](fig/waymo_summary/10_bev_heatmap_compare_boundary.png)

**ped_crossing score（frame 35 / 40 / 45）：**

![ped_crossing 热力图对比](fig/waymo_summary/10_bev_heatmap_compare_ped_crossing.png)

完整面板（GT / hard pred / visibility mask / 三类 score；每帧 Before/After 一组）：

![Seg 面板整体对比](fig/waymo_summary/11_bev_seg_panel_compare.png)

### 4.5 相机投影叠加示意

把 map 矢量投影到前视图像（divider=红，ped_crossing=蓝，boundary=绿）。外参正确时，车道线应贴合路面透视；外参错误时会明显错位或落到路面外。

![前视相机投影叠加](fig/waymo_summary/04_camera_projection_front.jpg)

可用 `tools/check_waymo_camera_projection.py` 对 packed pkl 做投影统计校验。修复是否生效，最直观的证据是 **visibility mask 从近乎全黑变为合理 FOV**，以及上一节 **BEV 热力图从全图条带噪声变为与可见区对齐的结构化响应**。

---

## 5. 修复后的训练与可视化结果

在 pinhole 数据上重新跑三阶段训练（`exp_tag=xm15_x45_y15_pinhole`）：

| 阶段 | 作用 | 代表指标（日志摘录） |
|------|------|----------------------|
| Stage1 BEV pretrain | 语义分割 + BEV 表征 | val mIoU 约从 0.51 → **0.62** |
| Stage2 warmup | 接入向量头 | — |
| Stage3 joint finetune | 端到端微调 | val **mAP_normal ≈ 0.36–0.37** |

Stage3 单帧 vector GT vs Pred（`score_thr=0.30`，ROI 已是 `[-15,45]×[-15,15]`）：

![Vector GT / Pred](fig/waymo_summary/05_vector_gt_pred_score030.png)

更多帧的 contact sheet：

![Vector contact sheet](fig/waymo_summary/07_vector_contact_sheet.png)

---

## 6. 相关文件与路径速查

### 代码

| 模块 | 路径 |
|------|------|
| 非对称 ROI | `plugin/roi.py`，`docs/waymo_asymmetric_roi.md` |
| Visibility mask | `plugin/models/mapers/MapTracker.py` → `get_bev_visibility_mask` |
| Masked seg loss | `plugin/models/heads/MapSegHead.py` |
| 时序 visibility | `plugin/temporal_visibility.py` |
| 外参 / pinhole 变换 | `pack_waymo_for_maptracker.py` → `WAYMO_CAM_TO_PINHOLE` |
| 投影检查 | `tools/check_waymo_camera_projection.py` |
| Mask / seg 可视化 | `tools/check_seg.py`（`save_visibility_mask`） |
| Pipeline | `tools/waymo_pipeline.py`，`tools/waymo_pipeline_xm15_x45_y15_pinhole.yml` |

### 数据与实验

| 内容 | 路径 |
|------|------|
| 修复后 pack 数据 | `/data/waymo_maptracker_xm15_x45_y15_pinhole` |
| Stage1–3 work dirs | `/data/maptr_workspace/work_dirs/maptracker_waymo_5cam_*_xm15_x45_y15_pinhole/` |
| Stage1 seg 可视化 | `.../stage1_..._pinhole/vis_iter11638/semantic/` |
| Stage3 val vector 可视化 | `.../stage3_..._pinhole/eval_full_val/vis_vector/score_thr_0.30/` |
| 训练日志 | `0708_pinhole_stage123.out` 等 |

---

## 7. 一句话回顾

> 先用非对称 ROI 把监督范围对齐前方视野，再用 BEV visibility mask（及时序扩展）只在相机看得到的格子上做分割监督；最后修正 Waymo→pinhole 外参，使投影、mask 与 BEV 注意力共用正确的几何——这是整个适配链路里影响最大的一次修复。
