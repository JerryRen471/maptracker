# MapTracker Model Code Structure

本文按 `README.md` 中的模型结构，梳理 MapTracker 各部分对应的代码、输入输出和主要数据结构。说明以当前代码中的典型配置 `plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py` 为参考。

## 1. 总体入口

`README.md` 中的整体模型 `MapTracker` 对应主代码：

- 主模型：`plugin/models/mapers/MapTracker.py`
- BEV 模块：`plugin/models/backbones/bevformer_backbone.py`
- VEC 模块：`plugin/models/heads/MapDetectorHead.py`
- BEV 语义分割头：`plugin/models/heads/MapSegHead.py`
- Vector memory：`plugin/models/mapers/vector_memory.py`
- Query propagation / PropMLP：`plugin/models/utils/query_update.py`
- Transformer decoder：`plugin/models/transformer_utils/MapTransformer.py`

主模型在 `MapTracker.__init__()` 中组装：

```python
self.backbone = build_backbone(backbone_cfg)
self.neck = build_head(neck_cfg) if neck_cfg is not None else nn.Identity()
self.head = build_head(head_cfg)
self.query_propagate = MotionMLP(...)
self.seg_decoder = build_head(seg_cfg)
self.memory_bank = VectorInstanceMemory(...)  # use_memory=True 时
```

训练入口是 `MapTracker.forward_train()`，推理入口是 `MapTracker.forward_test()`。

### 1.1 代码位置索引

| 结构/流程 | 代码位置 |
| --- | --- |
| 模型组装 | `plugin/models/mapers/MapTracker.py`：`MapTracker.__init__()` |
| 训练主流程 | `plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()` |
| 推理主流程 | `plugin/models/mapers/MapTracker.py`：`MapTracker.forward_test()` |
| GT 预处理 | `plugin/models/mapers/MapTracker.py`：`MapTracker.batch_data()` |
| BEV backbone | `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone` |
| 图像特征提取 | `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.extract_img_feat()` |
| BEV feature 生成和 BEV memory fusion | `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()` |
| 历史位姿和 BEV warp grid | `plugin/models/mapers/MapTracker.py`：`MapTracker.process_history_info()` |
| BEV semantic segmentation | `plugin/models/heads/MapSegHead.py`：`MapSegHead` |
| Vector detection head | `plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead` |
| Vector head 训练 | `plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_train()` |
| Vector head 推理 | `plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_test()` |
| Vector decoder transformer | `plugin/models/transformer_utils/MapTransformer.py`：`MapTransformer`、`MapTransformerDecoder_new`、`MapTransformerLayer` |
| Query propagation / PropMLP | `plugin/models/utils/query_update.py`：`MotionMLP` |
| 跨帧 query 传播 | `plugin/models/mapers/MapTracker.py`：`MapTracker.temporal_propagate()` |
| 两帧 GT 匹配 | `plugin/models/mapers/MapTracker.py`：`MapTracker.get_two_frame_matching()` |
| Track query 构造 | `plugin/models/mapers/MapTracker.py`：`MapTracker.prepare_track_queries_and_targets()` |
| Vector memory | `plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory` |
| Memory 写入 | `plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.update_memory()` |
| Memory 变换和选择 | `plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.trans_memory_bank()`、`VectorInstanceMemory.select_memory_entries()` |
| Vector memory fusion | `plugin/models/transformer_utils/MapTransformer.py`：`MapTransformerLayer.forward()` 中第二个 `cross_attn` |
| 推理结果后处理 | `plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.post_process()` |
| 推理时序缓存 | `plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.prepare_temporal_propagation()`、`MapDetectorHead.get_track_info()` |
| 主要模型配置 | `plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py`：`model` |

## 2. 输入数据结构

### 2.1 当前帧输入

代码位置：

- `plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()`
- `plugin/models/mapers/MapTracker.py`：`MapTracker.forward_test()`

`MapTracker.forward_train()` 的主要输入：

```python
img: Tensor[B, N, 3, H, W]
vectors: List[Dict[int, List[np.ndarray]]]
semantic_mask: Tensor[B, canvas_h, canvas_w]
img_metas: List[Dict]
all_prev_data: Optional[List[Dict]]
all_local2global_info: List[Dict]
```

含义：

- `B`：batch size。
- `N`：camera 数量。
- `vectors`：每个 sample 的矢量地图 GT，结构大致是 `vectors[b][label] = [line_0, line_1, ...]`。
- `line` 可能是 `[num_points, 2]`，也可能是 `[num_permute, num_points, 2]`。
- `img_metas[b]` 包含 `ego2global_translation`、`ego2global_rotation`、`token`、`local_idx`、camera calibration 等信息。

### 2.2 GT 预处理结构

代码位置：

- `plugin/models/mapers/MapTracker.py`：`MapTracker.batch_data()`
- 调用位置：`MapTracker.forward_train()` 中当前帧和 `all_prev_data` 历史帧的预处理

`batch_data()` 把 `vectors` 转成 `gts`：

```python
gts = {
    'labels': List[Tensor[num_gt]],
    'lines': List[Tensor[num_gt, 2 * num_points]],
    'gt2local': List[List[[label, local_instance_id]]],
    'local2gt': List[Dict[(label, local_instance_id), gt_idx]],
}
```

如果使用 permutation GT，`lines` 中的单个 sample 可能是 `[num_gt, num_perm, 2 * num_points]`。

随后代码会把 `gts` 复制成 decoder layer 数量：

```python
gts = [deepcopy(gts) for _ in range(self.num_decoder_layers)]
```

最终 `gts` 是长度为 `num_decoder_layers` 的 `List[Dict]`。

## 3. BEV Module

README 中的 **BEV Module** 接收车载多视角图像特征、BEV memory buffer 和车辆运动信息，输出当前帧 BEV feature。

对应代码：

- `plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()`
- `plugin/models/mapers/MapTracker.py`：`MapTracker.forward_test()`
- `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()`
- `plugin/models/mapers/MapTracker.py`：`MapTracker.process_history_info()`

### 3.1 图像特征提取

代码位置：

- `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.extract_img_feat()`
- 配置位置：`plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py`：`backbone_cfg.img_backbone`、`backbone_cfg.img_neck`

配置中 BEV backbone 包含 `ResNet + FPN + PerceptionTransformer`：

```python
backbone_cfg=dict(
    type='BEVFormerBackbone',
    img_backbone=dict(type='ResNet', depth=50, ...),
    img_neck=dict(type='FPN', ...),
    transformer=dict(type='PerceptionTransformer', ...)
)
```

输入：

```python
img: Tensor[B, N, 3, H, W]
```

输出：

```python
mlvl_feats: List[Tensor[B, N, C, H_i, W_i]]
```

这里是 `List[...]`，是因为 FPN 会输出多个尺度的 image feature；第 `i` 个尺度的特征记为 `Tensor[B, N, C, H_i, W_i]`。

这个形状来自 `BEVFormerBackbone.extract_img_feat()` 的处理流程：

1. 输入图像一开始是多相机 batch：

   ```python
   img: Tensor[B, N, 3, H, W]
   ```

2. 送入 2D image backbone 前，会把 batch 维和 camera 维合并：

   ```python
   img = img.reshape(B * N, 3, H, W)
   ```

   这样 ResNet/FPN 会把每个 camera view 当作一张普通 2D 图像处理。

3. ResNet + FPN 输出每个尺度的特征：

   ```python
   img_feat: Tensor[B * N, C, H_i, W_i]
   ```

   这里的 `ResNet + FPN` 结构来自配置里的 `img_backbone` 和 `img_neck`。

   `ResNet` 部分：

   ```python
   img_backbone=dict(
       type='ResNet',
       depth=50,
       num_stages=4,
       out_indices=(1, 2, 3),
       style='caffe',
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
       stage_with_dcn=(False, False, True, True),
   )
   ```

   也就是说，输入的每张 camera image 会先经过 ResNet-50。这里没有取 ResNet 的所有 stage，而是只取 `out_indices=(1, 2, 3)` 对应的三个高层 stage 输出。按照 MMDetection ResNet 的常见定义，这三个输出大致对应：

   ```python
   stage 1: Tensor[B * N, 512,  H/8,  W/8]
   stage 2: Tensor[B * N, 1024, H/16, W/16]
   stage 3: Tensor[B * N, 2048, H/32, W/32]
   ```

   其中后两个 stage 启用了 DCNv2，因为 `stage_with_dcn=(False, False, True, True)`。

   `FPN` 部分：

   ```python
   img_neck=dict(
       type='FPN',
       in_channels=[512, 1024, 2048],
       out_channels=bev_embed_dims,  # 这里 bev_embed_dims = 256
       start_level=0,
       add_extra_convs=True,
       num_outs=num_feat_levels,     # 这里 num_feat_levels = 3
   )
   ```

   FPN 接收 ResNet 的三个尺度输出，并通过 lateral convolution + top-down fusion，把每个尺度都统一成 `256` 通道。因此 FPN 输出是一个长度为 3 的 list：

   ```python
   [
       Tensor[B * N, 256, H_0, W_0],
       Tensor[B * N, 256, H_1, W_1],
       Tensor[B * N, 256, H_2, W_2],
   ]
   ```

   对当前配置来说，后续代码里的 `C` 就是 `bev_embed_dims = 256`。`H_i, W_i` 表示第 `i` 个 FPN level 的空间分辨率，通常相对输入图像分别接近 `1/8`、`1/16`、`1/32`。

4. 代码再把第一维拆回 batch 和 camera：

   ```python
   img_feat.view(B, N, C, H_i, W_i)
   ```

因此最终 `mlvl_feats` 的每个元素是：

```python
Tensor[B, N, C, H_i, W_i]
```

其中 `N` 保留了多相机维度，后续 BEVFormer 的 spatial cross-attention 需要利用每个 camera 的特征和相机几何关系，把 perspective image features 聚合到 BEV query 上。

### 3.2 BEV Query 与 BEVFormer

代码位置：

- BEV query 定义：`plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone._init_layers()`
- BEV feature 生成：`plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()`
- BEVFormer transformer 配置：`plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py`：`backbone_cfg.transformer`

`BEVFormerBackbone` 内部维护 BEV query：

```python
self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
```

典型配置：

```python
bev_h = 50
bev_w = 100
bev_embed_dims = 256
```

因此 BEV query 初始形状为：

```python
[bev_h * bev_w, 256] = [5000, 256]
```

经过 `PerceptionTransformer.get_bev_features()` 后输出：

```python
_bev_feats: Tensor[B, 256, 50, 100]
```

如果 `neck_cfg=None`，则 `self.neck = nn.Identity()`，最终：

```python
bev_feats: Tensor[B, 256, 50, 100]
```

### 3.3 BEV Memory Buffer

代码位置：

- 训练历史 BEV 缓存：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()`
- 推理历史 BEV 缓存：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_test()`
- 历史 BEV 使用：`plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()`

README 中的 BEV memory buffer 在代码里不是独立 class，而是用列表维护。

训练时：

```python
history_bev_feats = []
history_img_metas = []
```

推理时：

```python
self.history_bev_feats_all = []
self.history_img_metas_all = []
```

每个历史 BEV feature 的结构：

```python
bev_feats: Tensor[B, 256, 50, 100]
```

### 3.4 Vehicle Motion 与 BEV Warping

代码位置：

- 位姿矩阵和 warp grid 计算：`plugin/models/mapers/MapTracker.py`：`MapTracker.process_history_info()`
- BEV feature warp：`plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()`
- track query 几何变换：`plugin/models/mapers/MapTracker.py`：`MapTracker.temporal_propagate()`

`process_history_info()` 根据当前帧和历史帧的 ego pose 计算运动变换：

```python
all_history_curr2prev: List[Tensor[T, 4, 4]]
all_history_prev2curr: List[Tensor[T, 4, 4]]
all_history_coord: List[Tensor[T, bev_h, bev_w, 2]]
```

其中 `all_history_coord` 会传入 `BEVFormerBackbone.forward()`，用于：

```python
F.grid_sample(history_bev_feats_i, history_coord, ...)
```

即把历史 BEV feature warp 到当前坐标系。

### 3.5 BEV Module 输出

代码位置：

- `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()`
- `plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()` / `MapTracker.forward_test()` 中的 `self.backbone(...)` 和 `self.neck(...)`

`self.backbone(...)` 输出：

```python
_bev_feats: Tensor[B, 256, 50, 100]
mlvl_feats: List[Tensor[B, N, C, H_i, W_i]]
```

`self.neck(_bev_feats)` 输出：

```python
bev_feats: Tensor[B, 256, 50, 100]
```

## 4. BEV Semantic Segmentation Head

README 图中 BEV Module 的输出一部分用于 semantic segmentation。

对应代码：

- `plugin/models/heads/MapSegHead.py`：`MapSegHead`
- 模型组装：`plugin/models/mapers/MapTracker.py`：`MapTracker.__init__()` 中的 `self.seg_decoder = build_head(seg_cfg)`
- 训练调用：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()` 中的 `self.seg_decoder(...)`
- 推理调用：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_test()` 中的 `self.seg_decoder(...)`
- 配置位置：`plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py`：`seg_cfg`

配置：

```python
seg_cfg=dict(
    type='MapSegHead',
    num_classes=3,
    in_channels=256,
    embed_dims=256,
    bev_size=(100, 50),
    canvas_size=(200, 100),
)
```

训练输入：

```python
bev_features: Tensor[B, 256, 50, 100]
gts: Tensor[B, 100, 200]
```

训练输出：

```python
preds: Tensor[B, num_classes, 100, 200]
seg_feats: Tensor[B, 256, 50, 100]
seg_loss: Tensor
dice_loss: Tensor
```

推理输出：

```python
seg_preds: Tensor[B, num_classes, 100, 200]
seg_feats: Tensor[B, 256, 50, 100]
```

`seg_loss` 和 `seg_dice_loss` 会加入总 loss。推理时 `seg_preds` 会被转换成 `semantic_mask` 写入结果。

## 5. VEC Module

README 中的 **VEC Module** 负责传播上一帧 vector latent、融合 vector memory，并解码出道路元素。

对应代码：

- `plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead`
- `plugin/models/transformer_utils/MapTransformer.py`：`MapTransformer`
- `plugin/models/transformer_utils/MapTransformer.py`：`MapTransformerDecoder_new`
- `plugin/models/transformer_utils/MapTransformer.py`：`MapTransformerLayer`
- `plugin/models/mapers/MapTracker.py`：`MapTracker.temporal_propagate()`
- `plugin/models/utils/query_update.py`：`MotionMLP`
- `plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory`
- 配置位置：`plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py`：`head_cfg`

### 5.1 Vector Head 构成

代码位置：

- Head 初始化：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.__init__()`
- Query 和 reference point 初始化：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead._init_embedding()`
- 分类/回归分支初始化：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead._init_branch()`
- 模型组装：`plugin/models/mapers/MapTracker.py`：`MapTracker.__init__()` 中的 `self.head = build_head(head_cfg)`

`self.head = build_head(head_cfg)`，配置类型是：

```python
type='MapDetectorHead'
```

主要组件：

```python
self.query_embedding
self.reference_points_embed
self.input_proj
self.transformer
self.cls_branches
self.reg_branches
self.loss_cls
self.loss_reg
self.assigner
```

典型配置：

```python
num_queries = 100
num_points = 20
num_classes = 3
embed_dims = 512
num_decoder_layers = 6
```

### 5.2 VEC Module 输入

代码位置：

- 训练入口：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_train()`
- 推理入口：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_test()`
- 调用位置：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()` / `MapTracker.forward_test()`

输入给 `MapDetectorHead.forward_train()`：

```python
bev_features: Tensor[B, 256, 50, 100]
img_metas: List[Dict]
gts: List[Dict]
track_query_info: Optional[List[Dict]]
memory_bank: Optional[VectorInstanceMemory]
```

### 5.3 Query 数据结构

代码位置：

- 普通 detection query 构造：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_train()` / `MapDetectorHead.forward_test()`
- track query 拼接：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_train()` / `MapDetectorHead.forward_test()`
- track query 生成：`plugin/models/mapers/MapTracker.py`：`MapTracker.prepare_track_queries_and_targets()`
- 推理时 track query 读取：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.get_track_info()`

普通 detection query：

```python
query_embedding: Tensor[B, num_queries, 512]
init_reference_points: Tensor[B, num_queries, num_points, 2]
```

如果有上一帧 track query，会拼接：

```python
query_embedding =
    [track_query_hs_embeds, normal_queries, pad_hs_embeds]

init_reference_points =
    [trans_track_query_boxes, normal_reference_points, pad_query_boxes]
```

`track_query_info[b]` 的主要字段：

```python
{
    'track_query_hs_embeds': Tensor[num_tracks, 512],
    'track_query_boxes': Tensor[num_tracks, 2 * num_points],
    'trans_track_query_boxes': Tensor[num_tracks, 2 * num_points],
    'track_query_labels': Tensor[num_tracks],
    'track_query_scores': Tensor[num_tracks],
    'track_queries_mask': Tensor[num_tracks + num_queries],
    'track_queries_fal_pos_mask': Tensor[num_tracks + num_queries],
    'pad_hs_embeds': Tensor[pad_len, 512],
    'pad_query_boxes': Tensor[pad_len, 2 * num_points],
    'query_padding_mask': Tensor[max_query_len],
}
```

## 6. PropMLP / Query Propagation

代码位置：

- PropMLP 定义：`plugin/models/utils/query_update.py`：`MotionMLP`
- PropMLP 组装：`plugin/models/mapers/MapTracker.py`：`MapTracker.__init__()` 中的 `self.query_propagate`
- track query latent 传播：`plugin/models/mapers/MapTracker.py`：`MapTracker.temporal_propagate()`
- vector memory latent 传播：`plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.trans_memory_bank()`

README 中的 PropMLP 对应：

```python
self.query_propagate = MotionMLP(c_dim=7, f_dim=self.head.embed_dims, identity=True)
```

`MotionMLP` 输入：

```python
x: Tensor[num_tracks, 512]
pose_info: Tensor[num_tracks, 7]
```

其中 `pose_info` 是：

```python
[quat_x, quat_y, quat_z, quat_w, trans_x, trans_y, trans_z]
```

输出：

```python
updated_query: Tensor[num_tracks, 512]
```

在 `MapTracker.temporal_propagate()` 中，上一帧的 track query latent 会通过 PropMLP 更新：

```python
track_query_updated = self.query_propagate(prop_q, pose_info.repeat(len(prop_q), 1))
```

同时上一帧预测的 polyline 点会通过 pose matrix 变换到当前帧，写入：

```python
track_query_info[b_i]['trans_track_query_boxes']
```

## 7. Vector Memory Buffer

代码位置：

- Memory 定义：`plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory`
- Memory 初始化：`plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.init_memory()`
- Memory 写入：`plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.update_memory()`
- Memory 变换和当前帧准备：`plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.trans_memory_bank()`
- Memory entry 选择：`plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.select_memory_entries()`
- 训练/推理调用：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()` / `MapTracker.forward_test()`

README 中的 vector memory buffer 对应：

```python
VectorInstanceMemory
```

主要内部结构：

```python
mem_bank: Tensor[bank_size, B, max_number_ins, dim]
mem_bank_seq_id: Tensor[bank_size, B, max_number_ins]
mem_bank_trans: Tensor[bank_size, B, 3]
mem_bank_rot: Tensor[bank_size, B, 3, 3]
```

含义：

- `bank_size`：保存多少历史帧。
- `max_number_ins`：每个 batch 最多保存多少实例。
- `dim`：query embedding 维度，通常是 512。
- `mem_bank`：历史 vector latent。
- `mem_bank_seq_id`：每条 memory 对应的帧 id。
- `mem_bank_trans / mem_bank_rot`：每帧 ego pose。

更新入口：

```python
memory_bank.update_memory(...)
```

准备当前帧 memory fusion 输入：

```python
memory_bank.trans_memory_bank(self.query_propagate, b_i, img_metas[b_i])
```

`trans_memory_bank()` 会缓存：

```python
batch_mem_embeds_dict[b_i]: Tensor[mem_len, num_active_tracks, 512]
batch_mem_relative_pe_dict[b_i]: Tensor[mem_len, num_active_tracks, 512]
batch_key_padding_dict[b_i]: Tensor[num_active_tracks, mem_len]
valid_track_idx[b_i]: Tensor[num_valid_tracks]
```

这些会在 `MapTransformerLayer` 的 memory cross attention 中使用。

## 8. Vector Memory Fusion Layer

代码位置：

- Decoder layer：`plugin/models/transformer_utils/MapTransformer.py`：`MapTransformerLayer.forward()`
- Decoder sequence：`plugin/models/transformer_utils/MapTransformer.py`：`MapTransformerDecoder_new.forward()`
- Transformer wrapper：`plugin/models/transformer_utils/MapTransformer.py`：`MapTransformer.forward()`
- 配置位置：`plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py`：`head_cfg.transformer.decoder.transformerlayers`

README 中的 vector memory fusion 对应 `MapTransformerLayer.forward()` 中的第二个 `cross_attn`。

配置中的 decoder layer 顺序：

```python
('self_attn', 'norm',
 'cross_attn', 'norm',
 'cross_attn', 'norm',
 'ffn', 'norm')
```

含义：

1. 第一个 `self_attn`：query 之间交互。
2. 第一个 `cross_attn`：query 与 BEV feature 交互。
3. 第二个 `cross_attn`：query 与 vector memory 交互。
4. `ffn`：前馈网络更新 query。

memory fusion 关键逻辑：

```python
mem_embeds = memory_bank.batch_mem_embeds_dict[b_i][:, valid_track_idx, :]
mem_key_padding_mask = memory_bank.batch_key_padding_dict[b_i][valid_track_idx]
mem_key_pos = memory_bank.batch_mem_relative_pe_dict[b_i][:, valid_track_idx]

query_i[:, valid_track_idx] = self.attentions[attn_index](
    query_i[:, valid_track_idx],
    mem_embeds,
    mem_embeds,
    key_pos=mem_key_pos,
    key_padding_mask=mem_key_padding_mask,
)
```

只有有效 track query 会 attend 自己对应的历史 vector memory。

## 9. Vector Decoding 输出

代码位置：

- decoder 输出整理：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_train()` / `MapDetectorHead.forward_test()`
- loss 和 matching：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.loss()`、`MapDetectorHead.get_targets()`、`MapDetectorHead._get_target_single()`
- 推理后处理：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.post_process()`
- 时序传播结果缓存：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.prepare_temporal_propagation()`

`MapDetectorHead` 每个 decoder layer 输出：

```python
pred_dict = {
    'lines': List[Tensor[num_queries_total, 2 * num_points]],
    'scores': List[Tensor[num_queries_total, num_classes]],
    'hs_embeds': Tensor[B, num_queries_total, 512],
}
```

其中：

- `lines`：归一化 polyline 坐标，范围大致是 `[0, 1]`。
- `scores`：每个 query 的类别 logits。
- `hs_embeds`：decoder latent，用于下一帧 tracking / memory。

训练时还会输出：

```python
loss_dict
det_match_idxs
det_match_gt_idxs
gt_list
matched_reg_cost
```

推理最终 `post_process()` 输出：

```python
results_list = [
    {
        'vectors': np.ndarray[num_preds, num_points, 2],
        'scores': np.ndarray[num_preds],
        'labels': np.ndarray[num_preds],
        'props': np.ndarray[num_preds],
        'token': str,
        'track_scores': ...,
        'track_vectors': ...,
        'track_labels': ...,
        'semantic_mask': np.ndarray,
        'pos_results': Dict,
        'meta': Dict,
    }
]
```

## 10. 训练时序流程

代码位置：

- 主流程：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_train()`
- 当前帧/历史帧 GT 预处理：`plugin/models/mapers/MapTracker.py`：`MapTracker.batch_data()`
- BEV 生成：`plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()`
- Seg loss：`plugin/models/heads/MapSegHead.py`：`MapSegHead.forward_train()`
- Vector loss：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_train()`、`MapDetectorHead.loss()`
- 两帧 GT 匹配：`plugin/models/mapers/MapTracker.py`：`MapTracker.get_two_frame_matching()`
- 下一帧 track query 构造：`plugin/models/mapers/MapTracker.py`：`MapTracker.prepare_track_queries_and_targets()`
- vector memory 更新：`plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory.update_memory()`

`forward_train()` 按时间帧展开：

1. 当前帧和历史帧 GT 通过 `batch_data()` 整理成 `gts`。
2. 如果启用 memory，调用 `memory_bank.init_memory(bs)`。
3. 对每个 previous frame：
   - `process_history_info()` 计算历史帧到当前帧的 pose/grid。
   - `backbone()` 得到 BEV feature。
   - `seg_decoder()` 计算 BEV segmentation loss。
   - `head()` 预测 vector map，并计算 detection loss。
   - `get_two_frame_matching()` 根据 GT global id 找跨帧对应关系。
   - `prepare_track_queries_and_targets()` 生成下一帧要用的 track query。
   - `memory_bank.update_memory()` 写入 vector memory。
4. 对当前帧重复 BEV、segmentation 和 vector head 流程。
5. 汇总 vector detection loss、segmentation loss、dice loss 和 transformation loss。

最终返回：

```python
loss, log_vars, num_sample
```

## 11. 推理时序流程

代码位置：

- 主流程：`plugin/models/mapers/MapTracker.py`：`MapTracker.forward_test()`
- 历史 BEV 选择：`plugin/models/mapers/MapTracker.py`：`MapTracker.select_memory_entries()`
- 上一帧 track query 读取：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.get_track_info()`
- track query 传播：`plugin/models/mapers/MapTracker.py`：`MapTracker.temporal_propagate()`
- vector head 推理：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.forward_test()`
- 下一帧传播缓存：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.prepare_temporal_propagation()`
- 结果后处理：`plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead.post_process()`

`forward_test()` 当前只支持：

```python
B = 1
```

每帧流程：

1. 如果是 scene 第一帧，初始化 vector memory，并清空 BEV history。
2. 从历史 BEV 中按距离选择 memory entries。
3. `process_history_info()` 计算历史 BEV warp grid。
4. `backbone()` 生成当前 BEV feature。
5. 如果是第一帧，直接跑 `self.head(...)`。
6. 如果不是第一帧：
   - `self.head.get_track_info()` 取上一帧保留的 vector query。
   - `temporal_propagate()` 用 pose 和 PropMLP 更新 track query。
   - `self.head(..., track_query_info, memory_bank)` 做 vector decoding 和 memory fusion。
7. `prepare_temporal_propagation()` 保存下一帧需要的 positive vectors、query embeddings、scores、labels、global ids 和 vector memory。
8. `post_process()` 输出最终矢量结果。

## 12. README 结构与代码对应表

| README 模块 | 代码位置 | 输入 | 输出 |
| --- | --- | --- | --- |
| BEV Module | `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone` | `img`, `img_metas`, `history_bev_feats`, `all_history_coord` | `_bev_feats: Tensor[B, 256, 50, 100]` |
| BEV Memory Buffer | `plugin/models/mapers/MapTracker.py`：`history_bev_feats`、`history_img_metas` | 历史 BEV feature + pose | warped history BEV |
| Vehicle Motion | `plugin/models/mapers/MapTracker.py`：`process_history_info()` | 当前/历史 `ego2global` pose | `curr2prev`, `prev2curr`, `history_coord` |
| BEV Memory Fusion | `plugin/models/backbones/bevformer_backbone.py`：`BEVFormerBackbone.forward()` + `PerceptionTransformer` | 当前图像特征 + warped BEV memory | 当前 BEV feature |
| Semantic Segmentation | `plugin/models/heads/MapSegHead.py`：`MapSegHead` | `bev_feats`, `semantic_mask` | `seg_preds`, `seg_loss`, `dice_loss` |
| VEC Module | `plugin/models/heads/MapDetectorHead.py`：`MapDetectorHead` | `bev_feats`, `gts`, `track_query_info`, `memory_bank` | `lines`, `scores`, `hs_embeds`, losses |
| PropMLP | `plugin/models/utils/query_update.py`：`MotionMLP` | track query + relative pose | propagated query embedding |
| Vector Memory Buffer | `plugin/models/mapers/vector_memory.py`：`VectorInstanceMemory` | 历史 query embeddings + pose | per-instance memory tensors |
| Vector Memory Fusion | `plugin/models/transformer_utils/MapTransformer.py`：`MapTransformerLayer.forward()` 第二个 `cross_attn` | current track query + vector memory | fused query |
| Vector Decoder | `plugin/models/transformer_utils/MapTransformer.py`：`MapTransformerDecoder_new` + `plugin/models/heads/MapDetectorHead.py`：`cls_branches` / `reg_branches` | fused query | polyline coordinates + class logits |

## 13. 总结

MapTracker 的代码实现可以概括为：

```text
multi-view images
  -> ResNet + FPN
  -> BEVFormerBackbone with BEV memory fusion
  -> BEV feature
  -> MapSegHead for BEV segmentation
  -> MapDetectorHead for vector map decoding
  -> PropMLP + VectorInstanceMemory for temporal tracking and memory fusion
  -> vector HD map results
```

BEV memory 主要在 `BEVFormerBackbone` 中通过历史 BEV warp 融合；vector memory 主要在 `MapTransformerLayer` 的第二个 cross-attention 中按实例融合。
