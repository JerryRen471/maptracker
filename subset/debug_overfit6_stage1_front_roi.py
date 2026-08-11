_base_ = [
    "../plugin/configs/maptracker/waymo_5cam/"
    "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"
]

roi_range = (0, -15, 45, 15)
roi_size = (45, 30)
pc_range = [roi_range[0], roi_range[1], -3, roi_range[2], roi_range[3], 5]

overfit_data_root = "/data/waymo_maptracker_x0_x45_y15_overfit6"

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_h = 608
img_w = 608
img_size = (img_h, img_w)

cat2id = {
    "ped_crossing": 0,
    "divider": 1,
    "boundary": 2,
}

coords_dim = 2
num_points = 20
permute = True
canvas_size = (200, 100)
thickness = 3
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format="vector")

num_gpus = 1
batch_size = 1
num_iters_per_epoch = 100
num_epochs = 20
total_iters = num_epochs * num_iters_per_epoch

work_dir = (
    "/data/maptr_workspace/work_dirs/"
    "debug_overfit6_stage1_front_roi")

model = dict(
    roi_size=roi_size,
    roi_range=roi_range,
    backbone_cfg=dict(
        roi_size=roi_size,
        roi_range=roi_range,
        transformer=dict(
            encoder=dict(
                pc_range=pc_range,
            ),
        ),
    ),
    head_cfg=dict(
        roi_size=roi_size,
        roi_range=roi_range,
    ),
)

train_pipeline = [
    dict(
        type="VectorizeMap",
        coords_dim=coords_dim,
        roi_size=roi_size,
        roi_range=roi_range,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(
        type="RasterizeMap",
        roi_size=roi_size,
        roi_range=roi_range,
        coords_dim=coords_dim,
        canvas_size=canvas_size,
        thickness=thickness,
        semantic_mask=True,
    ),
    dict(type="LoadMultiViewImagesFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="ResizeMultiViewImages",
        size=img_size,
        change_intrinsics=True,
    ),
    dict(type="Normalize3D", **img_norm_cfg),
    dict(type="PadMultiViewImages", size_divisor=32),
    dict(type="FormatBundleMap"),
    dict(
        type="Collect3D",
        keys=["img", "vectors", "semantic_mask"],
        meta_keys=(
            "token", "ego2img", "sample_idx", "ego2global_translation",
            "ego2global_rotation", "img_shape", "scene_name")),
]

test_pipeline = [
    dict(type="LoadMultiViewImagesFromFiles", to_float32=True),
    dict(
        type="ResizeMultiViewImages",
        size=img_size,
        change_intrinsics=True,
    ),
    dict(type="Normalize3D", **img_norm_cfg),
    dict(type="PadMultiViewImages", size_divisor=32),
    dict(type="FormatBundleMap"),
    dict(
        type="Collect3D",
        keys=["img"],
        meta_keys=(
            "token", "ego2img", "sample_idx", "ego2global_translation",
            "ego2global_rotation", "img_shape", "scene_name")),
]

eval_config = dict(
    type="WaymoMapDataset",
    ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl",
    meta=meta,
    roi_size=roi_size,
    roi_range=roi_range,
    cat2id=cat2id,
    pipeline=[
        dict(
            type="VectorizeMap",
            coords_dim=coords_dim,
            simplify=True,
            normalize=False,
            roi_size=roi_size,
            roi_range=roi_range,
        ),
        dict(
            type="RasterizeMap",
            roi_size=roi_size,
            roi_range=roi_range,
            coords_dim=coords_dim,
            canvas_size=canvas_size,
            thickness=thickness,
            semantic_mask=True,
        ),
        dict(type="FormatBundleMap"),
        dict(
            type="Collect3D",
            keys=["vectors", "semantic_mask"],
            meta_keys=[
                "token", "ego2img", "sample_idx", "ego2global_translation",
                "ego2global_rotation", "img_shape", "scene_name"]),
    ],
    interval=1,
)

match_config = dict(
    type="WaymoMapDataset",
    ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl",
    meta=meta,
    roi_size=roi_size,
    roi_range=roi_range,
    cat2id=cat2id,
    pipeline=[
        dict(
            type="VectorizeMap",
            coords_dim=coords_dim,
            simplify=False,
            normalize=True,
            roi_size=roi_size,
            roi_range=roi_range,
            sample_num=num_points,
        ),
        dict(
            type="RasterizeMap",
            roi_size=roi_size,
            roi_range=roi_range,
            coords_dim=coords_dim,
            canvas_size=canvas_size,
            thickness=thickness,
        ),
        dict(type="FormatBundleMap"),
        dict(
            type="Collect3D",
            keys=["vectors", "semantic_mask"],
            meta_keys=[
                "token", "ego2img", "sample_idx", "ego2global_translation",
                "ego2global_rotation", "img_shape", "scene_name"]),
    ],
    interval=1,
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type="WaymoMapDataset",
        ann_file=f"{overfit_data_root}/waymo_map_infos_train.pkl",
        meta=meta,
        roi_size=roi_size,
        roi_range=roi_range,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-2,
        matching=True,
        multi_frame=5,
        interval=1,
    ),
    val=dict(
        type="WaymoMapDataset",
        ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl",
        meta=meta,
        roi_size=roi_size,
        roi_range=roi_range,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        interval=1,
        eval_semantic=True,
    ),
    test=dict(
        type="WaymoMapDataset",
        ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl",
        meta=meta,
        roi_size=roi_size,
        roi_range=roi_range,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        interval=1,
        eval_semantic=True,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

optimizer = dict(lr=5e-4)
lr_config = dict(warmup_iters=100)

evaluation = dict(interval=200)
checkpoint_config = dict(interval=200)
runner = dict(type="MyRunnerWrapper", max_iters=total_iters)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ])

SyncBN = False
