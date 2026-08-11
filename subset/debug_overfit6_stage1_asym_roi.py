_base_ = [
    "../plugin/configs/maptracker/waymo_5cam/"
    "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"
]

overfit_data_root = "/data/waymo_maptracker_xm15_x45_y15_overfit6"

num_gpus = 1
batch_size = 1
num_iters_per_epoch = 100
num_epochs = 20
total_iters = num_epochs * num_iters_per_epoch

work_dir = (
    "/data/maptr_workspace/work_dirs/"
    "debug_overfit6_stage1_asym_roi")

eval_config = dict(
    ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl")

match_config = dict(
    ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl")

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        ann_file=f"{overfit_data_root}/waymo_map_infos_train.pkl"),
    val=dict(
        ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl",
        eval_config=eval_config),
    test=dict(
        ann_file=f"{overfit_data_root}/waymo_map_infos_val.pkl",
        eval_config=eval_config),
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
