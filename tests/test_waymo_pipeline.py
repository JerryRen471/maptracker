import importlib.util
import json
import pathlib
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "waymo_pipeline.py"
STAGE1_CONFIG = (
    "plugin/configs/maptracker/waymo_5cam/"
    "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"
)


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("waymo_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_maptracker_payload(path, sample_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": [
            {"scene_name": f"scene_{index // 40:03d}", "sample_idx": index}
            for index in range(sample_count)
        ]
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def write_config(tmpdir, **overrides):
    base = {
        "paths": {
            "conda_home": "/root/miniconda3",
            "waymo_env": "waymo_pack",
            "train_env": "maptracker",
            "waymo_train_dir": str(tmpdir / "waymo_train"),
            "waymo_val_dir": str(tmpdir / "waymo_val"),
            "processed_dir": str(tmpdir / "processed"),
            "maptracker_dir": str(tmpdir / "maptracker"),
            "work_root": str(tmpdir / "work_dirs"),
            "reuse_images_from": "",
            "mmdet3d_dir": "",
            "extra_pythonpath": [],
        },
        "configs": {
            "stage1": STAGE1_CONFIG,
            "stage2": None,
            "stage3": None,
        },
        "roi": {
            "x_min": -15,
            "y_min": -15,
            "x_max": 45,
            "y_max": 15,
        },
        "convert": {
            "frame_stride": 5,
            "num_points": 20,
            "num_workers": 1,
        },
        "runtime": {
            "gpus": "0",
            "num_gpus": 1,
            "exp_tag": "unit",
            "dry_run": True,
            "init_ckpt": "",
        },
        "generated_configs": {
            "enabled": True,
            "out_dir": "",
            "overwrite": True,
            "schedule": {},
        },
        "subset": {
            "enabled": False,
            "output_dir": "",
            "num_scenes": 6,
            "scenes": [],
        },
        "visualize": {
            "scene_ids": [],
            "per_frame_result": 1,
            "overwrite": 1,
            "draw_bev_range": True,
            "semantic": {},
        },
    }
    for section, values in overrides.items():
        base[section].update(values)

    path = tmpdir / "waymo_pipeline.yml"
    path.write_text(
        textwrap.dedent(
            f"""
            paths:
              conda_home: {base['paths']['conda_home']}
              waymo_env: {base['paths']['waymo_env']}
              train_env: {base['paths']['train_env']}
              waymo_train_dir: {base['paths']['waymo_train_dir']}
              waymo_val_dir: {base['paths']['waymo_val_dir']}
              processed_dir: {base['paths']['processed_dir']}
              maptracker_dir: {base['paths']['maptracker_dir']}
              work_root: {base['paths']['work_root']}
              reuse_images_from: "{base['paths']['reuse_images_from']}"
              mmdet3d_dir: "{base['paths']['mmdet3d_dir']}"
              extra_pythonpath: {base['paths']['extra_pythonpath']}
            configs:
              stage1: {base['configs']['stage1']}
              stage2: {base['configs']['stage2'] or 'null'}
              stage3: {base['configs']['stage3'] or 'null'}
            roi:
              x_min: {base['roi']['x_min']}
              y_min: {base['roi']['y_min']}
              x_max: {base['roi']['x_max']}
              y_max: {base['roi']['y_max']}
            convert:
              frame_stride: {base['convert']['frame_stride']}
              num_points: {base['convert']['num_points']}
              num_workers: {base['convert']['num_workers']}
            runtime:
              gpus: "{base['runtime']['gpus']}"
              num_gpus: {base['runtime']['num_gpus']}
              exp_tag: {base['runtime']['exp_tag']}
              dry_run: {str(base['runtime']['dry_run']).lower()}
              init_ckpt: "{base['runtime']['init_ckpt']}"
            generated_configs:
              enabled: {str(base['generated_configs']['enabled']).lower()}
              out_dir: "{base['generated_configs']['out_dir']}"
              overwrite: {str(base['generated_configs']['overwrite']).lower()}
              schedule: {base['generated_configs']['schedule']}
            subset:
              enabled: {str(base['subset']['enabled']).lower()}
              output_dir: "{base['subset']['output_dir']}"
              num_scenes: {base['subset']['num_scenes']}
              scenes: {base['subset']['scenes']}
            visualize:
              scene_ids: []
              per_frame_result: {base['visualize']['per_frame_result']}
              overwrite: {base['visualize']['overwrite']}
              draw_bev_range: {str(base['visualize']['draw_bev_range']).lower()}
              semantic: {base['visualize']['semantic']}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


class WaymoPipelineTest(unittest.TestCase):

    def test_check_only_reports_missing_pack_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            cfg = write_config(tmpdir)

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "pack",
                    "--check-only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "blocked")
            missing_paths = {item["path"] for item in payload["missing"]}
            self.assertIn(str(tmpdir / "processed" / "train_infos.pkl"), missing_paths)
            self.assertIn(str(tmpdir / "processed" / "val_infos.pkl"), missing_paths)

    def test_check_only_accepts_pack_when_inputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            cfg = write_config(tmpdir)
            (tmpdir / "processed").mkdir()
            (tmpdir / "processed" / "train_infos.pkl").write_bytes(b"")
            (tmpdir / "processed" / "val_infos.pkl").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "pack",
                    "--check-only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["requested_steps"], ["pack"])
            self.assertEqual(payload["missing"], [])

    def test_dry_run_generates_stage_commands_with_consistent_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            cfg = write_config(tmpdir, generated_configs={"enabled": False})
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_train.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_train_gt_tracks.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val_gt_tracks.pkl").write_bytes(b"")
            stage1_work = (
                tmpdir
                / "work_dirs"
                / "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_unit"
            )
            stage1_work.mkdir(parents=True)
            (stage1_work / "latest.pth").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "train_stage2",
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ready")
            command = payload["commands"][0]["command"]
            self.assertIn("tools/dist_train.sh", command)
            self.assertIn("maptracker_waymo_5cam_5frame_span10_stage2_warmup.py", command)
            self.assertIn(
                f"data.train.ann_file={tmpdir}/maptracker/waymo_map_infos_train.pkl",
                command,
            )
            self.assertIn(
                f"match_config.ann_file={tmpdir}/maptracker/waymo_map_infos_val.pkl",
                command,
            )
            self.assertIn("roi_range='[-15,-15,45,15]'", command)
            self.assertIn(f"load_from={stage1_work}/latest.pth", command)
            self.assertNotIn("/MapTR/mmdetection3d", command)
            self.assertIn(":${PYTHONPATH:-}", command)

    def test_stage1_uses_init_ckpt_when_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            init_ckpt = tmpdir / "init_stage1.pth"
            init_ckpt.write_bytes(b"")
            cfg = write_config(
                tmpdir,
                runtime={"init_ckpt": str(init_ckpt)},
                generated_configs={"enabled": False},
            )
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_train.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_train_gt_tracks.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val_gt_tracks.pkl").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "train_stage1",
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ready")
            command = payload["commands"][0]["command"]
            self.assertIn(f"load_from={init_ckpt}", command)

    def test_generated_overrides_use_stage1_init_ckpt_without_changing_stage2_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            init_ckpt = tmpdir / "init_stage1.pth"
            cfg_path = write_config(tmpdir, runtime={"init_ckpt": str(init_ckpt)})

            pipeline = load_pipeline_module()
            cfg = pipeline.build_config(pipeline.load_config(cfg_path), config_file=cfg_path)

            stage1_overrides = pipeline.cfg_override_dict(cfg, "1")
            stage2_overrides = pipeline.cfg_override_dict(cfg, "2")

            self.assertEqual(stage1_overrides["load_from"], str(init_ckpt))
            self.assertEqual(stage2_overrides["load_from"], str(cfg.derived["stage1_checkpoint"]))

    def test_all_steps_treat_selected_upstream_outputs_as_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            cfg = write_config(tmpdir)
            (tmpdir / "waymo_train").mkdir()
            (tmpdir / "waymo_val").mkdir()
            (tmpdir / "waymo_train" / "segment-train.tfrecord").write_bytes(b"")
            (tmpdir / "waymo_val" / "segment-val.tfrecord").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "all",
                    "--check-only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["missing"], [])
            self.assertIn("generate_configs", payload["requested_steps"])
            self.assertNotIn("subset", payload["requested_steps"])

    def test_train_stage_uses_generated_config_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            generated_dir = tmpdir / "generated"
            cfg = write_config(tmpdir, generated_configs={"out_dir": str(generated_dir)})
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_train.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_train_gt_tracks.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val_gt_tracks.pkl").write_bytes(b"")
            stage1_work = (
                tmpdir
                / "work_dirs"
                / "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_unit"
            )
            stage1_work.mkdir(parents=True)
            (stage1_work / "latest.pth").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "generate_configs,train_stage2",
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["commands"][0]["step"], "generate_configs")
            train_command = payload["commands"][1]["command"]
            self.assertIn(str(generated_dir), train_command)
            self.assertIn(
                "maptracker_waymo_5cam_5frame_span10_stage2_warmup_unit.py",
                train_command,
            )
            self.assertNotIn(" --cfg-options ", train_command)

    def test_train_stage_reports_missing_generated_config_without_generation_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            generated_dir = tmpdir / "generated"
            cfg = write_config(tmpdir, generated_configs={"out_dir": str(generated_dir)})
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_train.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_train_gt_tracks.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val_gt_tracks.pkl").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "train_stage1",
                    "--check-only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            payload = json.loads(result.stdout)
            missing_paths = {item["path"] for item in payload["missing"]}
            self.assertIn(
                str(generated_dir / "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_unit.py"),
                missing_paths,
            )

    def test_generated_schedule_uses_subset_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            subset_dir = tmpdir / "maptracker_subset"
            cfg_path = write_config(
                tmpdir,
                runtime={"num_gpus": 4},
                subset={"enabled": True, "output_dir": str(subset_dir)},
            )
            write_maptracker_payload(subset_dir / "waymo_map_infos_train.pkl", 240)

            pipeline = load_pipeline_module()
            cfg = pipeline.build_config(pipeline.load_config(cfg_path), config_file=cfg_path)
            stage_cfg = {
                "num_gpus": 4,
                "batch_size": 1,
                "num_epochs": 3,
                "num_epochs_interval": 1,
                "total_iters": 75624,
                "runner": {"type": "MyRunnerWrapper", "max_iters": 75624},
                "evaluation": {"interval": 25208},
                "checkpoint_config": {"interval": 25208},
                "lr_config": {"warmup_iters": 500},
            }

            overrides = pipeline.schedule_override_dict(cfg, stage_cfg)

            self.assertEqual(overrides["num_train_samples"], 240)
            self.assertEqual(overrides["num_iters_per_epoch"], 60)
            self.assertEqual(overrides["total_iters"], 180)
            self.assertEqual(overrides["runner.max_iters"], 180)
            self.assertEqual(overrides["evaluation.interval"], 60)
            self.assertEqual(overrides["checkpoint_config.interval"], 60)
            self.assertEqual(overrides["lr_config.warmup_iters"], 18)

    def test_generated_schedule_allows_num_epochs_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            subset_dir = tmpdir / "maptracker_subset"
            cfg_path = write_config(
                tmpdir,
                runtime={"num_gpus": 2},
                generated_configs={"schedule": {"num_epochs": 20}},
                subset={"enabled": True, "output_dir": str(subset_dir)},
            )
            write_maptracker_payload(subset_dir / "waymo_map_infos_train.pkl", 240)

            pipeline = load_pipeline_module()
            cfg = pipeline.build_config(pipeline.load_config(cfg_path), config_file=cfg_path)
            stage_cfg = {
                "batch_size": 3,
                "num_epochs": 3,
                "num_epochs_interval": 1,
                "runner": {"max_iters": 75624},
                "checkpoint_config": {"interval": 25208},
                "lr_config": {"warmup_iters": 500},
            }

            overrides = pipeline.schedule_override_dict(cfg, stage_cfg)

            self.assertEqual(overrides["num_iters_per_epoch"], 40)
            self.assertEqual(overrides["num_epochs"], 20)
            self.assertEqual(overrides["total_iters"], 800)
            self.assertEqual(overrides["runner.max_iters"], 800)
            self.assertEqual(overrides["lr_config.warmup_iters"], 80)

    def test_subset_step_generates_subset_command_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            subset_dir = tmpdir / "maptracker_subset"
            cfg = write_config(
                tmpdir,
                subset={
                    "enabled": True,
                    "output_dir": str(subset_dir),
                    "num_scenes": 3,
                },
            )
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_train.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val.pkl").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "subset",
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["requested_steps"], ["subset"])
            command = payload["commands"][0]["command"]
            self.assertIn("subset/make_waymo_overfit6_subset.py", command)
            self.assertIn(f"--source-dir {tmpdir}/maptracker", command)
            self.assertIn(f"--output-dir {subset_dir}", command)
            self.assertIn("--num-scenes 3", command)

    def test_subset_enabled_makes_downstream_use_subset_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            subset_dir = tmpdir / "maptracker_subset"
            cfg = write_config(
                tmpdir,
                generated_configs={"enabled": False},
                subset={"enabled": True, "output_dir": str(subset_dir), "num_scenes": 2},
            )
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_train.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val.pkl").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "subset,gt_tracks",
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ready")
            gt_command = payload["commands"][1]["command"]
            self.assertIn(
                f"data.train.ann_file={subset_dir}/waymo_map_infos_train.pkl",
                gt_command,
            )
            self.assertIn(
                f"match_config.ann_file={subset_dir}/waymo_map_infos_val.pkl",
                gt_command,
            )

    def test_subset_enabled_requires_subset_outputs_for_downstream_without_subset_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            subset_dir = tmpdir / "maptracker_subset"
            cfg = write_config(
                tmpdir,
                generated_configs={"enabled": False},
                subset={"enabled": True, "output_dir": str(subset_dir)},
            )
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_train.pkl").write_bytes(b"")
            (tmpdir / "maptracker" / "waymo_map_infos_val.pkl").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "gt_tracks",
                    "--check-only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            payload = json.loads(result.stdout)
            missing_paths = {item["path"] for item in payload["missing"]}
            self.assertIn(str(subset_dir / "waymo_map_infos_train.pkl"), missing_paths)
            self.assertIn(str(subset_dir / "waymo_map_infos_val.pkl"), missing_paths)

    def test_visualize_includes_semantic_segmentation_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            generated_dir = tmpdir / "generated"
            cfg = write_config(
                tmpdir,
                generated_configs={"out_dir": str(generated_dir)},
                visualize={"semantic": {"max_frames": 5, "split": "val"}},
            )
            generated_dir.mkdir()
            stage1_config = generated_dir / "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_unit.py"
            stage3_config = generated_dir / "maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_unit.py"
            stage1_config.write_text("# stage1\n", encoding="utf-8")
            stage3_config.write_text("# stage3\n", encoding="utf-8")
            (tmpdir / "maptracker").mkdir()
            (tmpdir / "maptracker" / "waymo_map_infos_val_gt_tracks.pkl").write_bytes(b"")
            stage1_work = (
                tmpdir
                / "work_dirs"
                / "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_unit"
            )
            stage3_work = (
                tmpdir
                / "work_dirs"
                / "maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_unit"
            )
            (stage1_work).mkdir(parents=True)
            (stage3_work / "eval").mkdir(parents=True)
            (stage1_work / "latest.pth").write_bytes(b"")
            (stage3_work / "latest.pth").write_bytes(b"")
            (stage3_work / "eval" / "pos_predictions.pkl").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "visualize",
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["requested_steps"], ["visualize", "visualize_semantic"])
            self.assertEqual([item["step"] for item in payload["commands"]], [
                "visualize_pred",
                "visualize_gt",
                "visualize_semantic",
            ])
            self.assertIn("--draw_bev_range 1", payload["commands"][0]["command"])
            self.assertIn("--draw_bev_range 1", payload["commands"][1]["command"])
            semantic_command = payload["commands"][2]["command"]
            self.assertIn("tools/check_seg.py", semantic_command)
            self.assertIn(f"--config {stage1_config}", semantic_command)
            self.assertIn(f"--checkpoint {stage1_work}/latest.pth", semantic_command)
            self.assertIn(f"--out-dir {stage3_work}/visualization/semantic", semantic_command)
            self.assertIn("--split val", semantic_command)
            self.assertIn("--draw-bev-range 1", semantic_command)
            self.assertIn("--max-frames 5", semantic_command)

    def test_visualize_semantic_can_run_without_vector_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            generated_dir = tmpdir / "generated"
            cfg = write_config(
                tmpdir,
                generated_configs={"out_dir": str(generated_dir)},
                visualize={"semantic": {"stage": "stage2", "max_frames": 2}},
            )
            generated_dir.mkdir()
            stage2_config = generated_dir / "maptracker_waymo_5cam_5frame_span10_stage2_warmup_unit.py"
            stage2_config.write_text("# stage2\n", encoding="utf-8")
            stage2_work = (
                tmpdir
                / "work_dirs"
                / "maptracker_waymo_5cam_5frame_span10_stage2_warmup_unit"
            )
            stage2_work.mkdir(parents=True)
            (stage2_work / "latest.pth").write_bytes(b"")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--config",
                    str(cfg),
                    "--steps",
                    "visualize_semantic",
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["requested_steps"], ["visualize_semantic"])
            self.assertEqual(len(payload["commands"]), 1)
            command = payload["commands"][0]["command"]
            self.assertIn("tools/check_seg.py", command)
            self.assertIn(f"--config {stage2_config}", command)
            self.assertNotIn("tools/visualization/vis_global.py", command)


if __name__ == "__main__":
    unittest.main()
