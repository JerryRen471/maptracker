import json
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "waymo_pipeline.py"
STAGE1_CONFIG = (
    "plugin/configs/maptracker/waymo_5cam/"
    "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"
)


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
        },
        "generated_configs": {
            "enabled": True,
            "out_dir": "",
            "overwrite": True,
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
            generated_configs:
              enabled: {str(base['generated_configs']['enabled']).lower()}
              out_dir: "{base['generated_configs']['out_dir']}"
              overwrite: {str(base['generated_configs']['overwrite']).lower()}
            subset:
              enabled: {str(base['subset']['enabled']).lower()}
              output_dir: "{base['subset']['output_dir']}"
              num_scenes: {base['subset']['num_scenes']}
              scenes: {base['subset']['scenes']}
            visualize:
              scene_ids: []
              per_frame_result: {base['visualize']['per_frame_result']}
              overwrite: {base['visualize']['overwrite']}
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


if __name__ == "__main__":
    unittest.main()
