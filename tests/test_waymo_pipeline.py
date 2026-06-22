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
            cfg = write_config(tmpdir)
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


if __name__ == "__main__":
    unittest.main()
