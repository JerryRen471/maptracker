import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_waymo_pipeline.sh"
STAGE1_CONFIG = (
    "plugin/configs/maptracker/waymo_5cam/"
    "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"
)


class RunWaymoPipelineTest(unittest.TestCase):

    def test_dry_run_infers_stages_and_prints_pipeline_commands(self):
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--dry-run",
                "--stage1-config",
                STAGE1_CONFIG,
                "--exp-tag",
                "unittest",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("maptracker_waymo_5cam_5frame_span10_stage2_warmup.py", result.stdout)
        self.assertIn("maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune.py", result.stdout)
        self.assertIn("tools/data_converter/waymo_map_converter.py", result.stdout)
        self.assertIn("pack_waymo_for_maptracker.py", result.stdout)
        self.assertIn("tools/tracking/prepare_gt_tracks.py", result.stdout)
        self.assertIn("tools/dist_train.sh", result.stdout)
        self.assertIn("tools/test.py", result.stdout)
        self.assertIn("tools/visualization/vis_global.py", result.stdout)

    def test_dry_run_propagates_maptracker_dir_to_all_config_consumers(self):
        maptracker_dir = "/tmp/waymo_maptracker_custom"
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--dry-run",
                "--stage1-config",
                STAGE1_CONFIG,
                "--maptracker-dir",
                maptracker_dir,
                "--skip-convert",
                "--skip-pack",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"data.train.ann_file={maptracker_dir}/waymo_map_infos_train.pkl",
            result.stdout,
        )
        self.assertIn(
            f"data.val.ann_file={maptracker_dir}/waymo_map_infos_val.pkl",
            result.stdout,
        )
        self.assertIn(
            f"data.test.ann_file={maptracker_dir}/waymo_map_infos_val.pkl",
            result.stdout,
        )
        self.assertIn(
            f"match_config.ann_file={maptracker_dir}/waymo_map_infos_val.pkl",
            result.stdout,
        )
        self.assertIn(
            f"eval_config.ann_file={maptracker_dir}/waymo_map_infos_val.pkl",
            result.stdout,
        )

    def test_dry_run_propagates_roi_to_all_config_consumers(self):
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--dry-run",
                "--stage1-config",
                STAGE1_CONFIG,
                "--x-min",
                "0",
                "--y-min",
                "-20",
                "--x-max",
                "50",
                "--y-max",
                "20",
                "--skip-convert",
                "--skip-pack",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("roi_range=[0,-20,50,20]", result.stdout)
        self.assertIn("roi_size=[50,40]", result.stdout)
        self.assertIn("pc_range=[0,-20,-3,50,20,5]", result.stdout)


if __name__ == "__main__":
    unittest.main()
