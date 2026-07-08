import importlib.util
import pathlib
import pickle
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_waymo_camera_projection.py"
SPEC = importlib.util.spec_from_file_location(
    "check_waymo_camera_projection", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckWaymoCameraProjectionTest(unittest.TestCase):

    def test_project_points_uses_third_coordinate_as_depth(self):
        intrinsic = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        extrinsic = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        points = [
            [2.0, 4.0, 2.0, 1.0],
            [3.0, 6.0, 3.0, 1.0],
        ]

        projected = CHECKER.project_points(points, intrinsic, extrinsic)

        self.assertEqual(projected["depth"], [2.0, 3.0])
        self.assertEqual(projected["u"], [1.0, 1.0])
        self.assertEqual(projected["v"], [2.0, 2.0])

    def test_camera_stats_reports_visible_grid_and_probe_counts(self):
        sample = {
            "cams": {
                "FRONT": {
                    "intrinsics": [
                        [1.0, 0.0, 5.0],
                        [0.0, 1.0, 5.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "extrinsics": [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "img_shape": (10, 10),
                }
            }
        }

        stats = CHECKER.camera_projection_stats(
            sample,
            roi_range=(1.0, 1.0, 3.0, 3.0),
            bev_h=2,
            bev_w=2,
            probe_x=[1.0, 2.0],
            probe_y=2.0,
            cameras=None,
        )

        front = stats["FRONT"]
        self.assertEqual(front["grid_total"], 4)
        self.assertEqual(front["grid_depth_positive"], 4)
        self.assertEqual(front["grid_in_image"], 4)
        self.assertEqual(front["probe_depth_positive"], 2)

    def test_load_samples_accepts_dict_with_samples(self):
        payload = {"samples": [{"token": "a"}, {"token": "b"}]}
        with tempfile.TemporaryDirectory() as tmp_dir:
            ann_file = pathlib.Path(tmp_dir) / "infos.pkl"
            with ann_file.open("wb") as f:
                pickle.dump(payload, f)

            samples = CHECKER.load_samples(ann_file)

        self.assertEqual([sample["token"] for sample in samples], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
