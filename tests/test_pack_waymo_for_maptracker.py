import importlib.util
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pack_waymo_for_maptracker.py"
SPEC = importlib.util.spec_from_file_location("pack_waymo_for_maptracker", SCRIPT)
PACKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKER)


class PackWaymoForMapTrackerTest(unittest.TestCase):

    def test_cam2ego_to_ego2cam_outputs_pinhole_positive_depth(self):
        ext = PACKER.cam2ego_to_ego2cam(
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )

        point_ego = np.array([10.0, 0.0, 0.0, 1.0])
        point_cam = ext @ point_ego

        self.assertGreater(point_cam[2], 0.0)
        np.testing.assert_allclose(point_cam[:3], [0.0, 0.0, 10.0])

    def test_convert_cams_preserves_img_shape_for_projection_checks(self):
        cams = {
            "FRONT": {
                "cam_intrinsic": np.eye(3, dtype=np.float64),
                "sensor2ego_rotation": np.eye(3, dtype=np.float64),
                "sensor2ego_translation": np.zeros(3, dtype=np.float64),
                "data_path": "images/0000_FRONT.jpg",
                "img_shape": (1280, 1920),
            }
        }

        converted = PACKER.convert_cams(cams, img_root="/data/waymo")

        self.assertEqual(converted["FRONT"]["img_shape"], (1280, 1920))

    def test_validate_roi_accepts_matching_asym_metadata(self):
        metadata = {
            "roi_range": (-15.0, -15.0, 45.0, 15.0),
            "bev_x": 30.0,
            "bev_y": 15.0,
        }
        samples = [{
            "token": "ok",
            "gt_polylines": [np.array([[-15.0, -15.0], [45.0, 15.0]], dtype=np.float32)],
        }]
        PACKER.validate_roi(metadata, samples, (-15, -15, 45, 15))

    def test_validate_roi_rejects_legacy_v3_half_extents(self):
        metadata = {"bev_x": 15.0, "bev_y": 30.0}
        samples = [{
            "token": "bad",
            "gt_polylines": [np.array([[-15.0, -30.0], [15.0, 30.0]], dtype=np.float32)],
        }]
        with self.assertRaises(ValueError):
            PACKER.validate_roi(metadata, samples, (-15, -15, 45, 15))


if __name__ == "__main__":
    unittest.main()
