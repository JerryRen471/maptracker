import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "plugin"
    / "roi.py"
)
SPEC = importlib.util.spec_from_file_location("maptracker_roi", MODULE_PATH)
ROI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROI)


class RoiTest(unittest.TestCase):

    def test_symmetric_roi_size_remains_supported(self):
        roi_range, roi_size = ROI.resolve_roi((30, 60))

        np.testing.assert_allclose(roi_range, [-15, -30, 15, 30])
        np.testing.assert_allclose(roi_size, [30, 60])

    def test_asymmetric_roi_normalize_round_trip(self):
        roi_range, roi_size = ROI.resolve_roi(
            roi_size=(60, 30),
            roi_range=(-15, -15, 45, 15),
        )
        points = np.array([
            [-15.0, -15.0],
            [15.0, 0.0],
            [45.0, 15.0],
        ])

        normalized = ROI.normalize_points(points, roi_range, roi_size)
        restored = ROI.denormalize_points(
            normalized, roi_range, roi_size)

        np.testing.assert_allclose(
            normalized,
            [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            atol=1e-6,
        )
        np.testing.assert_allclose(restored, points, atol=1e-6)

    def test_metric_to_grid_uses_asymmetric_origin(self):
        roi_range, roi_size = ROI.resolve_roi(
            roi_size=(60, 30),
            roi_range=(-15, -15, 45, 15),
        )
        points = np.array([
            [-15.0, 15.0],
            [15.0, 0.0],
            [45.0, -15.0],
        ])

        grid = ROI.metric_to_grid(points, roi_range, roi_size)

        np.testing.assert_allclose(
            grid,
            [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
            atol=1e-6,
        )

    def test_cache_tag_includes_asymmetric_bounds(self):
        tag = ROI.roi_cache_tag((-15, -15, 45, 15))

        self.assertEqual(tag, "xm15_ym15_x45_y15")


if __name__ == "__main__":
    unittest.main()
