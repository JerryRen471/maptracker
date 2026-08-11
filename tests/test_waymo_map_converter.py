import importlib.util
import pathlib
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "tools"
    / "data_converter"
    / "waymo_map_converter.py"
)
SPEC = importlib.util.spec_from_file_location("waymo_map_converter", MODULE_PATH)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class WaymoMapConverterTest(unittest.TestCase):

    def test_asymmetric_roi_clips_longitudinal_line(self):
        points = np.array([[-30.0, 0.0], [60.0, 0.0]])

        clipped = CONVERTER.crop_polyline_to_roi(
            points, (-15.0, -15.0, 45.0, 15.0))

        self.assertEqual(len(clipped), 1)
        np.testing.assert_allclose(
            clipped[0], [[-15.0, 0.0], [45.0, 0.0]])

    def test_asymmetric_roi_rejects_lateral_outside_line(self):
        points = np.array([[-10.0, 20.0], [30.0, 20.0]])

        clipped = CONVERTER.crop_polyline_to_roi(
            points, (-15.0, -15.0, 45.0, 15.0))

        self.assertEqual(clipped, [])


if __name__ == "__main__":
    unittest.main()
