import pathlib
import runpy
import unittest


CONFIG_PATH = (
    pathlib.Path(__file__).parents[1]
    / "subset"
    / "debug_overfit6_stage1_front_roi.py"
)


class FrontRoiSubsetConfigTest(unittest.TestCase):

    def test_front_roi_overfit_config_values(self):
        cfg = runpy.run_path(str(CONFIG_PATH))

        self.assertEqual(cfg["roi_range"], (0, -15, 45, 15))
        self.assertEqual(cfg["roi_size"], (45, 30))
        self.assertEqual(cfg["pc_range"], [0, -15, -3, 45, 15, 5])
        self.assertEqual(cfg["overfit_data_root"],
                         "/data/waymo_maptracker_x0_x45_y15_overfit6")
        self.assertEqual(cfg["runner"]["max_iters"], 2000)
        self.assertIn("front_roi", cfg["work_dir"])


if __name__ == "__main__":
    unittest.main()
