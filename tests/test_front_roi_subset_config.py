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
        cfg = runpy.run_path(str(CONFIG_PATH), init_globals={
            "coords_dim": 2,
            "num_points": 20,
            "permute": True,
            "canvas_size": (200, 100),
            "thickness": 3,
            "img_size": (608, 608),
            "img_norm_cfg": {
                "mean": [103.530, 116.280, 123.675],
                "std": [1.0, 1.0, 1.0],
                "to_rgb": False,
            },
            "meta": {},
            "cat2id": {"ped_crossing": 0, "divider": 1, "boundary": 2},
        })

        self.assertEqual(cfg["roi_range"], (0, -15, 45, 15))
        self.assertEqual(cfg["roi_size"], (45, 30))
        self.assertEqual(cfg["pc_range"], [0, -15, -3, 45, 15, 5])
        self.assertEqual(cfg["overfit_data_root"],
                         "/data/waymo_maptracker_x0_x45_y15_overfit6")
        self.assertEqual(cfg["runner"]["max_iters"], 2000)
        self.assertIn("front_roi", cfg["work_dir"])


if __name__ == "__main__":
    unittest.main()
