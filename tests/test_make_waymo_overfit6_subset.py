import importlib.util
import pathlib
import pickle
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).parents[1]
    / "subset"
    / "make_waymo_overfit6_subset.py"
)
SPEC = importlib.util.spec_from_file_location("make_waymo_overfit6_subset", SCRIPT_PATH)
SUBSET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBSET)


class MakeWaymoOverfitSubsetTest(unittest.TestCase):

    def test_writes_same_selected_scenes_to_train_and_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            source_dir = tmp_path / "source"
            output_dir = tmp_path / "subset"
            source_dir.mkdir()

            samples = []
            for scene_idx in range(8):
                scene = f"scene_{scene_idx:02d}"
                for frame_idx in range(2):
                    samples.append({
                        "scene_name": scene,
                        "frame_idx": frame_idx,
                        "token": f"{scene}_{frame_idx}",
                    })

            payload = {
                "samples": samples,
                "id2map": {},
                "metadata": {"roi_range": (-15, -15, 45, 15)},
            }
            for split in ("train", "val"):
                with open(source_dir / f"waymo_map_infos_{split}.pkl", "wb") as f:
                    pickle.dump(payload, f)

            selected = SUBSET.create_subset(
                source_dir=source_dir,
                output_dir=output_dir,
                num_scenes=6,
            )

            self.assertEqual(len(selected), 6)
            self.assertTrue((output_dir / "selected_scenes.txt").is_file())

            for split in ("train", "val"):
                with open(output_dir / f"waymo_map_infos_{split}.pkl", "rb") as f:
                    out = pickle.load(f)
                out_scenes = {s["scene_name"] for s in out["samples"]}
                self.assertEqual(out_scenes, set(selected))
                self.assertEqual(len(out["samples"]), 12)
                self.assertEqual(out["metadata"]["roi_range"], (-15, -15, 45, 15))


if __name__ == "__main__":
    unittest.main()
