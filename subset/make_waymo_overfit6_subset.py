#!/usr/bin/env python3
import argparse
import pickle
from collections import OrderedDict
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("/data/waymo_maptracker_xm15_x45_y15")
DEFAULT_OUTPUT_DIR = Path("/data/waymo_maptracker_xm15_x45_y15_overfit6")


def _load_split(source_dir, split):
    path = source_dir / f"waymo_map_infos_{split}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _scene_order(samples):
    scenes = OrderedDict()
    for sample in samples:
        scenes.setdefault(sample["scene_name"], None)
    return list(scenes.keys())


def _filter_payload(payload, selected_scenes):
    selected = set(selected_scenes)
    out = dict(payload)
    out["samples"] = [
        sample for sample in payload["samples"]
        if sample["scene_name"] in selected
    ]
    return out


def create_subset(source_dir=DEFAULT_SOURCE_DIR, output_dir=DEFAULT_OUTPUT_DIR,
                  num_scenes=6, scenes=None):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    train_payload = _load_split(source_dir, "train")
    val_payload = _load_split(source_dir, "val")

    if scenes:
        selected_scenes = list(scenes)
    else:
        selected_scenes = _scene_order(train_payload["samples"])[:num_scenes]

    if len(selected_scenes) != num_scenes:
        raise ValueError(
            f"expected {num_scenes} scenes, got {len(selected_scenes)}")

    train_scenes = set(_scene_order(train_payload["samples"]))
    missing = [scene for scene in selected_scenes if scene not in train_scenes]
    if missing:
        raise ValueError(f"selected scenes are not in train split: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": _filter_payload(train_payload, selected_scenes),
        "val": _filter_payload(train_payload, selected_scenes),
    }

    for split, payload in outputs.items():
        path = output_dir / f"waymo_map_infos_{split}.pkl"
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_dir / "selected_scenes.txt", "w") as f:
        for scene in selected_scenes:
            f.write(f"{scene}\n")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "selected_scenes": selected_scenes,
        "train_samples": len(outputs["train"]["samples"]),
        "val_samples": len(outputs["val"]["samples"]),
        "source_val_samples": len(val_payload["samples"]),
    }
    with open(output_dir / "subset_summary.pkl", "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    return selected_scenes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 6-scene Waymo MapTracker overfit subset.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-scenes", type=int, default=6)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Explicit scene_name values. Defaults to the first N train scenes.")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = create_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        scenes=args.scenes,
    )
    print("OVERFIT_SUBSET_READY")
    print(f"source_dir={args.source_dir}")
    print(f"output_dir={args.output_dir}")
    print("selected_scenes=" + ",".join(selected))


if __name__ == "__main__":
    main()
