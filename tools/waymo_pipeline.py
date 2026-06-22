#!/usr/bin/env python3
"""Config-driven Waymo MapTracker pipeline runner."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

PIPELINE_ORDER = [
    "generate_configs",
    "convert",
    "pack",
    "gt_tracks",
    "train_stage1",
    "train_stage2",
    "train_stage3",
    "test",
    "visualize",
]

STEP_ALIASES = {
    "all": PIPELINE_ORDER,
    "full": PIPELINE_ORDER,
    "generate_configs": ["generate_configs"],
    "configs": ["generate_configs"],
    "train": ["train_stage1", "train_stage2", "train_stage3"],
    "train_stage": ["train_stage1", "train_stage2", "train_stage3"],
    "pack": ["pack"],
    "convert": ["convert"],
    "gt": ["gt_tracks"],
    "gt_tracks": ["gt_tracks"],
    "test": ["test"],
    "eval": ["test"],
    "visualize": ["visualize"],
    "vis": ["visualize"],
    "train_stage1": ["train_stage1"],
    "train_stage2": ["train_stage2"],
    "train_stage3": ["train_stage3"],
    "stage1": ["train_stage1"],
    "stage2": ["train_stage2"],
    "stage3": ["train_stage3"],
}


@dataclass
class MissingItem:
    step: str
    kind: str
    path: str
    message: str


@dataclass
class CommandSpec:
    step: str
    command: str
    env: str


@dataclass
class PipelineConfig:
    config_file: Path | None
    raw: dict[str, Any]
    paths: dict[str, Any]
    configs: dict[str, Any]
    generated_configs: dict[str, Any]
    roi: dict[str, Any]
    convert: dict[str, Any]
    runtime: dict[str, Any]
    visualize: dict[str, Any]
    stage1_config: Path
    stage2_config: Path
    stage3_config: Path
    derived: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or precheck the Waymo MapTracker pipeline")
    parser.add_argument("--config", required=True, help="YAML or JSON pipeline config")
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps or aliases: all, convert, pack, gt_tracks, train, "
        "train_stage1, train_stage2, train_stage3, test, visualize",
    )
    parser.add_argument("--check-only", action="store_true", help="Only check prerequisites")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--materialize-configs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def config_path(value: str | None, default: str | None = None) -> Path:
    if value is None:
        value = default
    if value is None:
        raise ValueError("missing required config path")
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel_config(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def infer_stage_config(stage1: str, target: str) -> str:
    if target == "stage2":
        return stage1.replace("stage1_bev_pretrain", "stage2_warmup")
    if target == "stage3":
        return stage1.replace("stage1_bev_pretrain", "stage3_joint_finetune")
    raise ValueError(f"invalid target stage: {target}")


def build_config(raw: dict[str, Any], config_file: Path | None = None) -> PipelineConfig:
    paths = {
        "conda_home": "/root/miniconda3",
        "waymo_env": "waymo_pack",
        "train_env": "maptracker",
        "waymo_train_dir": "/data8012/waymo/training",
        "waymo_val_dir": "/data8012/waymo/validation",
        "processed_dir": "/data/waymo_processed_xm15_x45_y15",
        "maptracker_dir": "/data/waymo_maptracker_xm15_x45_y15",
        "work_root": "/data/maptr_workspace/work_dirs",
        "reuse_images_from": "/data/waymo_processed_v3",
    }
    paths.update(section(raw, "paths"))

    configs = section(raw, "configs")
    stage1_value = configs.get(
        "stage1",
        "plugin/configs/maptracker/waymo_5cam/"
        "maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py",
    )
    stage2_value = configs.get("stage2") or infer_stage_config(stage1_value, "stage2")
    stage3_value = configs.get("stage3") or infer_stage_config(stage1_value, "stage3")

    roi = {"x_min": -15, "y_min": -15, "x_max": 45, "y_max": 15}
    roi.update(section(raw, "roi"))

    convert = {"frame_stride": 5, "num_points": 20, "num_workers": 8}
    convert.update(section(raw, "convert"))

    runtime = {"gpus": "0", "num_gpus": None, "exp_tag": "asym_roi", "dry_run": False}
    runtime.update(section(raw, "runtime"))

    visualize = {"scene_ids": [], "per_frame_result": 1, "overwrite": 1}
    visualize.update(section(raw, "visualize"))

    stage1_config = config_path(stage1_value)
    stage2_config = config_path(stage2_value)
    stage3_config = config_path(stage3_value)

    generated_configs = {
        "enabled": True,
        "out_dir": None,
        "overwrite": True,
    }
    generated_configs.update(section(raw, "generated_configs"))

    cfg = PipelineConfig(
        config_file=config_file,
        raw=raw,
        paths=paths,
        configs=configs,
        generated_configs=generated_configs,
        roi=roi,
        convert=convert,
        runtime=runtime,
        visualize=visualize,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        stage3_config=stage3_config,
    )
    cfg.derived = derive_values(cfg)
    return cfg


def derive_values(cfg: PipelineConfig) -> dict[str, Any]:
    x_min = cfg.roi["x_min"]
    y_min = cfg.roi["y_min"]
    x_max = cfg.roi["x_max"]
    y_max = cfg.roi["y_max"]
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    maptracker_dir = Path(str(cfg.paths["maptracker_dir"]))
    processed_dir = Path(str(cfg.paths["processed_dir"]))

    stage1_work = stage_work_dir(cfg, cfg.stage1_config)
    stage2_work = stage_work_dir(cfg, cfg.stage2_config)
    stage3_work = stage_work_dir(cfg, cfg.stage3_config)
    generated_out_dir = generated_config_dir(cfg)
    generated_stage1 = generated_out_dir / f"{cfg.stage1_config.stem}_{cfg.runtime['exp_tag']}.py"
    generated_stage2 = generated_out_dir / f"{cfg.stage2_config.stem}_{cfg.runtime['exp_tag']}.py"
    generated_stage3 = generated_out_dir / f"{cfg.stage3_config.stem}_{cfg.runtime['exp_tag']}.py"

    return {
        "roi_range": f"[{x_min},{y_min},{x_max},{y_max}]",
        "roi_size": f"[{roi_width},{roi_height}]",
        "pc_range": f"[{x_min},{y_min},-3,{x_max},{y_max},5]",
        "processed_train": processed_dir / "train_infos.pkl",
        "processed_val": processed_dir / "val_infos.pkl",
        "maptracker_train": maptracker_dir / "waymo_map_infos_train.pkl",
        "maptracker_val": maptracker_dir / "waymo_map_infos_val.pkl",
        "maptracker_train_tracks": maptracker_dir / "waymo_map_infos_train_gt_tracks.pkl",
        "maptracker_val_tracks": maptracker_dir / "waymo_map_infos_val_gt_tracks.pkl",
        "stage1_work": stage1_work,
        "stage2_work": stage2_work,
        "stage3_work": stage3_work,
        "stage1_checkpoint": stage1_work / "latest.pth",
        "stage2_checkpoint": stage2_work / "latest.pth",
        "stage3_checkpoint": stage3_work / "latest.pth",
        "generated_config_dir": generated_out_dir,
        "generated_stage1_config": generated_stage1,
        "generated_stage2_config": generated_stage2,
        "generated_stage3_config": generated_stage3,
        "test_work_dir": Path(str(cfg.runtime.get("test_work_dir") or (stage3_work / "eval"))),
        "vis_out_dir": Path(str(cfg.visualize.get("out_dir") or (stage3_work / "visualization"))),
    }


def stage_work_dir(cfg: PipelineConfig, config: Path) -> Path:
    base = config.stem
    return Path(str(cfg.paths["work_root"])) / f"{base}_{cfg.runtime['exp_tag']}"


def generated_config_dir(cfg: PipelineConfig) -> Path:
    out_dir = cfg.generated_configs.get("out_dir")
    if out_dir:
        return Path(str(out_dir))
    return Path(str(cfg.paths["work_root"])) / "generated_configs" / str(cfg.runtime["exp_tag"])


def generated_configs_enabled(cfg: PipelineConfig) -> bool:
    return bool(cfg.generated_configs.get("enabled", True))


def active_stage_config(cfg: PipelineConfig, stage_num: str) -> Path:
    if not generated_configs_enabled(cfg):
        return {"1": cfg.stage1_config, "2": cfg.stage2_config, "3": cfg.stage3_config}[stage_num]
    return {
        "1": cfg.derived["generated_stage1_config"],
        "2": cfg.derived["generated_stage2_config"],
        "3": cfg.derived["generated_stage3_config"],
    }[stage_num]


def expand_steps(value: str) -> list[str]:
    steps: list[str] = []
    for item in value.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in STEP_ALIASES:
            raise ValueError(f"unknown step: {name}")
        for expanded in STEP_ALIASES[name]:
            if expanded not in steps:
                steps.append(expanded)
    return [step for step in PIPELINE_ORDER if step in steps]


def path_has_tfrecords(path: Path) -> bool:
    return path.is_dir() and any(path.glob("segment-*.tfrecord"))


def produced_paths(cfg: PipelineConfig, step: str) -> list[Path]:
    d = cfg.derived
    return {
        "generate_configs": (
            [d["generated_stage1_config"], d["generated_stage2_config"], d["generated_stage3_config"]]
            if generated_configs_enabled(cfg)
            else []
        ),
        "convert": [d["processed_train"], d["processed_val"]],
        "pack": [d["maptracker_train"], d["maptracker_val"]],
        "gt_tracks": [d["maptracker_train_tracks"], d["maptracker_val_tracks"]],
        "train_stage1": [d["stage1_checkpoint"]],
        "train_stage2": [d["stage2_checkpoint"]],
        "train_stage3": [d["stage3_checkpoint"]],
        "test": [d["test_work_dir"] / "pos_predictions.pkl"],
        "visualize": [d["vis_out_dir"] / "pred", d["vis_out_dir"] / "gt"],
    }.get(step, [])


def required_paths(cfg: PipelineConfig, step: str) -> list[tuple[str, Path, str]]:
    d = cfg.derived
    requirements: dict[str, list[tuple[str, Path, str]]] = {
        "generate_configs": [
            ("file", cfg.stage1_config, "generate_configs requires base stage1 config"),
            ("file", cfg.stage2_config, "generate_configs requires base stage2 config"),
            ("file", cfg.stage3_config, "generate_configs requires base stage3 config"),
        ],
        "pack": [
            ("file", d["processed_train"], "pack requires converted train_infos.pkl"),
            ("file", d["processed_val"], "pack requires converted val_infos.pkl"),
        ],
        "gt_tracks": [
            ("file", d["maptracker_train"], "gt_tracks requires packed train pkl"),
            ("file", d["maptracker_val"], "gt_tracks requires packed val pkl"),
            ("file", active_stage_config(cfg, "1"), "gt_tracks requires stage1 config"),
        ],
        "train_stage1": [
            ("file", d["maptracker_train"], "stage1 requires packed train pkl"),
            ("file", d["maptracker_val"], "stage1 requires packed val pkl"),
            ("file", d["maptracker_train_tracks"], "stage1 requires train GT tracks"),
            ("file", d["maptracker_val_tracks"], "stage1 requires val GT tracks"),
            ("file", active_stage_config(cfg, "1"), "stage1 requires stage1 config"),
        ],
        "train_stage2": [
            ("file", d["stage1_checkpoint"], "stage2 requires stage1 checkpoint"),
            ("file", d["maptracker_train"], "stage2 requires packed train pkl"),
            ("file", d["maptracker_val"], "stage2 requires packed val pkl"),
            ("file", d["maptracker_train_tracks"], "stage2 requires train GT tracks"),
            ("file", d["maptracker_val_tracks"], "stage2 requires val GT tracks"),
            ("file", active_stage_config(cfg, "2"), "stage2 requires stage2 config"),
        ],
        "train_stage3": [
            ("file", d["stage2_checkpoint"], "stage3 requires stage2 checkpoint"),
            ("file", d["maptracker_train"], "stage3 requires packed train pkl"),
            ("file", d["maptracker_val"], "stage3 requires packed val pkl"),
            ("file", d["maptracker_train_tracks"], "stage3 requires train GT tracks"),
            ("file", d["maptracker_val_tracks"], "stage3 requires val GT tracks"),
            ("file", active_stage_config(cfg, "3"), "stage3 requires stage3 config"),
        ],
        "test": [
            ("file", d["stage3_checkpoint"], "test requires stage3 checkpoint"),
            ("file", d["maptracker_val"], "test requires packed val pkl"),
            ("file", active_stage_config(cfg, "3"), "test requires stage3 config"),
        ],
        "visualize": [
            ("file", d["test_work_dir"] / "pos_predictions.pkl", "visualize requires predictions"),
            ("file", d["maptracker_val_tracks"], "visualize requires val GT tracks"),
            ("file", active_stage_config(cfg, "3"), "visualize requires stage3 config"),
        ],
    }
    return requirements.get(step, [])


def precheck(cfg: PipelineConfig, steps: list[str]) -> list[MissingItem]:
    missing: list[MissingItem] = []
    virtually_available: set[Path] = set()

    if "convert" in steps:
        train_dir = Path(str(cfg.paths["waymo_train_dir"]))
        if not path_has_tfrecords(train_dir):
            missing.append(
                MissingItem(
                    "convert",
                    "directory",
                    str(train_dir),
                    "convert requires segment-*.tfrecord in waymo_train_dir",
                )
            )
        val_dir_value = cfg.paths.get("waymo_val_dir")
        if val_dir_value:
            val_dir = Path(str(val_dir_value))
            if not path_has_tfrecords(val_dir):
                missing.append(
                    MissingItem(
                        "convert",
                        "directory",
                        str(val_dir),
                        "convert requires segment-*.tfrecord in waymo_val_dir",
                    )
                )
        reuse = cfg.paths.get("reuse_images_from")
        if reuse:
            image_dir = Path(str(reuse)) / "images"
            if not image_dir.is_dir():
                missing.append(
                    MissingItem(
                        "convert",
                        "directory",
                        str(image_dir),
                        "convert with reuse_images_from requires an images directory",
                    )
                )

    for step in steps:
        for kind, path, message in required_paths(cfg, step):
            if path in virtually_available:
                continue
            if kind == "directory":
                exists = path.is_dir()
            else:
                exists = path.is_file()
            if not exists:
                missing.append(MissingItem(step, kind, str(path), message))
        virtually_available.update(produced_paths(cfg, step))

    return missing


def cfg_options(cfg: PipelineConfig) -> list[str]:
    d = cfg.derived
    roi_range = d["roi_range"]
    roi_size = d["roi_size"]
    pc_range = d["pc_range"]
    train = d["maptracker_train"]
    val = d["maptracker_val"]
    return [
        f"roi_range='{roi_range}'",
        f"roi_size='{roi_size}'",
        f"pc_range='{pc_range}'",
        f"model.roi_range='{roi_range}'",
        f"model.roi_size='{roi_size}'",
        f"model.backbone_cfg.roi_range='{roi_range}'",
        f"model.backbone_cfg.roi_size='{roi_size}'",
        f"model.backbone_cfg.transformer.encoder.pc_range='{pc_range}'",
        f"model.head_cfg.roi_range='{roi_range}'",
        f"model.head_cfg.roi_size='{roi_size}'",
        f"eval_config.ann_file={val}",
        f"eval_config.roi_range='{roi_range}'",
        f"eval_config.roi_size='{roi_size}'",
        f"match_config.ann_file={val}",
        f"match_config.roi_range='{roi_range}'",
        f"match_config.roi_size='{roi_size}'",
        f"data.train.ann_file={train}",
        f"data.train.roi_range='{roi_range}'",
        f"data.train.roi_size='{roi_size}'",
        f"data.val.ann_file={val}",
        f"data.val.roi_range='{roi_range}'",
        f"data.val.roi_size='{roi_size}'",
        f"data.val.eval_config.ann_file={val}",
        f"data.val.eval_config.roi_range='{roi_range}'",
        f"data.val.eval_config.roi_size='{roi_size}'",
        f"data.test.ann_file={val}",
        f"data.test.roi_range='{roi_range}'",
        f"data.test.roi_size='{roi_size}'",
        f"data.test.eval_config.ann_file={val}",
        f"data.test.eval_config.roi_range='{roi_range}'",
        f"data.test.eval_config.roi_size='{roi_size}'",
        f"data.train.pipeline.0.roi_range='{roi_range}'",
        f"data.train.pipeline.0.roi_size='{roi_size}'",
        f"data.train.pipeline.1.roi_range='{roi_range}'",
        f"data.train.pipeline.1.roi_size='{roi_size}'",
        f"eval_config.pipeline.0.roi_range='{roi_range}'",
        f"eval_config.pipeline.0.roi_size='{roi_size}'",
        f"match_config.pipeline.0.roi_range='{roi_range}'",
        f"match_config.pipeline.0.roi_size='{roi_size}'",
        f"match_config.pipeline.1.roi_range='{roi_range}'",
        f"match_config.pipeline.1.roi_size='{roi_size}'",
    ]


def cfg_override_dict(cfg: PipelineConfig, stage_num: str) -> dict[str, Any]:
    d = cfg.derived
    roi_range = [cfg.roi["x_min"], cfg.roi["y_min"], cfg.roi["x_max"], cfg.roi["y_max"]]
    roi_size = [cfg.roi["x_max"] - cfg.roi["x_min"], cfg.roi["y_max"] - cfg.roi["y_min"]]
    pc_range = [cfg.roi["x_min"], cfg.roi["y_min"], -3, cfg.roi["x_max"], cfg.roi["y_max"], 5]
    train = str(d["maptracker_train"])
    val = str(d["maptracker_val"])
    overrides: dict[str, Any] = {
        "roi_range": tuple(roi_range),
        "roi_size": tuple(roi_size),
        "pc_range": pc_range,
        "model.roi_range": tuple(roi_range),
        "model.roi_size": tuple(roi_size),
        "model.backbone_cfg.roi_range": tuple(roi_range),
        "model.backbone_cfg.roi_size": tuple(roi_size),
        "model.backbone_cfg.transformer.encoder.pc_range": pc_range,
        "model.head_cfg.roi_range": tuple(roi_range),
        "model.head_cfg.roi_size": tuple(roi_size),
        "eval_config.ann_file": val,
        "eval_config.roi_range": tuple(roi_range),
        "eval_config.roi_size": tuple(roi_size),
        "match_config.ann_file": val,
        "match_config.roi_range": tuple(roi_range),
        "match_config.roi_size": tuple(roi_size),
        "data.train.ann_file": train,
        "data.train.roi_range": tuple(roi_range),
        "data.train.roi_size": tuple(roi_size),
        "data.val.ann_file": val,
        "data.val.roi_range": tuple(roi_range),
        "data.val.roi_size": tuple(roi_size),
        "data.val.eval_config.ann_file": val,
        "data.val.eval_config.roi_range": tuple(roi_range),
        "data.val.eval_config.roi_size": tuple(roi_size),
        "data.test.ann_file": val,
        "data.test.roi_range": tuple(roi_range),
        "data.test.roi_size": tuple(roi_size),
        "data.test.eval_config.ann_file": val,
        "data.test.eval_config.roi_range": tuple(roi_range),
        "data.test.eval_config.roi_size": tuple(roi_size),
        "data.train.pipeline.0.roi_range": tuple(roi_range),
        "data.train.pipeline.0.roi_size": tuple(roi_size),
        "data.train.pipeline.1.roi_range": tuple(roi_range),
        "data.train.pipeline.1.roi_size": tuple(roi_size),
        "eval_config.pipeline.0.roi_range": tuple(roi_range),
        "eval_config.pipeline.0.roi_size": tuple(roi_size),
        "match_config.pipeline.0.roi_range": tuple(roi_range),
        "match_config.pipeline.0.roi_size": tuple(roi_size),
        "match_config.pipeline.1.roi_range": tuple(roi_range),
        "match_config.pipeline.1.roi_size": tuple(roi_size),
    }
    if stage_num == "2":
        overrides["load_from"] = str(d["stage1_checkpoint"])
    elif stage_num == "3":
        overrides["load_from"] = str(d["stage2_checkpoint"])
    return overrides


def materialize_configs(cfg: PipelineConfig) -> list[Path]:
    if not generated_configs_enabled(cfg):
        return []

    from mmcv import Config

    output_paths = {
        "1": cfg.derived["generated_stage1_config"],
        "2": cfg.derived["generated_stage2_config"],
        "3": cfg.derived["generated_stage3_config"],
    }
    base_paths = {
        "1": cfg.stage1_config,
        "2": cfg.stage2_config,
        "3": cfg.stage3_config,
    }
    overwrite = bool(cfg.generated_configs.get("overwrite", True))

    generated = []
    for stage_num in ["1", "2", "3"]:
        out_path = output_paths[stage_num]
        if out_path.exists() and not overwrite:
            generated.append(out_path)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stage_cfg = Config.fromfile(str(base_paths[stage_num]))
        stage_cfg.merge_from_dict(cfg_override_dict(cfg, stage_num))
        stage_cfg.dump(str(out_path))
        generated.append(out_path)
    return generated


def shell_join(parts: list[str]) -> str:
    return " ".join(shell_quote(part) for part in parts)


def shell_quote(part: Any) -> str:
    text = str(part)
    # Keep mmcv DictAction list overrides readable in dry-run output while
    # still passing them as a single shell word.
    if "='" in text and text.endswith("'") and " " not in text:
        return text
    return shlex.quote(text)


def command_with_env(cfg: PipelineConfig, env_name: str, body: str) -> str:
    conda_home = cfg.paths["conda_home"]
    pythonpath = (
        f"{REPO_ROOT}:{REPO_ROOT / 'MapTR/mmdetection3d'}:"
        f"{WORKSPACE_ROOT / 'MapTR/mmdetection3d'}:${{PYTHONPATH:-}}"
    )
    return (
        f"source {shlex.quote(str(conda_home))}/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(str(env_name))} && "
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"export PYTHONPATH={shlex.quote(pythonpath)} && "
        f"{body}"
    )


def command_specs(cfg: PipelineConfig, steps: list[str]) -> list[CommandSpec]:
    d = cfg.derived
    options = cfg_options(cfg)
    specs: list[CommandSpec] = []
    num_gpus = cfg.runtime.get("num_gpus") or len(str(cfg.runtime["gpus"]).split(","))
    gpus = str(cfg.runtime["gpus"])

    for step in steps:
        if step == "generate_configs":
            if not generated_configs_enabled(cfg):
                continue
            if cfg.config_file is None:
                raise ValueError("generate_configs requires --config to be a real file path")
            args = [
                "python",
                "tools/waymo_pipeline.py",
                "--config",
                cfg.config_file,
                "--materialize-configs",
            ]
            specs.append(
                CommandSpec(
                    step,
                    command_with_env(cfg, cfg.paths["train_env"], shell_join(args)),
                    cfg.paths["train_env"],
                )
            )
        elif step == "convert":
            args = [
                "python",
                "tools/data_converter/waymo_map_converter.py",
                "--data-dir",
                cfg.paths["waymo_train_dir"],
                "--out-dir",
                cfg.paths["processed_dir"],
                "--x-min",
                cfg.roi["x_min"],
                "--y-min",
                cfg.roi["y_min"],
                "--x-max",
                cfg.roi["x_max"],
                "--y-max",
                cfg.roi["y_max"],
                "--num-workers",
                cfg.convert["num_workers"],
                "--frame-stride",
                cfg.convert["frame_stride"],
                "--num-points",
                cfg.convert["num_points"],
            ]
            if cfg.paths.get("waymo_val_dir"):
                args.extend(["--val-data-dir", cfg.paths["waymo_val_dir"]])
            if cfg.paths.get("reuse_images_from"):
                args.extend(["--reuse-images-from", cfg.paths["reuse_images_from"]])
            specs.append(CommandSpec(step, command_with_env(cfg, cfg.paths["waymo_env"], shell_join(args)), cfg.paths["waymo_env"]))
        elif step == "pack":
            args = [
                "python",
                "pack_waymo_for_maptracker.py",
                "--v3-dir",
                cfg.paths["processed_dir"],
                "--out-dir",
                cfg.paths["maptracker_dir"],
            ]
            specs.append(CommandSpec(step, command_with_env(cfg, cfg.paths["train_env"], shell_join(args)), cfg.paths["train_env"]))
        elif step == "gt_tracks":
            stage1_config = active_stage_config(cfg, "1")
            args = [
                "python",
                "tools/tracking/prepare_gt_tracks.py",
                rel_config(stage1_config),
                "--out-dir",
                Path(str(cfg.paths["maptracker_dir"])) / "track_visualization",
            ]
            if not generated_configs_enabled(cfg):
                args.extend(["--cfg-options", *options])
            specs.append(CommandSpec(step, command_with_env(cfg, cfg.paths["train_env"], shell_join(args)), cfg.paths["train_env"]))
        elif step.startswith("train_stage"):
            stage_num = step[-1]
            config = active_stage_config(cfg, stage_num)
            work_dir = {"1": d["stage1_work"], "2": d["stage2_work"], "3": d["stage3_work"]}[stage_num]
            port = {"1": "29511", "2": "29513", "3": "29514"}[stage_num]
            stage_options = list(options)
            if stage_num == "2":
                stage_options.append(f"load_from={d['stage1_checkpoint']}")
            elif stage_num == "3":
                stage_options.append(f"load_from={d['stage2_checkpoint']}")
            args = [
                "bash",
                "tools/dist_train.sh",
                rel_config(config),
                num_gpus,
                "--work-dir",
                work_dir,
            ]
            if not generated_configs_enabled(cfg):
                args.extend(["--cfg-options", *stage_options])
            body = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpus)} PORT={port} {shell_join(args)}"
            specs.append(CommandSpec(step, command_with_env(cfg, cfg.paths["train_env"], body), cfg.paths["train_env"]))
        elif step == "test":
            args = [
                "python",
                "tools/test.py",
                rel_config(active_stage_config(cfg, "3")),
                d["stage3_checkpoint"],
                "--eval",
                "--work-dir",
                d["test_work_dir"],
            ]
            if not generated_configs_enabled(cfg):
                args.extend(["--cfg-options", *options])
            body = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpus)} {shell_join(args)}"
            specs.append(CommandSpec(step, command_with_env(cfg, cfg.paths["train_env"], body), cfg.paths["train_env"]))
        elif step == "visualize":
            scene_args: list[Any] = []
            if cfg.visualize.get("scene_ids"):
                scene_args = ["--scene_id", *cfg.visualize["scene_ids"]]
            pred_args = [
                "python",
                "tools/visualization/vis_global.py",
                rel_config(active_stage_config(cfg, "3")),
                "--data_path",
                d["test_work_dir"] / "pos_predictions.pkl",
                "--out_dir",
                d["vis_out_dir"] / "pred",
                "--option",
                "vis-pred",
                "--per_frame_result",
                cfg.visualize["per_frame_result"],
                "--overwrite",
                cfg.visualize["overwrite"],
                *scene_args,
            ]
            if not generated_configs_enabled(cfg):
                pred_args.extend(["--cfg-options", *options])
            gt_args = [
                "python",
                "tools/visualization/vis_global.py",
                rel_config(active_stage_config(cfg, "3")),
                "--data_path",
                d["maptracker_val_tracks"],
                "--out_dir",
                d["vis_out_dir"] / "gt",
                "--option",
                "vis-gt",
                "--per_frame_result",
                cfg.visualize["per_frame_result"],
                "--overwrite",
                cfg.visualize["overwrite"],
                *scene_args,
            ]
            if not generated_configs_enabled(cfg):
                gt_args.extend(["--cfg-options", *options])
            specs.append(CommandSpec("visualize_pred", command_with_env(cfg, cfg.paths["train_env"], shell_join(pred_args)), cfg.paths["train_env"]))
            specs.append(CommandSpec("visualize_gt", command_with_env(cfg, cfg.paths["train_env"], shell_join(gt_args)), cfg.paths["train_env"]))
    return specs


def report_payload(status: str, steps: list[str], missing: list[MissingItem], commands: list[CommandSpec]) -> dict[str, Any]:
    return {
        "status": status,
        "requested_steps": steps,
        "missing": [item.__dict__ for item in missing],
        "commands": [item.__dict__ for item in commands],
    }


def print_text_report(payload: dict[str, Any]) -> None:
    print(f"STATUS={payload['status']}")
    print(f"REQUESTED_STEPS={','.join(payload['requested_steps'])}")
    if payload["missing"]:
        print("MISSING:")
        for item in payload["missing"]:
            print(f"- [{item['step']}] {item['message']}: {item['path']}")
    if payload["commands"]:
        print("COMMANDS:")
        for command in payload["commands"]:
            print(f"+ {command['command']}")


def run_commands(commands: list[CommandSpec]) -> int:
    for command in commands:
        print(f"+ {command.command}", flush=True)
        result = subprocess.run(["bash", "-lc", command.command])
        if result.returncode != 0:
            return result.returncode
    return 0


def main() -> int:
    args = parse_args()
    try:
        config_file = Path(args.config).resolve()
        cfg = build_config(load_config(config_file), config_file=config_file)
        if args.materialize_configs:
            generated = materialize_configs(cfg)
            payload = {
                "status": "generated",
                "requested_steps": ["generate_configs"],
                "missing": [],
                "commands": [],
                "generated_configs": [str(path) for path in generated],
            }
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                for path in generated:
                    print(path)
            return 0
        steps = expand_steps(args.steps)
    except Exception as exc:
        payload = {"status": "error", "requested_steps": [], "missing": [], "commands": [], "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    missing = precheck(cfg, steps)
    commands = [] if missing else command_specs(cfg, steps)
    status = "blocked" if missing else "ready"
    payload = report_payload(status, steps, missing, commands)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_text_report(payload)

    if missing:
        return 2
    if args.check_only or args.dry_run or cfg.runtime.get("dry_run"):
        return 0
    return run_commands(commands)


if __name__ == "__main__":
    raise SystemExit(main())
