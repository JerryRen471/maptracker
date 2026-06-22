#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
WAYMO_ENV="${WAYMO_ENV:-waymo_pack}"
TRAIN_ENV="${TRAIN_ENV:-maptracker}"

STAGE1_CONFIG=""
STAGE2_CONFIG=""
STAGE3_CONFIG=""

WAYMO_DATA_DIR="${WAYMO_DATA_DIR:-/data8012/waymo/training}"
WAYMO_VAL_DATA_DIR="${WAYMO_VAL_DATA_DIR:-/data8012/waymo/validation}"
PROCESSED_DIR="${PROCESSED_DIR:-/data/waymo_processed_xm15_x45_y15}"
MAPTRACKER_DIR="${MAPTRACKER_DIR:-/data/waymo_maptracker_xm15_x45_y15}"
WORK_ROOT="${WORK_ROOT:-/data/maptr_workspace/work_dirs}"
EXP_TAG="${WAYMO_EXP_TAG:-asym_roi}"

X_MIN="${X_MIN:--15}"
Y_MIN="${Y_MIN:--15}"
X_MAX="${X_MAX:-45}"
Y_MAX="${Y_MAX:-15}"
FRAME_STRIDE="${FRAME_STRIDE:-5}"
NUM_POINTS="${NUM_POINTS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REUSE_IMAGES_FROM="${REUSE_IMAGES_FROM:-/data/waymo_processed_v3}"

GPUS="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=""
PORT_STAGE1="${PORT_STAGE1:-29511}"
PORT_STAGE2="${PORT_STAGE2:-29513}"
PORT_STAGE3="${PORT_STAGE3:-29514}"

TEST_WORK_DIR=""
VIS_OUT_DIR=""
VIS_SCENE_ARGS=()
PER_FRAME_RESULT="${PER_FRAME_RESULT:-1}"
OVERWRITE="${OVERWRITE:-1}"

DRY_RUN=0
SKIP_CONVERT=0
SKIP_PACK=0
SKIP_GT_TRACKS=0
SKIP_TRAIN=0
SKIP_TEST=0
SKIP_VIS=0
VISUALIZE_GT_TRACKS=0

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_waymo_pipeline.sh --stage1-config CONFIG [options]

Runs the Waymo MapTracker pipeline:
  1. Convert Waymo TFRecords to processed train/val pkl
  2. Pack processed pkl files into MapTracker format
  3. Prepare GT tracking metadata
  4. Train stage1, stage2, and stage3 sequentially
  5. Run stage3 evaluation/inference
  6. Visualize GT and prediction results with vis_global.py

Required:
  --stage1-config PATH          Stage1 BEV pretrain config.

Common options:
  --stage2-config PATH          Stage2 config. Inferred from stage1 name if omitted.
  --stage3-config PATH          Stage3 config. Inferred from stage1 name if omitted.
  --data-dir PATH               Waymo training TFRecord directory.
  --val-data-dir PATH           Waymo validation TFRecord directory.
  --processed-dir PATH          Converter output directory.
  --maptracker-dir PATH         MapTracker pkl output directory.
  --work-root PATH              Training work_dir root.
  --exp-tag NAME                Suffix appended to each stage work_dir.
  --reuse-images-from PATH      Existing processed image root. Empty string disables reuse.
  --gpus LIST                   CUDA_VISIBLE_DEVICES value, e.g. 0,1,2,3.
  --num-gpus N                  Number of GPUs passed to tools/dist_train.sh.
  --dry-run                     Print commands without running them.

ROI/conversion options:
  --x-min N --y-min N --x-max N --y-max N
  --frame-stride N --num-points N --num-workers N

Selective execution:
  --skip-convert --skip-pack --skip-gt-tracks --skip-train --skip-test --skip-vis

Visualization:
  --vis-out-dir PATH            Defaults to <stage3_work_dir>/visualization.
  --scene-id SCENE              Can be repeated.
  --per-frame-result 0|1
  --overwrite 0|1
  --visualize-gt-tracks         Pass --visualize to prepare_gt_tracks.py.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

quote_cmd() {
    printf '%q ' "$@"
}

run_shell() {
    local env_name="$1"
    local body="$2"
    local cmd
    cmd="source \"$CONDA_HOME/etc/profile.d/conda.sh\" && conda activate \"$env_name\" && cd \"$REPO_ROOT\" && export PYTHONPATH=\"$REPO_ROOT:$REPO_ROOT/MapTR/mmdetection3d:$WORKSPACE_ROOT/MapTR/mmdetection3d:\${PYTHONPATH:-}\" && $body"
    echo "+ $cmd"
    if [[ "$DRY_RUN" == "0" ]]; then
        bash -lc "$cmd"
    fi
}

resolve_config() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$REPO_ROOT/$path"
    fi
}

rel_config() {
    local abs
    abs="$(resolve_config "$1")"
    if [[ "$abs" == "$REPO_ROOT/"* ]]; then
        echo "${abs#"$REPO_ROOT/"}"
    else
        echo "$abs"
    fi
}

infer_config() {
    local stage1="$1"
    local target="$2"
    case "$target" in
        stage2)
            echo "${stage1/stage1_bev_pretrain/stage2_warmup}"
            ;;
        stage3)
            echo "${stage1/stage1_bev_pretrain/stage3_joint_finetune}"
            ;;
        *)
            die "invalid stage target: $target"
            ;;
    esac
}

stage_work_dir() {
    local config="$1"
    local base
    base="$(basename "$config" .py)"
    echo "$WORK_ROOT/${base}_${EXP_TAG}"
}

count_gpus() {
    local value="$1"
    if [[ -z "$value" ]]; then
        echo 0
        return
    fi
    awk -F',' '{print NF}' <<<"$value"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stage1-config) STAGE1_CONFIG="$2"; shift 2 ;;
            --stage2-config) STAGE2_CONFIG="$2"; shift 2 ;;
            --stage3-config) STAGE3_CONFIG="$2"; shift 2 ;;
            --data-dir) WAYMO_DATA_DIR="$2"; shift 2 ;;
            --val-data-dir) WAYMO_VAL_DATA_DIR="$2"; shift 2 ;;
            --processed-dir) PROCESSED_DIR="$2"; shift 2 ;;
            --maptracker-dir) MAPTRACKER_DIR="$2"; shift 2 ;;
            --work-root) WORK_ROOT="$2"; shift 2 ;;
            --exp-tag) EXP_TAG="$2"; shift 2 ;;
            --reuse-images-from) REUSE_IMAGES_FROM="$2"; shift 2 ;;
            --gpus) GPUS="$2"; shift 2 ;;
            --num-gpus) NUM_GPUS="$2"; shift 2 ;;
            --x-min) X_MIN="$2"; shift 2 ;;
            --y-min) Y_MIN="$2"; shift 2 ;;
            --x-max) X_MAX="$2"; shift 2 ;;
            --y-max) Y_MAX="$2"; shift 2 ;;
            --frame-stride) FRAME_STRIDE="$2"; shift 2 ;;
            --num-points) NUM_POINTS="$2"; shift 2 ;;
            --num-workers) NUM_WORKERS="$2"; shift 2 ;;
            --test-work-dir) TEST_WORK_DIR="$2"; shift 2 ;;
            --vis-out-dir) VIS_OUT_DIR="$2"; shift 2 ;;
            --scene-id) VIS_SCENE_ARGS+=("$2"); shift 2 ;;
            --per-frame-result) PER_FRAME_RESULT="$2"; shift 2 ;;
            --overwrite) OVERWRITE="$2"; shift 2 ;;
            --conda-home) CONDA_HOME="$2"; shift 2 ;;
            --waymo-env) WAYMO_ENV="$2"; shift 2 ;;
            --train-env) TRAIN_ENV="$2"; shift 2 ;;
            --dry-run) DRY_RUN=1; shift ;;
            --skip-convert) SKIP_CONVERT=1; shift ;;
            --skip-pack) SKIP_PACK=1; shift ;;
            --skip-gt-tracks) SKIP_GT_TRACKS=1; shift ;;
            --skip-train) SKIP_TRAIN=1; shift ;;
            --skip-test) SKIP_TEST=1; shift ;;
            --skip-vis) SKIP_VIS=1; shift ;;
            --visualize-gt-tracks) VISUALIZE_GT_TRACKS=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
}

validate_args() {
    [[ -n "$STAGE1_CONFIG" ]] || die "--stage1-config is required"
    STAGE1_CONFIG="$(rel_config "$STAGE1_CONFIG")"
    [[ -n "$STAGE2_CONFIG" ]] || STAGE2_CONFIG="$(infer_config "$STAGE1_CONFIG" stage2)"
    [[ -n "$STAGE3_CONFIG" ]] || STAGE3_CONFIG="$(infer_config "$STAGE1_CONFIG" stage3)"
    STAGE2_CONFIG="$(rel_config "$STAGE2_CONFIG")"
    STAGE3_CONFIG="$(rel_config "$STAGE3_CONFIG")"

    [[ -f "$REPO_ROOT/$STAGE1_CONFIG" ]] || die "missing stage1 config: $STAGE1_CONFIG"
    [[ -f "$REPO_ROOT/$STAGE2_CONFIG" ]] || die "missing stage2 config: $STAGE2_CONFIG"
    [[ -f "$REPO_ROOT/$STAGE3_CONFIG" ]] || die "missing stage3 config: $STAGE3_CONFIG"

    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS="$(count_gpus "$GPUS")"
    fi
    [[ "$NUM_GPUS" -gt 0 ]] || die "--num-gpus must be positive"

    local stage3_work
    stage3_work="$(stage_work_dir "$STAGE3_CONFIG")"
    [[ -n "$TEST_WORK_DIR" ]] || TEST_WORK_DIR="$stage3_work/eval"
    [[ -n "$VIS_OUT_DIR" ]] || VIS_OUT_DIR="$stage3_work/visualization"
}

print_summary() {
    cat <<EOF
PIPELINE=waymo_maptracker
DRY_RUN=$DRY_RUN
STAGE1_CONFIG=$STAGE1_CONFIG
STAGE2_CONFIG=$STAGE2_CONFIG
STAGE3_CONFIG=$STAGE3_CONFIG
PROCESSED_DIR=$PROCESSED_DIR
MAPTRACKER_DIR=$MAPTRACKER_DIR
WORK_ROOT=$WORK_ROOT
EXP_TAG=$EXP_TAG
ROI=[$X_MIN,$Y_MIN,$X_MAX,$Y_MAX]
GPUS=$GPUS
NUM_GPUS=$NUM_GPUS
NOTE=Config ann_file paths are not rewritten; keep config data paths aligned with MAPTRACKER_DIR.
EOF
}

convert_data() {
    [[ "$SKIP_CONVERT" == "0" ]] || return 0
    local cmd
    cmd="$(quote_cmd python tools/data_converter/waymo_map_converter.py \
        --data-dir "$WAYMO_DATA_DIR" \
        --out-dir "$PROCESSED_DIR" \
        --x-min "$X_MIN" --y-min "$Y_MIN" --x-max "$X_MAX" --y-max "$Y_MAX" \
        --num-workers "$NUM_WORKERS" \
        --frame-stride "$FRAME_STRIDE" \
        --num-points "$NUM_POINTS")"
    if [[ -n "$WAYMO_VAL_DATA_DIR" ]]; then
        cmd="$cmd $(quote_cmd --val-data-dir "$WAYMO_VAL_DATA_DIR")"
    fi
    if [[ -n "$REUSE_IMAGES_FROM" ]]; then
        cmd="$cmd $(quote_cmd --reuse-images-from "$REUSE_IMAGES_FROM")"
    fi
    run_shell "$WAYMO_ENV" "$cmd"
}

pack_data() {
    [[ "$SKIP_PACK" == "0" ]] || return 0
    local cmd
    cmd="$(quote_cmd python pack_waymo_for_maptracker.py \
        --v3-dir "$PROCESSED_DIR" \
        --out-dir "$MAPTRACKER_DIR")"
    run_shell "$TRAIN_ENV" "$cmd"
}

prepare_gt_tracks() {
    [[ "$SKIP_GT_TRACKS" == "0" ]] || return 0
    local cmd
    cmd="$(quote_cmd python tools/tracking/prepare_gt_tracks.py "$STAGE1_CONFIG" \
        --out-dir "$MAPTRACKER_DIR/track_visualization")"
    if [[ "$VISUALIZE_GT_TRACKS" == "1" ]]; then
        cmd="$cmd --visualize"
    fi
    run_shell "$TRAIN_ENV" "$cmd"
}

train_stage() {
    local stage_name="$1"
    local config="$2"
    local work_dir="$3"
    local port="$4"
    local load_from="${5:-}"
    local cmd
    cmd="CUDA_VISIBLE_DEVICES=$(printf '%q' "$GPUS") PORT=$(printf '%q' "$port") $(quote_cmd bash tools/dist_train.sh "$config" "$NUM_GPUS" --work-dir "$work_dir")"
    if [[ -n "$load_from" ]]; then
        cmd="$cmd $(quote_cmd --cfg-options "load_from=$load_from")"
    fi
    echo "STAGE=$stage_name"
    echo "WORK_DIR=$work_dir"
    run_shell "$TRAIN_ENV" "$cmd"
}

train_all() {
    [[ "$SKIP_TRAIN" == "0" ]] || return 0
    local stage1_work stage2_work stage3_work
    stage1_work="$(stage_work_dir "$STAGE1_CONFIG")"
    stage2_work="$(stage_work_dir "$STAGE2_CONFIG")"
    stage3_work="$(stage_work_dir "$STAGE3_CONFIG")"

    train_stage stage1 "$STAGE1_CONFIG" "$stage1_work" "$PORT_STAGE1"
    train_stage stage2 "$STAGE2_CONFIG" "$stage2_work" "$PORT_STAGE2" "$stage1_work/latest.pth"
    train_stage stage3 "$STAGE3_CONFIG" "$stage3_work" "$PORT_STAGE3" "$stage2_work/latest.pth"
}

run_test() {
    [[ "$SKIP_TEST" == "0" ]] || return 0
    local stage3_work cmd
    stage3_work="$(stage_work_dir "$STAGE3_CONFIG")"
    cmd="CUDA_VISIBLE_DEVICES=$(printf '%q' "$GPUS") $(quote_cmd python tools/test.py "$STAGE3_CONFIG" "$stage3_work/latest.pth" --eval --work-dir "$TEST_WORK_DIR")"
    run_shell "$TRAIN_ENV" "$cmd"
}

visualize_results() {
    [[ "$SKIP_VIS" == "0" ]] || return 0
    local pred_path gt_path scene_args pred_cmd gt_cmd
    pred_path="$TEST_WORK_DIR/pos_predictions.pkl"
    gt_path="$MAPTRACKER_DIR/waymo_map_infos_val_gt_tracks.pkl"
    scene_args=""
    if [[ "${#VIS_SCENE_ARGS[@]}" -gt 0 ]]; then
        scene_args="$(quote_cmd --scene_id "${VIS_SCENE_ARGS[@]}")"
    fi

    pred_cmd="$(quote_cmd python tools/visualization/vis_global.py "$STAGE3_CONFIG" \
        --data_path "$pred_path" \
        --out_dir "$VIS_OUT_DIR/pred" \
        --option vis-pred \
        --per_frame_result "$PER_FRAME_RESULT" \
        --overwrite "$OVERWRITE") $scene_args"
    gt_cmd="$(quote_cmd python tools/visualization/vis_global.py "$STAGE3_CONFIG" \
        --data_path "$gt_path" \
        --out_dir "$VIS_OUT_DIR/gt" \
        --option vis-gt \
        --per_frame_result "$PER_FRAME_RESULT" \
        --overwrite "$OVERWRITE") $scene_args"

    run_shell "$TRAIN_ENV" "$pred_cmd"
    run_shell "$TRAIN_ENV" "$gt_cmd"
}

main() {
    parse_args "$@"
    validate_args
    print_summary
    convert_data
    pack_data
    prepare_gt_tracks
    train_all
    run_test
    visualize_results
    echo "WAYMO_PIPELINE_FINISHED"
}

main "$@"
