#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${WAYMO_WORK_ROOT:-/data/maptr_workspace/work_dirs}"
CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-maptracker}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
EXP_TAG="${WAYMO_EXP_TAG:-asym_roi_mapseg_visible_iface}"

usage() {
    cat <<'EOF'
Usage:
  bash tools/train_waymo_asym.sh start stage1|stage2|stage3
  bash tools/train_waymo_asym.sh status stage1|stage2|stage3
  bash tools/train_waymo_asym.sh logs stage1|stage2|stage3
  bash tools/train_waymo_asym.sh chain

Environment overrides:
  CUDA_VISIBLE_DEVICES=4,5,6,7
  NUM_GPUS=4
  WAYMO_WORK_ROOT=/data/maptr_workspace/work_dirs
  CONDA_HOME=/root/miniconda3
  CONDA_ENV=maptracker
EOF
}

stage_config() {
    case "$1" in
        stage1)
            echo "plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain.py"
            ;;
        stage2)
            echo "plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage2_warmup.py"
            ;;
        stage3)
            echo "plugin/configs/maptracker/waymo_5cam/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune.py"
            ;;
        *)
            echo "Unknown stage: $1" >&2
            return 2
            ;;
    esac
}

stage_work_dir() {
    case "$1" in
        stage1)
            echo "$WORK_ROOT/maptracker_waymo_5cam_5frame_span10_stage1_bev_pretrain_${EXP_TAG}"
            ;;
        stage2)
            echo "$WORK_ROOT/maptracker_waymo_5cam_5frame_span10_stage2_warmup_${EXP_TAG}"
            ;;
        stage3)
            echo "$WORK_ROOT/maptracker_waymo_5cam_5frame_span10_stage3_joint_finetune_${EXP_TAG}"
            ;;
        *)
            echo "Unknown stage: $1" >&2
            return 2
            ;;
    esac
}

stage_extra_args() {
    case "$1" in
        stage1)
            ;;
        stage2)
            echo "--cfg-options load_from=$(stage_work_dir stage1)/latest.pth"
            ;;
        stage3)
            echo "--cfg-options load_from=$(stage_work_dir stage2)/latest.pth"
            ;;
        *)
            return 2
            ;;
    esac
}

stage_port() {
    case "$1" in
        stage1) echo 29511 ;;
        stage2) echo 29513 ;;
        stage3) echo 29514 ;;
        *) return 2 ;;
    esac
}

previous_stage() {
    case "$1" in
        stage1) return 1 ;;
        stage2) echo stage1 ;;
        stage3) echo stage2 ;;
        *) return 2 ;;
    esac
}

stage_running() {
    local work_dir
    work_dir="$(stage_work_dir "$1")"
    pgrep -af 'python .*tools/train.py' | grep -Fq -- "$work_dir"
}

stage_succeeded() {
    local work_dir="$1"
    [[ -s "$work_dir/latest.pth" ]] &&
        [[ -f "$work_dir/stdout.log" ]] &&
        grep -q 'worker group successfully finished' "$work_dir/stdout.log"
}

validate_predecessor() {
    local stage="$1"
    local predecessor
    if ! predecessor="$(previous_stage "$stage")"; then
        return 0
    fi

    local predecessor_work
    predecessor_work="$(stage_work_dir "$predecessor")"
    if ! stage_succeeded "$predecessor_work"; then
        echo "$predecessor must finish successfully before $stage." >&2
        echo "Expected checkpoint: $predecessor_work/latest.pth" >&2
        return 1
    fi
}

archive_incomplete_work_dir() {
    local work_dir="$1"
    if [[ ! -d "$work_dir" ]] ||
            ! find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit |
                grep -q .; then
        return 0
    fi

    if stage_succeeded "$work_dir"; then
        echo "Refusing to overwrite completed run: $work_dir" >&2
        return 1
    fi

    local backup="${work_dir}_previous_$(date -u +%Y%m%d_%H%M%SUTC)"
    mv "$work_dir" "$backup"
    echo "Archived incomplete run: $backup"
}

run_stage() {
    local stage="$1"
    local config work_dir port
    config="$(stage_config "$stage")"
    work_dir="$(stage_work_dir "$stage")"
    port="$(stage_port "$stage")"

    set +u
    source "$CONDA_HOME/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/MapTR/mmdetection3d:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    export PORT="$port"
    extra_args="$(stage_extra_args "$stage")"
    exec bash tools/dist_train.sh "$config" "$NUM_GPUS" \
        --work-dir "$work_dir" \
        $extra_args

}

start_stage() {
    local stage="$1"
    local config work_dir port
    config="$(stage_config "$stage")"
    work_dir="$(stage_work_dir "$stage")"
    port="$(stage_port "$stage")"

    if [[ "${WAYMO_TRAIN_DRY_RUN:-0}" == "1" ]]; then
        echo "CONFIG=$config"
        echo "WORK_DIR=$work_dir"
        echo "CUDA_VISIBLE_DEVICES=$GPUS PORT=$port NUM_GPUS=$NUM_GPUS"
        echo "EXTRA_ARGS=$(stage_extra_args "$stage")"
        echo "BACKGROUND_PID=0"
        return 0
    fi


    validate_predecessor "$stage"
    if stage_running "$stage"; then
        echo "$stage is already running." >&2
        status_stage "$stage"
        return 1
    fi
    archive_incomplete_work_dir "$work_dir"

    mkdir -p "$work_dir"
    nohup bash "$0" _run "$stage" \
        > "$work_dir/stdout.log" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$work_dir/launcher.pid"
    echo "BACKGROUND_PID=$pid"
    echo "WORK_DIR=$work_dir"
    echo "LOG=$work_dir/stdout.log"
}

status_stage() {
    local stage="$1"
    local work_dir
    work_dir="$(stage_work_dir "$stage")"
    echo "STAGE=$stage"
    echo "WORK_DIR=$work_dir"

    if stage_running "$stage"; then
        echo "STATUS=RUNNING"
        pgrep -af 'python .*tools/train.py' | grep -F -- "$work_dir"
    elif stage_succeeded "$work_dir"; then
        echo "STATUS=SUCCEEDED"
    elif [[ -d "$work_dir" ]]; then
        echo "STATUS=STOPPED_OR_FAILED"
    else
        echo "STATUS=NOT_STARTED"
    fi

    if [[ -f "$work_dir/stdout.log" ]]; then
        grep -E 'Iter \[[0-9]+/[0-9]+\]|Traceback|ERROR|SUCCEEDED' \
            "$work_dir/stdout.log" | tail -n 5 || true
    fi
    if [[ -s "$work_dir/latest.pth" ]]; then
        echo "CHECKPOINT=$(readlink -f "$work_dir/latest.pth")"
    fi
}

logs_stage() {
    local work_dir
    work_dir="$(stage_work_dir "$1")"
    if [[ ! -f "$work_dir/stdout.log" ]]; then
        echo "Log does not exist: $work_dir/stdout.log" >&2
        return 1
    fi
    exec tail -f "$work_dir/stdout.log"
}

wait_for_stage() {
    local stage="$1"
    local work_dir
    work_dir="$(stage_work_dir "$stage")"
    while stage_running "$stage"; do
        sleep 60
    done
    if ! stage_succeeded "$work_dir"; then
        echo "$stage stopped without a successful checkpoint." >&2
        return 1
    fi
}

chain_worker() {
    local stage
    for stage in stage1 stage2 stage3; do
        local work_dir
        work_dir="$(stage_work_dir "$stage")"
        if stage_succeeded "$work_dir"; then
            echo "$stage already succeeded; skipping."
            continue
        fi
        if ! stage_running "$stage"; then
            start_stage "$stage"
        fi
        wait_for_stage "$stage"
        echo "$stage completed successfully."
    done
}

start_chain() {
    local chain_log="$WORK_ROOT/waymo_asym_training_chain.log"
    local chain_pid_file="$WORK_ROOT/waymo_asym_training_chain.pid"
    if [[ -s "$chain_pid_file" ]] &&
            kill -0 "$(cat "$chain_pid_file")" 2>/dev/null; then
        echo "Chain is already running with PID $(cat "$chain_pid_file")." >&2
        return 1
    fi

    if [[ "${WAYMO_TRAIN_DRY_RUN:-0}" == "1" ]]; then
        echo "CHAIN_LOG=$chain_log"
        echo "BACKGROUND_PID=0"
        return 0
    fi

    mkdir -p "$WORK_ROOT"
    nohup bash "$0" _chain_worker \
        > "$chain_log" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$chain_pid_file"
    echo "BACKGROUND_PID=$pid"
    echo "CHAIN_LOG=$chain_log"
}

command="${1:-help}"
case "$command" in
    start)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        start_stage "$2"
        ;;
    status)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        status_stage "$2"
        ;;
    logs)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        logs_stage "$2"
        ;;
    chain)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        start_chain
        ;;
    _run)
        run_stage "$2"
        ;;
    _chain_worker)
        chain_worker
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
