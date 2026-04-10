#!/usr/bin/env bash
#
# Batch runner for text-alignment fine-tuning experiments.
# Usage:  bash code/run_experiments.sh [exp_name]
#   exp_name: exp2 | exp3 | exp4 | exp5 | all
#   If omitted, prints the command without executing.
#
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT_PATH="${ROOT_PATH:-../DreamDiffusion/}"

# ── Common base args ──────────────────────────────────────────────
BASE_ARGS="--root_path ${ROOT_PATH}"

# ── Experiment definitions ────────────────────────────────────────
#
# exp2: unlock more encoder capacity
#   - 8 unfrozen blocks (vs 4), higher encoder LR
#
declare -A EXP2=(
    [desc]="Unfreeze 8 blocks, higher encoder LR"
    [args]="${BASE_ARGS} \
        --num_unfreeze_blocks 8 \
        --lr_encoder 5e-5 \
        --num_epoch 80"
)

# exp3: rebalance loss weights to emphasise text alignment
#   - builds on exp2 settings
#
declare -A EXP3=(
    [desc]="exp2 + rebalanced loss weights (text emphasis)"
    [args]="${BASE_ARGS} \
        --num_unfreeze_blocks 8 \
        --lr_encoder 5e-5 \
        --lambda_vis 300.0 \
        --lambda_txt 100.0 \
        --lambda_cons 200.0 \
        --num_epoch 80"
)

# exp4: further tuning - lower dropout, larger batch for InfoNCE
#   - builds on exp3 settings
#
declare -A EXP4=(
    [desc]="exp3 + low dropout + large batch"
    [args]="${BASE_ARGS} \
        --num_unfreeze_blocks 8 \
        --lr_encoder 5e-5 \
        --lambda_vis 300.0 \
        --lambda_txt 100.0 \
        --lambda_cons 200.0 \
        --proj_dropout 0.2 \
        --batch_size 64 \
        --num_epoch 100"
)

# exp5: Strategy B - joint dim_mapper fine-tuning (requires code changes)
#   - uses --use_conditioning_mapper flag
#
declare -A EXP5=(
    [desc]="Strategy B: joint dim_mapper/channel_mapper fine-tuning"
    [args]="${BASE_ARGS} \
        --num_unfreeze_blocks 8 \
        --lr_encoder 5e-5 \
        --lambda_vis 800.0 \
        --lambda_txt 1.0 \
        --lambda_cons 500.0 \
        --proj_dropout 0.2 \
        --batch_size 64 \
        --num_epoch 100 \
        --use_conditioning_mapper"
)

# ── Runner ────────────────────────────────────────────────────────
run_exp() {
    local name=$1
    local -n exp_ref=$2
    echo "============================================================"
    echo "  ${name}: ${exp_ref[desc]}"
    echo "============================================================"
    echo "python code/stageB_text_align_finetune.py ${exp_ref[args]}"
    echo ""
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        python code/stageB_text_align_finetune.py ${exp_ref[args]}
    fi
}

target="${1:-help}"
case "$target" in
    exp2)  run_exp exp2 EXP2 ;;
    exp3)  run_exp exp3 EXP3 ;;
    exp4)  run_exp exp4 EXP4 ;;
    exp5)  run_exp exp5 EXP5 ;;
    all)
        for e in EXP2 EXP3 EXP4 EXP5; do
            run_exp "${e,,}" "$e"
        done
        ;;
    *)
        echo "Usage: bash code/run_experiments.sh [exp2|exp3|exp4|exp5|all]"
        echo ""
        echo "Set DRY_RUN=1 to print commands without executing."
        echo "Set ROOT_PATH to override the default DreamDiffusion root."
        echo ""
        echo "Available experiments:"
        echo "  exp2: ${EXP2[desc]}"
        echo "  exp3: ${EXP3[desc]}"
        echo "  exp4: ${EXP4[desc]}"
        echo "  exp5: ${EXP5[desc]}"
        ;;
esac
