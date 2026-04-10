#!/usr/bin/env bash
#
# Batch runner for text-alignment fine-tuning experiments.
# Usage:  bash code/run_experiments.sh [exp_name]
#   exp_name: exp2 | exp3 | exp4 | exp5 | all
#   If omitted, prints usage info.
#
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT_PATH="${ROOT_PATH:-../DreamDiffusion/}"
BASE="--root_path ${ROOT_PATH}"

# ── Experiment definitions ────────────────────────────────────────

EXP2_DESC="Unfreeze 8 blocks, higher encoder LR"
EXP2_ARGS="${BASE} \
    --num_unfreeze_blocks 8 \
    --lr_encoder 5e-5 \
    --num_epoch 80"

EXP3_DESC="exp2 + rebalanced loss weights (text emphasis)"
EXP3_ARGS="${BASE} \
    --num_unfreeze_blocks 8 \
    --lr_encoder 5e-5 \
    --lambda_vis 300.0 \
    --lambda_txt 100.0 \
    --lambda_cons 200.0 \
    --num_epoch 80"

EXP4_DESC="exp3 + low dropout + large batch"
EXP4_ARGS="${BASE} \
    --num_unfreeze_blocks 8 \
    --lr_encoder 5e-5 \
    --lambda_vis 300.0 \
    --lambda_txt 100.0 \
    --lambda_cons 200.0 \
    --proj_dropout 0.2 \
    --batch_size 64 \
    --num_epoch 100"

EXP5_DESC="Strategy B: joint dim_mapper/channel_mapper fine-tuning"
EXP5_ARGS="${BASE} \
    --num_unfreeze_blocks 8 \
    --lr_encoder 5e-5 \
    --lambda_vis 800.0 \
    --lambda_txt 1.0 \
    --lambda_cons 500.0 \
    --proj_dropout 0.2 \
    --batch_size 64 \
    --num_epoch 100 \
    --use_conditioning_mapper"

# ── Runner ────────────────────────────────────────────────────────

run_exp() {
    local name="$1"
    local desc="$2"
    local args="$3"
    echo "============================================================"
    echo "  ${name}: ${desc}"
    echo "============================================================"
    echo "python code/stageB_text_align_finetune.py ${args}"
    echo ""
    if [ "${DRY_RUN:-0}" != "1" ]; then
        python code/stageB_text_align_finetune.py ${args}
    fi
}

target="${1:-help}"
case "$target" in
    exp2) run_exp exp2 "$EXP2_DESC" "$EXP2_ARGS" ;;
    exp3) run_exp exp3 "$EXP3_DESC" "$EXP3_ARGS" ;;
    exp4) run_exp exp4 "$EXP4_DESC" "$EXP4_ARGS" ;;
    exp5) run_exp exp5 "$EXP5_DESC" "$EXP5_ARGS" ;;
    all)
        run_exp exp2 "$EXP2_DESC" "$EXP2_ARGS"
        run_exp exp3 "$EXP3_DESC" "$EXP3_ARGS"
        run_exp exp4 "$EXP4_DESC" "$EXP4_ARGS"
        run_exp exp5 "$EXP5_DESC" "$EXP5_ARGS"
        ;;
    *)
        echo "Usage: bash code/run_experiments.sh [exp2|exp3|exp4|exp5|all]"
        echo ""
        echo "Set DRY_RUN=1 to print commands without executing."
        echo "Set ROOT_PATH to override the default DreamDiffusion root."
        echo ""
        echo "Available experiments:"
        echo "  exp2: ${EXP2_DESC}"
        echo "  exp3: ${EXP3_DESC}"
        echo "  exp4: ${EXP4_DESC}"
        echo "  exp5: ${EXP5_DESC}"
        ;;
esac
