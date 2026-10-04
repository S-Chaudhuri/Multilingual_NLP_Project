#!/bin/bash
#SBATCH --job-name=abl_train
#SBATCH --account=gpuuva086
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --output=slurm_logs/abl_train_%A_%a.out
#SBATCH --error=slurm_logs/abl_train_%A_%a.err


# One array task = one (model, k, lang, seed) configuration:
#   1. train a soft prompt with k virtual tokens (pipeline_v1/prompt_tuning_train.py)
#   2. evaluate it on the shared test split against the zero-shot model
#      (pipeline_v1/prompt_tuning_eval.py --compare_zero_shot)
#
# Submit through run_ablation.sh, which sets the array size and the grid.

set -eo pipefail

cd "$HOME/Multilingual_NLP_Project"

PYTHON="${PYTHON:-$HOME/.conda/envs/multilingual_nlp/bin/python}"
MODELS="${MODELS:-mbert_base,xlmr_base}"
KS="${KS:-5,10,20}"
LANGS="${LANGS:-en,nl,ru,bn,sw}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SPLIT_FILE="${SPLIT_FILE:-splits/shared_en_nl_ru_bn_sw_seed42.json}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/ablation}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation}"

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra K_LIST <<< "$KS"
IFS=',' read -ra LANG_LIST <<< "$LANGS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

# Decode the array index: model is the slowest-varying, seed the fastest.
i=${SLURM_ARRAY_TASK_ID:?must run as an array job}
n_seed=${#SEED_LIST[@]}; n_lang=${#LANG_LIST[@]}; n_k=${#K_LIST[@]}
SEED=${SEED_LIST[$(( i % n_seed ))]};                       i=$(( i / n_seed ))
LANG=${LANG_LIST[$(( i % n_lang ))]};                       i=$(( i / n_lang ))
K=${K_LIST[$(( i % n_k ))]};                                i=$(( i / n_k ))
MODEL=${MODEL_LIST[$i]:?array index out of range}

case "$MODEL" in
    mbert_base) MODEL_NAME=bert-base-multilingual-cased ;;
    xlmr_base)  MODEL_NAME=xlm-roberta-base ;;
    *) echo "Unknown model key: $MODEL" >&2; exit 1 ;;
esac

CKPT_DIR="$CKPT_ROOT/$MODEL/$LANG/P${K}_seed${SEED}"
RESULT_DIR="$RESULTS_ROOT/$MODEL/$LANG"
mkdir -p slurm_logs "$CKPT_DIR" "$RESULT_DIR"

echo "=========================================="
echo "Prompt-length ablation"
echo "Model: $MODEL ($MODEL_NAME)"
echo "k:     $K"
echo "Lang:  $LANG"
echo "Seed:  $SEED"
echo "=========================================="
echo "Host: $(hostname)  Date: $(date)"
echo "Task: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo
echo "Training..."
"$PYTHON" pipeline_v1/prompt_tuning_train.py \
    --lang "$LANG" \
    --model_name "$MODEL_NAME" \
    --num_prompt_tokens "$K" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --split_file "$SPLIT_FILE" \
    --output_dir "$CKPT_DIR" \
    --device cuda

echo
echo "Evaluating on the test split..."
"$PYTHON" pipeline_v1/prompt_tuning_eval.py \
    --lang "$LANG" \
    --model_name "$MODEL_NAME" \
    --soft_prompt_path "$CKPT_DIR/soft_prompt_${LANG}_P${K}_best.pt" \
    --num_prompt_tokens "$K" \
    --split_file "$SPLIT_FILE" \
    --split test \
    --compare_zero_shot \
    --output "$RESULT_DIR/eval_P${K}_seed${SEED}.json" \
    --device cuda

echo
echo "Done: $(date)"
