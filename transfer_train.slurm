#!/bin/bash
#SBATCH --job-name=transfer
#SBATCH --account=gpuuva086
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --output=pipeline_v1/results/slurm_logs/transfer_%A_%a.out
#SBATCH --error=pipeline_v1/results/slurm_logs/transfer_%A_%a.err


# Multilingual transfer-scaling experiments (fixed fact budget).
#
# One array task = one (model, run, seed) configuration:
#   1. jointly train one soft prompt on the run's source languages with the
#      budget partition splits/budget/<run>_B<B>_seed<seed>.json
#      (pipeline_v1/multilingual_train.py)
#   2. evaluate it on the shared test split of all five languages
#      (pipeline_v1/transfer_eval.py; alpha selected on source-language val)
#
# Submit through run_transfer_grid.sh, which creates the budget partitions,
# sets the array size and exports the grid.

set -eo pipefail

cd "$HOME/Multilingual_NLP_Project"

PYTHON="${PYTHON:-$HOME/.conda/envs/multilingual_nlp/bin/python}"
MODELS="${MODELS:-mbert_base,xlmr_base}"
# Runs are separated by ';', languages inside a run by ','.
RUNS="${RUNS:-en;en,nl;en,ru;en,nl,ru,el;en,ru,el,ko}"
SEEDS="${SEEDS:-42,43,44}"
EVAL_LANGS="${EVAL_LANGS:-en,nl,ru,el,ko}"
BUDGET="${BUDGET:-8928}"
SPLIT_FILE="${SPLIT_FILE:-splits/shared_en_nl_ru_el_ko_rulc_seed42.json}"
BUDGET_DIR="${BUDGET_DIR:-splits/budget}"
CONFIG="${CONFIG:-pipeline_v1/configs/transfer_config.json}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/transfer}"
RESULTS_ROOT="${RESULTS_ROOT:-pipeline_v1/results/transfer}"
# Extra flags, e.g. EXTRA_FLAGS=--allow_incomplete_split for XLM-R if some
# split facts are dropped by its tokenizer.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
# The split was built with lowercased Russian inflection (scripts/prompt.py).
export XFACTR_RU_LOWERCASE="${XFACTR_RU_LOWERCASE:-1}"
export PYTHONUTF8=1

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=';' read -ra RUN_LIST <<< "$RUNS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

# Decode the array index: model is the slowest-varying, seed the fastest.
i=${SLURM_ARRAY_TASK_ID:?must run as an array job}
n_seed=${#SEED_LIST[@]}; n_run=${#RUN_LIST[@]}
SEED=${SEED_LIST[$(( i % n_seed ))]};   i=$(( i / n_seed ))
RUN=${RUN_LIST[$(( i % n_run ))]};      i=$(( i / n_run ))
MODEL=${MODEL_LIST[$i]:?array index out of range}

# Per-model training overrides (passed to multilingual_train.py only).
TRAIN_FLAGS=""
case "$MODEL" in
    mbert_base) MODEL_NAME=bert-base-multilingual-cased ;;
    xlmr_base)  MODEL_NAME=xlm-roberta-base;;
    *) echo "Unknown model key: $MODEL" >&2; exit 1 ;;
esac


IFS=',' read -ra SRC_LANGS <<< "$RUN"
IFS=',' read -ra TGT_LANGS <<< "$EVAL_LANGS"
TAG=$(IFS='-'; echo "${SRC_LANGS[*]}")

BUDGET_FILE="$BUDGET_DIR/${TAG}_B${BUDGET}_seed${SEED}.json"
CKPT_DIR="$CKPT_ROOT/$MODEL/$TAG/seed${SEED}"
RESULT_DIR="$RESULTS_ROOT/$MODEL/$TAG/seed${SEED}"
mkdir -p "$CKPT_DIR" "$RESULT_DIR"

if [ ! -f "$BUDGET_FILE" ]; then
    echo "Missing budget partition $BUDGET_FILE (run run_transfer_grid.sh)" >&2
    exit 1
fi

echo "=========================================="
echo "Multilingual transfer (fixed budget B=$BUDGET)"
echo "Model:   $MODEL ($MODEL_NAME)"
echo "Sources: ${SRC_LANGS[*]}"
echo "Eval:    ${TGT_LANGS[*]}"
echo "Seed:    $SEED"
echo "Budget:  $BUDGET_FILE"
echo "=========================================="
echo "Date: $(date)"
echo "Task: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo
echo "Training..."
# shellcheck disable=SC2086
"$PYTHON" pipeline_v1/multilingual_train.py \
    --langs "${SRC_LANGS[@]}" \
    --budget_file "$BUDGET_FILE" \
    --split_file "$SPLIT_FILE" \
    --config "$CONFIG" \
    --model_name "$MODEL_NAME" \
    --seed "$SEED" \
    --output_dir "$CKPT_DIR" \
    --device cuda \
    $TRAIN_FLAGS \
    $EXTRA_FLAGS

echo
echo "Evaluating on the test split of all languages..."
# shellcheck disable=SC2086
"$PYTHON" pipeline_v1/transfer_eval.py \
    --soft_prompt_path "$CKPT_DIR/soft_prompt_${TAG}_P20_best.pt" \
    --source_langs "${SRC_LANGS[@]}" \
    --eval_langs "${TGT_LANGS[@]}" \
    --model_name "$MODEL_NAME" \
    --split_file "$SPLIT_FILE" \
    --alphas "$ALPHAS" \
    --output_dir "$RESULT_DIR" \
    --device cuda \
    $EXTRA_FLAGS

echo
echo "Done: $(date)"
