#!/bin/bash
#SBATCH --job-name=abl_ensemble
#SBATCH --account=gpuuva086
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --output=slurm_logs/abl_ensemble_%A_%a.out
#SBATCH --error=slurm_logs/abl_ensemble_%A_%a.err


# One array task = one (model, lang): ensemble every soft prompt trained for it
# (all k and seeds) plus the zero-shot model, with learnable scalar weights fit
# on the validation split and length-normalised candidate scores
# (pipeline_v1/prompt_ensemble.py).
#
# Submit through run_ablation.sh, which runs this after ablation_train.slurm.

set -eo pipefail

cd "$HOME/Multilingual_NLP_Project"

PYTHON="${PYTHON:-$HOME/.conda/envs/multilingual_nlp/bin/python}"
MODELS="${MODELS:-mbert_base,xlmr_base}"
KS="${KS:-5,10,20}"
LANGS="${LANGS:-en,nl,ru,el,ko}"
# Prompts from these seeds are ensembled; can include seeds trained in an earlier
# run (e.g. train SEEDS=43,44 now, ensemble ENSEMBLE_SEEDS=42,43,44).
SEEDS="${ENSEMBLE_SEEDS:-${SEEDS:-42}}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
SPLIT_FILE="${SPLIT_FILE:-splits/shared_en_nl_ru_el_ko_rulc_seed42.json}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/ablation}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation}"

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra K_LIST <<< "$KS"
IFS=',' read -ra LANG_LIST <<< "$LANGS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

i=${SLURM_ARRAY_TASK_ID:?must run as an array job}
n_lang=${#LANG_LIST[@]}
LANG=${LANG_LIST[$(( i % n_lang ))]}
MODEL=${MODEL_LIST[$(( i / n_lang ))]:?array index out of range}

case "$MODEL" in
    mbert_base) MODEL_NAME=bert-base-multilingual-cased ;;
    xlmr_base)  MODEL_NAME=xlm-roberta-base ;;
    *) echo "Unknown model key: $MODEL" >&2; exit 1 ;;
esac

mkdir -p slurm_logs "$RESULTS_ROOT/$MODEL/$LANG"

# Collect whichever checkpoints exist; a failed training task only removes
# that member instead of blocking the whole ensemble.
PROMPTS=()
NAMES=()
for k in "${K_LIST[@]}"; do
    for s in "${SEED_LIST[@]}"; do
        ckpt="$CKPT_ROOT/$MODEL/$LANG/P${k}_seed${s}/soft_prompt_${LANG}_P${k}_best.pt"
        if [ -f "$ckpt" ]; then
            PROMPTS+=("$ckpt")
            NAMES+=("P${k}_s${s}")
        else
            echo "WARNING: missing checkpoint $ckpt" >&2
        fi
    done
done

if [ ${#PROMPTS[@]} -eq 0 ]; then
    echo "No checkpoints found for $MODEL / $LANG" >&2
    exit 1
fi

echo "=========================================="
echo "Soft-prompt ensembling"
echo "Model:   $MODEL ($MODEL_NAME)"
echo "Lang:    $LANG"
echo "Members: zero_shot ${NAMES[*]}"
echo "Alphas:  $ALPHAS"
echo "=========================================="
echo "Host: $(hostname)  Date: $(date)"

"$PYTHON" pipeline_v1/prompt_ensemble.py \
    --lang "$LANG" \
    --model_name "$MODEL_NAME" \
    --soft_prompts "${PROMPTS[@]}" \
    --names "${NAMES[@]}" \
    --include_zero_shot \
    --split_file "$SPLIT_FILE" \
    --alphas "$ALPHAS" \
    --output "$RESULTS_ROOT/$MODEL/$LANG/ensemble.json" \
    --device cuda

echo
echo "Done: $(date)"
