#!/bin/bash
# Learning-rate x schedule sweep at k = 20 (keep split, en/nl/ru/el/ko, 3 seeds).
#
# Each configuration "schedule:learning_rate" is a full run_ablation.sh run with
# KS=20, written to results/lr_sweep/<schedule>_lr<rate> and
# checkpoints/lr_sweep/<schedule>_lr<rate>:
#
#   linear   warmup (10% of steps) + linear decay           (the original setup)
#   plateau  warmup, then halve the rate and restore the best prompt whenever
#            the validation loss stops improving
#
# Pick each model's configuration on VALIDATION EM in the notebook
# (pipeline_v1/analysis/ablation_results.ipynb, "Figure 8").
#
# Run from the project root on a Snellius login node:
#   bash run_lr_sweep.sh                                   # default grid (5 configs)
#   bash run_lr_sweep.sh linear:0.1 plateau:0.3            # only these configs
#   MODELS=xlmr_base bash run_lr_sweep.sh                  # one model
#
# Default grid: 5 configs x 2 models x 5 languages x 3 seeds = 150 training jobs
# (+ 10 ensembling jobs per config).

set -eo pipefail

cd "$(dirname "$0")"

export XFACTR_RU_LOWERCASE=1
export LANGS="${LANGS:-en,nl,ru,el,ko}"
export MODELS="${MODELS:-mbert_base,xlmr_base}"
export KS="${KS:-20}"
export SEEDS="${SEEDS:-42,43,44}"
export ENSEMBLE_SEEDS="$SEEDS"
export SPLIT_FILE=splits/shared_en_nl_ru_el_ko_rulc_seed42.json

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(linear:0.05 linear:0.1 linear:0.3 plateau:0.1 plateau:0.3)

for config in "${CONFIGS[@]}"; do
    schedule=${config%%:*}
    rate=${config#*:}
    case "$schedule" in
        linear|plateau) ;;
        *) echo "Unknown schedule in '$config' (expected linear:<rate> or plateau:<rate>)" >&2; exit 1 ;;
    esac
    name="${schedule}_lr${rate}"

    echo
    echo "########## $name: models $MODELS, k $KS, seeds $SEEDS ##########"
    LR_SCHEDULE=$schedule LEARNING_RATE=$rate \
    CKPT_ROOT=checkpoints/lr_sweep/$name RESULTS_ROOT=results/lr_sweep/$name \
        bash run_ablation.sh
done
