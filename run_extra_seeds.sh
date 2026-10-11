#!/bin/bash
# Add seeds 43 and 44 to every finished seed-42 experiment, so each setting has
# 3 seeds. Only the new seeds are trained; the ensembling jobs combine the
# prompts of all three seeds and overwrite each experiment's ensemble.json.
#
#   keep         all facts, Russian lowercased            (both models)
#   drop         filtered split                           (both models)
#   xlmr_lr005   XLM-R, learning rate 0.05                (keep split)
#   mbert_ep20   mBERT, 20 epochs                         (keep split)
#
# Run from the project root on a Snellius login node:
#   bash run_extra_seeds.sh                       # all four experiments
#   bash run_extra_seeds.sh keep drop             # only some of them
#
# Afterwards: python pipeline_v1/ablation_summary.py --results_root <results folder>,
# and the notebook (pipeline_v1/analysis/ablation_results.ipynb) averages over seeds.

set -eo pipefail

cd "$(dirname "$0")"

export XFACTR_RU_LOWERCASE=1
export LANGS="${LANGS:-en,nl,ru,el,ko}"
export SEEDS="${SEEDS:-43,44}"
export ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-42,43,44}"

KEEP_SPLIT=splits/shared_en_nl_ru_el_ko_rulc_seed42.json
DROP_SPLIT=splits/shared_en_nl_ru_el_ko_rulc_clean_seed42.json

EXPERIMENTS=("$@")
[ ${#EXPERIMENTS[@]} -eq 0 ] && EXPERIMENTS=(keep drop xlmr_lr005 mbert_ep20)

for experiment in "${EXPERIMENTS[@]}"; do
    echo
    echo "########## $experiment: train seeds $SEEDS, ensemble seeds $ENSEMBLE_SEEDS ##########"
    case "$experiment" in
        keep)
            SPLIT_FILE=$KEEP_SPLIT \
            CKPT_ROOT=checkpoints/ablation_keep RESULTS_ROOT=results/ablation_keep \
                bash run_ablation.sh ;;
        drop)
            SPLIT_FILE=$DROP_SPLIT \
            CKPT_ROOT=checkpoints/ablation_drop RESULTS_ROOT=results/ablation_drop \
                bash run_ablation.sh ;;
        xlmr_lr005)
            MODELS=xlmr_base LEARNING_RATE=0.05 SPLIT_FILE=$KEEP_SPLIT \
            CKPT_ROOT=checkpoints/ablation_keep_xlmr_lr005 RESULTS_ROOT=results/ablation_keep_xlmr_lr005 \
                bash run_ablation.sh ;;
        mbert_ep20)
            MODELS=mbert_base EPOCHS=20 SPLIT_FILE=$KEEP_SPLIT \
            CKPT_ROOT=checkpoints/ablation_keep_mbert_ep20 RESULTS_ROOT=results/ablation_keep_mbert_ep20 \
                bash run_ablation.sh ;;
        *)
            echo "Unknown experiment '$experiment' (expected: keep drop xlmr_lr005 mbert_ep20)" >&2
            exit 1 ;;
    esac
done
