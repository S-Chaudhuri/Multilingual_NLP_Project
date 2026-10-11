#!/bin/bash
# Prompt-length ablation + ensembling on en/nl/ru/el/ko, twice, differing only
# in how malformed UniMorph inflections (ru, el) are handled:
#
#   keep  all facts; Russian names inflected in lowercase
#         (XFACTR_RU_LOWERCASE=1, see scripts/prompt.py)
#   drop  the same, minus every fact with a malformed Russian or Greek
#         inflection (pipeline_v1/filter_inflection_noise.py)
#
# Run from the project root on a Snellius login node:
#   bash run_inflection_variants.sh
#
# Afterwards, summarise each variant separately:
#   $PYTHON pipeline_v1/ablation_summary.py --results_root results/ablation_keep
#   $PYTHON pipeline_v1/ablation_summary.py --results_root results/ablation_drop

set -eo pipefail

cd "$(dirname "$0")"

export XFACTR_RU_LOWERCASE=1
export LANGS="${LANGS:-en,nl,ru,el,ko}"

echo "########## Variant: keep ##########"
SPLIT_FILE=splits/shared_en_nl_ru_el_ko_rulc_seed42.json \
CKPT_ROOT=checkpoints/ablation_keep \
RESULTS_ROOT=results/ablation_keep \
    bash run_ablation.sh

echo
echo "########## Variant: drop ##########"
SPLIT_FILE=splits/shared_en_nl_ru_el_ko_rulc_clean_seed42.json \
CKPT_ROOT=checkpoints/ablation_drop \
RESULTS_ROOT=results/ablation_drop \
    bash run_ablation.sh
