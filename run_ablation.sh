#!/bin/bash
# Ablation experiments & soft-prompt ensembling.
#
#   1. ablation_train.slurm    (array) train + evaluate a soft prompt for every
#                              model x k x lang x seed
#   2. ablation_ensemble.slurm (array) ensemble all prompts per model x lang,
#                              after every training task has finished
#   3. pipeline_v1/ablation_summary.py  (run by hand afterwards) builds the tables
#
# Run from the project root on a Snellius login node:
#   bash run_ablation.sh
#
# Every setting can be overridden from the environment, e.g.:
#   LANGS=en,nl,sw bash run_ablation.sh              # skip languages needing UniMorph
#   SEEDS=42,43,44 bash run_ablation.sh              # more distinct prompts per k
#   SEEDS=43,44 ENSEMBLE_SEEDS=42,43,44 bash run_ablation.sh
#                                    # add seeds to an existing run, ensemble all three
#   MODELS=xlmr_base KS=20 bash run_ablation.sh

set -eo pipefail

cd "$(dirname "$0")"

export PYTHON="${PYTHON:-$HOME/.conda/envs/multilingual_nlp/bin/python}"
export MODELS="${MODELS:-mbert_base,xlmr_base}"
export KS="${KS:-5,10,20}"
export LANGS="${LANGS:-en,nl,ru,bn,sw}"
export SEEDS="${SEEDS:-42}"
export ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-$SEEDS}"
export EPOCHS="${EPOCHS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-0.3}"
export LR_SCHEDULE="${LR_SCHEDULE:-linear}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
export SPLIT_FILE="${SPLIT_FILE:-splits/shared_en_nl_ru_bn_sw_seed42.json}"
export CKPT_ROOT="${CKPT_ROOT:-checkpoints/ablation}"
export RESULTS_ROOT="${RESULTS_ROOT:-results/ablation}"

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra K_LIST <<< "$KS"
IFS=',' read -ra LANG_LIST <<< "$LANGS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

N_TRAIN=$(( ${#MODEL_LIST[@]} * ${#K_LIST[@]} * ${#LANG_LIST[@]} * ${#SEED_LIST[@]} ))
N_ENSEMBLE=$(( ${#MODEL_LIST[@]} * ${#LANG_LIST[@]} ))

echo "Models: $MODELS"
echo "k:      $KS"
echo "Langs:  $LANGS"
echo "Seeds:  $SEEDS (ensembling: $ENSEMBLE_SEEDS)"
echo "LR:     $LEARNING_RATE ($LR_SCHEDULE schedule), $EPOCHS epochs"
echo "Python: $PYTHON"

# ---------------------------------------------------------------------------
# Check the environment and pre-download both models before queueing anything
# (compute nodes may not have internet access).
# ---------------------------------------------------------------------------

echo
echo "Environment check..."
"$PYTHON" - <<'PY'
import os, sys, importlib.util

missing = [m for m in ("torch", "transformers", "sentencepiece", "overrides", "joblib")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"Missing packages: {missing}")

langs = set(os.environ["LANGS"].split(","))
inflected = langs & {"el", "tr", "ru", "hu", "mr", "bn"}
if inflected and importlib.util.find_spec("unimorph_inflect") is None:
    sys.exit(
        f"unimorph_inflect is required for {sorted(inflected)}: without it their "
        "templates keep unfilled markup such as [X.Nom]. Install it "
        "(python setup_env.py) or drop "
        "those languages, e.g. LANGS=en,nl,sw")

# Without its model files UniMorph asks for keyboard input and the job fails.
models = {"tr": "tur", "el": "ell2", "ru": "rus", "hu": "hun", "bn": "ben"}
if inflected and importlib.util.find_spec("unimorph_inflect") is not None:
    root = os.path.join(os.path.expanduser("~"), "unimorph_inflect_resources")
    absent = [models[l] for l in sorted(inflected) if l in models
              and not os.path.isdir(os.path.join(root, models[l]))]
    if absent:
        raise RuntimeError(f"UniMorph models {absent} missing from {root}; run: python setup_env.py")

from transformers import AutoTokenizer, AutoModelForMaskedLM
names = {"mbert_base": "bert-base-multilingual-cased", "xlmr_base": "xlm-roberta-base"}
for key in os.environ["MODELS"].split(","):
    if key not in names:
        sys.exit(f"Unknown model key {key!r}; expected one of {sorted(names)}")
    AutoTokenizer.from_pretrained(names[key])
    AutoModelForMaskedLM.from_pretrained(names[key])
    print(f"cached {names[key]}")
print("OK")
PY

mkdir -p slurm_logs

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

TRAIN_JOB=$(sbatch --parsable --export=ALL --array=0-$(( N_TRAIN - 1 )) ablation_train.slurm)
echo
echo "Submitted training array:   $TRAIN_JOB ($N_TRAIN tasks)"

# afterany: ensembling still runs if a few training tasks fail; it uses the
# checkpoints that exist and warns about the rest.
ENSEMBLE_JOB=$(sbatch --parsable --export=ALL --array=0-$(( N_ENSEMBLE - 1 )) \
    --dependency=afterany:"$TRAIN_JOB" ablation_ensemble.slurm)
echo "Submitted ensembling array: $ENSEMBLE_JOB ($N_ENSEMBLE tasks, after $TRAIN_JOB)"

echo
echo "When both have finished, build the tables with:"
echo "  $PYTHON pipeline_v1/ablation_summary.py --results_root $RESULTS_ROOT"
