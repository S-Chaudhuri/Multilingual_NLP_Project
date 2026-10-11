#!/bin/bash
# Multilingual transfer-scaling experiments (fixed fact budget).
#
#   1. pipeline_v1/make_budget_partitions.py  (login node) one budget partition
#                                             per run x seed
#   2. transfer_train.slurm (array)           train + evaluate one soft prompt
#                                             per model x run x seed
#   3. analysis (run by hand afterwards, commands printed at the end)
#
# Runs (source languages; all other languages are zero-shot targets):
#   R1 en | R2 en,nl | R3 en,ru | R4 en,nl,ru,el | R5 en,ru,el,ko
#
# Run from the project root on a Snellius login node:
#   bash run_transfer_grid.sh
#
# Every setting can be overridden from the environment, e.g.:
#   MODELS=mbert_base bash run_transfer_grid.sh
#   SEEDS=42 RUNS="en;en,nl,ru,el" bash run_transfer_grid.sh
#   EXTRA_FLAGS=--allow_incomplete_split MODELS=xlmr_base bash run_transfer_grid.sh
#   DRY_RUN=1 bash run_transfer_grid.sh      # partitions + checks, no sbatch

set -eo pipefail

cd "$(dirname "$0")"

export PYTHON="${PYTHON:-$HOME/.conda/envs/multilingual_nlp/bin/python}"
export MODELS="${MODELS:-mbert_base,xlmr_base}"
export RUNS="${RUNS:-en;en,nl;en,ru;en,nl,ru,el;en,ru,el,ko}"
export SEEDS="${SEEDS:-42,43,44}"
export EVAL_LANGS="${EVAL_LANGS:-en,nl,ru,el,ko}"
export BUDGET="${BUDGET:-8928}"   # 32 x 279, max for the keep split (8,958 train facts)
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SPLIT_FILE="${SPLIT_FILE:-splits/shared_en_nl_ru_el_ko_rulc_seed42.json}"
export BUDGET_DIR="${BUDGET_DIR:-splits/budget}"
export CONFIG="${CONFIG:-pipeline_v1/configs/transfer_config.json}"
export ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
export CKPT_ROOT="${CKPT_ROOT:-checkpoints/transfer}"
export RESULTS_ROOT="${RESULTS_ROOT:-pipeline_v1/results/transfer}"
export EXTRA_FLAGS="${EXTRA_FLAGS:-}"
export XFACTR_RU_LOWERCASE="${XFACTR_RU_LOWERCASE:-1}"
export PYTHONUTF8=1

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=';' read -ra RUN_LIST <<< "$RUNS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

N_TASKS=$(( ${#MODEL_LIST[@]} * ${#RUN_LIST[@]} * ${#SEED_LIST[@]} ))

echo "Models: $MODELS"
echo "Runs:   $RUNS"
echo "Seeds:  $SEEDS"
echo "Budget: $BUDGET facts (batch size $BATCH_SIZE)"
echo "Split:  $SPLIT_FILE"
echo "Python: $PYTHON"

# ---------------------------------------------------------------------------
# Environment check and model pre-download (compute nodes may be offline).
# ---------------------------------------------------------------------------

echo
echo "Environment check..."
"$PYTHON" - <<'PY'
import os, sys, importlib.util

missing = [m for m in ("torch", "transformers", "sentencepiece", "overrides", "joblib",
                       "pandas", "matplotlib")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"Missing packages: {missing}")

langs = set(os.environ["EVAL_LANGS"].split(",")) | {
    l for run in os.environ["RUNS"].split(";") for l in run.split(",")}
inflected = langs & {"el", "ru"}
if inflected and importlib.util.find_spec("unimorph_inflect") is None:
    sys.exit(f"unimorph_inflect is required for {sorted(inflected)}; run: python setup_env.py")

models = {"el": "ell2", "ru": "rus"}
if inflected:
    root = os.path.join(os.path.expanduser("~"), "unimorph_inflect_resources")
    absent = [models[l] for l in sorted(inflected)
              if not os.path.isdir(os.path.join(root, models[l]))]
    if absent:
        sys.exit(f"UniMorph models {absent} missing from {root}; run: python setup_env.py")

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

# ---------------------------------------------------------------------------
# Budget partitions: one per run x seed (shared by both models).
# For a given seed every run draws the same B facts, only the assignment of
# facts to languages differs.
# ---------------------------------------------------------------------------

echo
echo "Budget partitions..."
mkdir -p "$BUDGET_DIR"
for RUN in "${RUN_LIST[@]}"; do
    IFS=',' read -ra SRC_LANGS <<< "$RUN"
    TAG=$(IFS='-'; echo "${SRC_LANGS[*]}")
    for SEED in "${SEED_LIST[@]}"; do
        OUT="$BUDGET_DIR/${TAG}_B${BUDGET}_seed${SEED}.json"
        if [ -f "$OUT" ]; then
            echo "  exists: $OUT"
            continue
        fi
        "$PYTHON" pipeline_v1/make_budget_partitions.py \
            --langs "${SRC_LANGS[@]}" \
            --budget "$BUDGET" \
            --batch_size "$BATCH_SIZE" \
            --seed "$SEED" \
            --split_file "$SPLIT_FILE" \
            --output "$OUT"
    done
done

mkdir -p pipeline_v1/results/slurm_logs

if [ -n "${DRY_RUN:-}" ]; then
    echo
    echo "DRY_RUN set: would submit $N_TASKS array tasks of transfer_train.slurm"
    exit 0
fi

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

JOB=$(sbatch --parsable --export=ALL --array=0-$(( N_TASKS - 1 )) transfer_train.slurm)
echo
echo "Submitted transfer array: $JOB ($N_TASKS tasks)"

echo
echo "When it has finished:"
echo "  $PYTHON pipeline_v1/analysis/transfer_summary.py --results_root $RESULTS_ROOT"
echo "  $PYTHON pipeline_v1/analysis/analyze_target_tokenization.py --languages en nl ru el ko --models mbert_base xlmr_base --output_dir $RESULTS_ROOT/tokenization"
echo "  $PYTHON pipeline_v1/analysis/error_sampling.py --model mbert_base --run en-nl-ru-el --seed 42"
echo "  $PYTHON pipeline_v1/analysis/transfer_figures.py"
