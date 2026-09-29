#!/bin/bash
#SBATCH --job-name=baseline_probe
#SBATCH --account=gpuuva086
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/baseline_probe_%j.out
#SBATCH --error=slurm_logs/baseline_probe_%j.err

# Baseline discrete cloze probing (X-FACTR replication).
#
# For every (model, language) pair:
#   1. scripts/probe.py  — decode with the manual templates in data/TREx_prompts.csv
#   2. scripts/ana.py    — alias-aware Exact Match against data/alias/mTRExf,
#                          macro-averaged over relations (all / single-token / multi-token)
#
# Everything is overridable from the environment, e.g.:
#   LANGS=en,nl,sw sbatch baseline_probe.slurm
#   FACTS= sbatch baseline_probe.slurm            # probe all facts, not just the test split

set -eo pipefail

cd "$HOME/Multilingual_NLP_Project"
mkdir -p slurm_logs

PYTHON="${PYTHON:-$HOME/.conda/envs/multilingual_nlp/bin/python}"
MODELS="${MODELS:-mbert_base,xlmr_base}"
LANGS="${LANGS:-en,nl,tr,el,sw}"
PROBE="${PROBE:-mlamaf}"
NUM_MASK="${NUM_MASK:-5}"
BATCH_SIZE="${BATCH_SIZE:-20}"
# Restrict to the shared test split so the baseline is comparable to the
# prompt-tuning results. Set FACTS= (empty) to probe every fact.
FACTS="${FACTS-splits/shared_en_nl_tr_el_sw_seed42.json:test}"
OUT_DIR="${OUT_DIR:-results/baseline_probe}"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"

echo "=========================================="
echo "Baseline discrete cloze probing"
echo "Models:   $MODELS"
echo "Langs:    $LANGS"
echo "Probe:    $PROBE"
echo "Facts:    ${FACTS:-<all>}"
echo "Num mask: $NUM_MASK"
echo "Output:   $OUT_DIR"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# Fail fast on environment problems instead of partway through the grid.
echo
echo "Environment check:"
LANGS="$LANGS" "$PYTHON" - <<'PY'
import os, importlib

import torch
import transformers

print("PyTorch:", torch.__version__, "| transformers:", transformers.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the allocated GPU.")
print("GPU:", torch.cuda.get_device_name(0))

missing = [m for m in ("overrides", "joblib", "pandas", "sentencepiece", "SPARQLWrapper")
           if importlib.util.find_spec(m) is None]
if missing:
    raise RuntimeError(f"Missing packages: {missing}")

# These languages' templates need UniMorph inflection (see scripts/prompt.py).
inflected = set(os.environ["LANGS"].split(",")) & {"el", "tr", "ru", "hu", "mr", "bn"}
if inflected and importlib.util.find_spec("unimorph_inflect") is None:
    raise RuntimeError(
        f"unimorph_inflect is required for {sorted(inflected)}. Install it "
        "(pip install git+https://github.com/antonisa/unimorph_inflect) "
        "or drop those languages, e.g. LANGS=en,nl,sw")
print("OK")
PY

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra LANG_LIST <<< "$LANGS"

FACTS_ARG=()
if [ -n "$FACTS" ]; then
    FACTS_ARG=(--facts "$FACTS")
fi

printf 'model\tlang\tem_all\tem_single\tem_multi\tn_all\tn_single\tn_multi\n' > "$SUMMARY"

for m in "${MODEL_LIST[@]}"; do
    for l in "${LANG_LIST[@]}"; do
        pred_dir="$OUT_DIR/${m}__${l}"
        probe_log="$OUT_DIR/${m}__${l}.probe.out"
        ana_log="$OUT_DIR/${m}__${l}.ana.out"
        rm -rf "$pred_dir"

        echo
        echo "========== $m / $l =========="

        "$PYTHON" scripts/probe.py \
            --probe "$PROBE" \
            --model "$m" \
            --lang "$l" \
            --num_mask "$NUM_MASK" \
            --batch_size "$BATCH_SIZE" \
            --pred_dir "$pred_dir" \
            "${FACTS_ARG[@]}" \
            > "$probe_log" 2>&1
        tail -n 1 "$probe_log"

        "$PYTHON" scripts/ana.py \
            --task multi_eval \
            --probe "$PROBE" \
            --model "$m" \
            --lang "$l" \
            --norm \
            --inp "$pred_dir" \
            > "$ana_log" 2>&1
        tail -n 2 "$ana_log"

        # "overall acc <all>\t<single>\t<multi>" and "overall number <all>\t<single>\t<multi>"
        acc=$(grep '^overall acc' "$ana_log" | sed 's/^overall acc //')
        num=$(grep '^overall number' "$ana_log" | sed 's/^overall number //')
        printf '%s\t%s\t%s\t%s\n' "$m" "$l" "$acc" "$num" >> "$SUMMARY"
    done
done

echo
echo "=========================================="
echo "Baseline probing completed: $(date)"
echo "Macro-averaged EM (over relations):"
echo "=========================================="
column -t -s $'\t' "$SUMMARY"
