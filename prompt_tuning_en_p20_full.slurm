#!/bin/bash
#SBATCH --job-name=pt_en_p20
#SBATCH --account=gpuuva086
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/pt_en_p20_%j.out
#SBATCH --error=slurm_logs/pt_en_p20_%j.err

set -eo pipefail

cd "$HOME/Multilingual_NLP_Project"
mkdir -p slurm_logs
mkdir -p checkpoints/en_p20_seed42

PYTHON="/gpfs/home2/scur0521/.conda/envs/multilingual_nlp/bin/python"

echo "=========================================="
echo "Full prompt-tuning experiment"
echo "Language: English"
echo "Prompt tokens: 20"
echo "Seed: 42"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "GPU allocation:"
nvidia-smi

echo
echo "PyTorch CUDA check:"
"$PYTHON" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the allocated GPU.")

print("GPU:", torch.cuda.get_device_name(0))
print("GPU count:", torch.cuda.device_count())
PY

echo
echo "Starting full English prompt-tuning training..."
echo "Expected shared split:"
echo "  Train: 29388"
echo "  Val:    3656"
echo "  Test:   3716"
echo

PYTHONPATH="$HOME:${PYTHONPATH:-}" \
"$PYTHON" pipeline_v1/prompt_tuning_train.py \
    --lang en \
    --probe mlamaf \
    --split_file splits/shared_en_fr_nl_seed42.json \
    --num_prompt_tokens 20 \
    --learning_rate 0.3 \
    --epochs 10 \
    --batch_size 32 \
    --output_dir checkpoints/en_p20_seed42 \
    --device cuda

echo
echo "=========================================="
echo "Full English P=20 training completed"
echo "Date: $(date)"
echo "=========================================="

echo
echo "Generated files:"
find checkpoints/en_p20_seed42 \
    -maxdepth 1 -type f \
    -printf '%f  %s bytes\n'
