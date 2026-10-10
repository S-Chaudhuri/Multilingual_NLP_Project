# Fixed-LM Prompt Tuning for Multilingual Factual Knowledge Probing

This project implements **Fixed-LM Prompt Tuning** (Lester et al., 2021) on the [X-FACTR](https://x-factr.github.io/) multilingual factual knowledge probing benchmark, using **mBERT** (`bert-base-multilingual-cased`) as the base model.

> For the original X-FACTR project documentation, see [OLD_README.md](OLD_README.md).

---

## Overview

In Fixed-LM Prompt Tuning, the entire pre-trained language model remains **frozen** — no gradient updates flow through its ~177M parameters. Instead, a small set of **learnable continuous embeddings** (soft prompts) are prepended to the input, and only these are optimized during training. This makes the approach extremely parameter-efficient:

| Prompt Length (P) | Trainable Parameters | % of Model |
|---|---|---|
| P = 20 | **15,360** | 0.008% |
| P = 50 | **38,400** | 0.021% |
| P = 100 | **76,800** | 0.043% |

---

## Project Structure

```
Multilingual_NLP_Project/
├── pipeline_v1/                          # ← Fixed-LM Prompt Tuning pipeline
│   ├── __init__.py                       # Package init
│   ├── soft_prompt.py                    # SoftPromptEmbedding nn.Module
│   ├── prompt_tuning_data.py             # Dataset class for X-FACTR data
│   ├── prompt_tuning_train.py            # Training pipeline (frozen mBERT + soft prompt)
│   ├── prompt_tuning_eval.py             # Evaluation & zero-shot comparison
│   ├── prompt_ensemble.py                # Prompt ensemble evaluation
│   ├── ablation_summary.py               # Ablation study analysis
│   ├── create_multilingual_split.py      # Multilingual train/val/test split creation
│   ├── create_token_length_splits.py     # Single-token vs multi-token target splits
│   ├── make_budget_partitions.py         # Fixed-budget per-language partitions (transfer grid)
│   ├── multilingual_data.py              # Multilingual dataset + balanced batch sampler
│   ├── multilingual_train.py             # Joint multilingual prompt tuning
│   ├── transfer_eval.py                  # Evaluation on all five languages
│   ├── analysis/                         # Data validation & analysis scripts
│   │   ├── validate_language_coverage.py # UTF-8 integrity, missing relations, sample sizes
│   │   ├── analyze_target_tokenization.py# Token-length distribution analysis
│   │   ├── transfer_summary.py           # Transfer-grid tables (bootstrap CIs, McNemar)
│   │   └── transfer_figures.py           # Transfer-grid figures
│   ├── results/transfer/                 # Local copy of transfer results
│   │   └── figures/                      # PNGs written by transfer_figures.py
│   └── configs/
│       └── prompt_tuning_config.json     # Default hyperparameters
├── scripts/
│   ├── probe.py                          # [MODIFIED] Added --soft_prompt_path support
│   ├── ana.py                            # Alias-aware Exact Match analysis
│   └── ...                               # Original X-FACTR scripts
├── data/                                 # X-FACTR datasets (mTREx, mTRExf, etc.)
├── splits/                               # Deterministic multilingual data splits
│   ├── shared_en_nl_tr_el_sw_seed42.json # Cross-lingual shared train/val/test split
│   └── token_length_test_*.json          # Single-token / multi-token evaluation subsets
├── checkpoints/                          # Saved soft prompt weights (created at runtime)
├── results/                              # Probing & evaluation results
├── baseline_probe.slurm                  # SLURM script: baseline discrete cloze probing
├── ablation_train.slurm                  # SLURM script: ablation training runs
├── setup_env.py                          # Builds the conda env (Python 3.8 + requirements_pipeline.txt + UniMorph + model weights)
├── OLD_README.md                         # Original X-FACTR README
├── requirements.txt                      # Original X-FACTR pins (kept for reference only; do not install)
└── requirements_pipeline.txt             # Pinned dependencies for baseline probing and pipeline_v1/
```

---

## Quick Start

### Prerequisites

All dependencies for baseline probing (`scripts/probe.py`, `scripts/ana.py`) and the soft-prompt pipeline (`pipeline_v1/`) are pinned in [`requirements_pipeline.txt`](requirements_pipeline.txt):

| Package | Version |
|---|---|
| torch | 2.4.1 |
| transformers / tokenizers | 4.46.3 / 0.20.3 |
| sentencepiece | 0.2.0 |
| numpy / pandas | 1.24.4 / 2.0.3 |
| tqdm, matplotlib, joblib, overrides, SPARQLWrapper | see file |

The target environment is **Python 3.8 on Linux (Snellius)**. Python 3.8 is required for the optional UniMorph inflection used by the Russian and Greek templates (`dynet==2.1.2` only ships a prebuilt wheel for 3.8). `requirements.txt` holds the original X-FACTR pins and is kept for reference only.

#### Option A — full environment (recommended, Snellius login node)

`setup_env.py` creates a conda env `multilingual_nlp` (Python 3.8), installs `requirements_pipeline.txt`, sets up UniMorph with its `rus`/`ell2` models, caches the mBERT and XLM-R weights, and runs an import/inflection check:

```bash
module load 2025
module load Anaconda3/2025.06-1      # see: module spider Anaconda3
python setup_env.py                  # options: --recreate, --no-unimorph, --skip-models
export PYTHON=$HOME/.conda/envs/multilingual_nlp/bin/python
```

Rerunning is safe: an existing env is reused. Run it on a login node, because compute nodes may not have internet access.

#### Option B — packages only (e.g. local development)

```bash
conda create -n multilingual_nlp python=3.8 pip
conda activate multilingual_nlp
python -m pip install -r requirements_pipeline.txt
```

Without UniMorph, Russian and Greek templates cannot be inflected (they keep markup such as `[X.Nom]`, and most facts are dropped), so restrict local runs to `en`, `nl` and `ko`.

> **Notes**
> - Always install with `python -m pip` so packages go into the interpreter you actually run.
> - The current shared splits (`splits/shared_en_nl_ru_el_ko_rulc*_seed42.json`) were built with lowercased Russian inflection. Set `export XFACTR_RU_LOWERCASE=1` before training or evaluating on them, otherwise Russian facts will not match the split.
> - On Windows, set `PYTHONUTF8=1` so the UTF-8 data files are read correctly.

### 1. Train a Soft Prompt

Train on English data with default settings (P=20, lr=0.3, 10 epochs):

```bash
python pipeline_v1/prompt_tuning_train.py --lang en
```

Train with custom settings:

```bash
python pipeline_v1/prompt_tuning_train.py \
    --lang en \
    --num_prompt_tokens 50 \
    --learning_rate 0.1 \
    --epochs 20 \
    --batch_size 16 \
    --probe mlamaf
```

Train on specific relations only:

```bash
python pipeline_v1/prompt_tuning_train.py --lang en --pids P19,P20
```

Use a config file:

```bash
python pipeline_v1/prompt_tuning_train.py --lang en --config pipeline_v1/configs/prompt_tuning_config.json
```

Checkpoints are saved to `checkpoints/` by default (override with `--output_dir`).

### 2. Evaluate a Trained Soft Prompt

Evaluate the trained prompt and compare against zero-shot baseline:

```bash
python pipeline_v1/prompt_tuning_eval.py \
    --lang en \
    --soft_prompt_path checkpoints/soft_prompt_en_P20_best.pt \
    --num_prompt_tokens 20 \
    --compare_zero_shot \
    --detailed
```

Save results to a JSON file:

```bash
python pipeline_v1/prompt_tuning_eval.py \
    --lang en \
    --soft_prompt_path checkpoints/soft_prompt_en_P20_best.pt \
    --compare_zero_shot \
    --output results/eval_en_P20.json
```

### 3. Evaluate via the Original X-FACTR Probe

The original `scripts/probe.py` has been extended with `--soft_prompt_path` support:

```bash
cd scripts
python probe.py \
    --model mbert_base \
    --lang en \
    --probe mlamaf \
    --soft_prompt_path ../checkpoints/soft_prompt_en_P20_best.pt \
    --num_prompt_tokens 20
```

Without `--soft_prompt_path`, `probe.py` behaves exactly as before (zero-shot probing).

### 4. Multilingual Transfer-Scaling Experiments (fixed fact budget)

Does training one soft prompt on more source languages improve zero-shot transfer when the
total number of unique training facts is fixed (B = 8,928 = 32 × 279, i.e. nearly all 8,958
train facts of the split)? Each run trains jointly on its
source languages (balanced batches, equal loss weight per language) and is evaluated on the
test split of all five languages.

| Run | Source languages | Facts / language | Zero-shot targets |
|---|---|---|---|
| R1 | en | 8,928 | nl, ru, el, ko |
| R2 | en, nl | 4,464 | ru, el, ko |
| R3 | en, ru | 4,464 | nl, el, ko |
| R4 | en, nl, ru, el | 2,232 | ko |
| R5 | en, ru, el, ko | 2,232 | nl |

Grid: 2 models (mBERT, XLM-R) × 5 runs × 3 seeds (42, 43, 44) = 30 jobs; k = 20, lr 0.3
(linear), batch 32, 10 epochs, keep split `splits/shared_en_nl_ru_el_ko_rulc_seed42.json`.
Language subsets are disjoint and never overlap val/test facts.

Primary metric: candidate-ranking EM (as in `prompt_ensemble.py`) on a candidate pool shared
by all languages, with the length-normalisation α selected on the **source-language** val
split. Token accuracy, full-pool EM, chance level and the no-prompt model are also reported.

```bash
# On a Snellius login node, from the project root (Russian/Greek need UniMorph: python setup_env.py)
bash run_transfer_grid.sh                 # budget partitions + 30-task SLURM array

# After the array has finished
python pipeline_v1/analysis/transfer_summary.py --results_root pipeline_v1/results/transfer
python pipeline_v1/analysis/analyze_target_tokenization.py --languages en nl ru el ko \
    --models mbert_base xlmr_base --output_dir pipeline_v1/results/transfer/tokenization
python pipeline_v1/analysis/transfer_figures.py   # PNGs in pipeline_v1/results/transfer/figures/
```

> **Optional: failure-mode analysis (F8).** [`error_sampling.py`](pipeline_v1/analysis/error_sampling.py)
> samples incorrect predictions (R1 vs R4, both models) into `pipeline_v1/results/transfer/annotation/`.
> The `label` column has been pre-filled by an LLM and must be validated by human annotators
> ([`annotation_agreement.py`](pipeline_v1/analysis/annotation_agreement.py) `sample` / `score`)
> before F8 is reported.

#### Figures

| Figure | Content |
|---|---|
| F1 `transfer_heatmap` | Runs × languages: EM and gain over no prompt |
| F2 `scaling` | Zero-shot EM vs number of source languages |
| F3 `transfer_bars` | Per target: no prompt vs every run where it is zero-shot |
| F4 `fertility` | Gold token length vs EM drop; EM by length bucket |
| F5 `same_fact` | P(correct in X \| correct in English) by gold length |
| F6 `overlap` | Number of languages a fact is correct in; Jaccard overlap |
| F7 `training_curves` | Validation loss per source language and epoch |
| F9 `relation_diff` | Per-relation EM change on Korean, R4 vs R1 |

Single run by hand:

```bash
export XFACTR_RU_LOWERCASE=1
python pipeline_v1/make_budget_partitions.py --langs en nl --budget 8928 --seed 42
python pipeline_v1/multilingual_train.py --langs en nl \
    --budget_file splits/budget/en-nl_B8928_seed42.json --seed 42 \
    --model_name bert-base-multilingual-cased \
    --output_dir checkpoints/transfer/mbert_base/en-nl/seed42
python pipeline_v1/transfer_eval.py --source_langs en nl \
    --soft_prompt_path checkpoints/transfer/mbert_base/en-nl/seed42/soft_prompt_en-nl_P20_best.pt \
    --model_name bert-base-multilingual-cased \
    --output_dir pipeline_v1/results/transfer/mbert_base/en-nl/seed42
```

---

## Architecture

```
Input Tokens → mBERT Tokenizer → Token IDs
                                      ↓
                           mBERT Word Embeddings (❄️ FROZEN)
                                      ↓
           [Soft Prompt Embeds (🔥 TRAINABLE)] + [Input Embeds]
                                      ↓
                         12× Transformer Blocks (❄️ FROZEN)
                                      ↓
                              MLM Head (❄️ FROZEN)
                                      ↓
                     Logits at [MASK] position → CrossEntropyLoss
                                      ↓
                    Backprop gradients → Only to Soft Prompt
```

---

## Training Configuration

| Hyperparameter | Default | Notes |
|---|---|---|
| Base Model | `bert-base-multilingual-cased` | 12 layers, 768 hidden, 12 heads |
| Model State | ❄️ Completely Frozen | `requires_grad = False` for all params |
| Prompt Length (P) | 20 | Tunable: 10, 20, 50, 100 |
| Initialization | Random vocab embeddings | Per Lester et al. best practice |
| Loss Function | CrossEntropyLoss | Over [MASK] logits vs. gold token ID |
| Optimizer | AdamW | Applied only to soft prompt params |
| Learning Rate | 0.3 | Higher LR for few-parameter optimization |
| Weight Decay | 1e-5 | Light L2 regularization |
| LR Schedule | Linear warmup + decay | 10% warmup steps |
| Batch Size | 32 | Adjustable based on GPU memory |
| Epochs | 10 | With best-checkpoint saving on val loss |

---

## Key Files

| File | Description |
|---|---|
| [`soft_prompt.py`](pipeline_v1/soft_prompt.py) | `SoftPromptEmbedding` — the only trainable module. Prepends learnable vectors to input embeddings. |
| [`prompt_tuning_data.py`](pipeline_v1/prompt_tuning_data.py) | `PromptTuningDataset` — loads X-FACTR triples, fills prompt templates, tokenizes with [MASK]. |
| [`prompt_tuning_train.py`](pipeline_v1/prompt_tuning_train.py) | Training loop: freezes mBERT, attaches soft prompt, runs AdamW with warmup scheduler. |
| [`prompt_tuning_eval.py`](pipeline_v1/prompt_tuning_eval.py) | Evaluation: prompt-tuned vs. zero-shot comparison, per-example prediction output. |
| [`create_multilingual_split.py`](pipeline_v1/create_multilingual_split.py) | Creates deterministic multilingual train/val/test splits shared across languages. |
| [`create_token_length_splits.py`](pipeline_v1/create_token_length_splits.py) | Separates test facts into single-token and multi-token subsets per language. |
| [`validate_language_coverage.py`](pipeline_v1/analysis/validate_language_coverage.py) | Validates UTF-8 integrity, identifies missing relations, reports sample sizes. |
| [`analyze_target_tokenization.py`](pipeline_v1/analysis/analyze_target_tokenization.py) | Analyzes token-length distributions across languages and models. |
| [`prompt_ensemble.py`](pipeline_v1/prompt_ensemble.py) | Prompt ensemble evaluation across multiple trained checkpoints. |
| [`ablation_summary.py`](pipeline_v1/ablation_summary.py) | Summarizes and compares ablation study results. |
| [`configs/prompt_tuning_config.json`](pipeline_v1/configs/prompt_tuning_config.json) | Default hyperparameters in JSON format. |
| [`make_budget_partitions.py`](pipeline_v1/make_budget_partitions.py) | Fixed-budget, disjoint per-language training partitions for the transfer experiments. |
| [`multilingual_data.py`](pipeline_v1/multilingual_data.py) | Multilingual dataset and balanced per-language batch sampler. |
| [`multilingual_train.py`](pipeline_v1/multilingual_train.py) | Joint multilingual prompt tuning (one prompt, several source languages). |
| [`transfer_eval.py`](pipeline_v1/transfer_eval.py) | Evaluates a prompt on all languages: token accuracy, EM (shared/full pool, source-val α). |
| [`configs/transfer_config.json`](pipeline_v1/configs/transfer_config.json) | Frozen hyperparameters of the transfer grid. |
| [`analysis/transfer_summary.py`](pipeline_v1/analysis/transfer_summary.py) | Aggregates runs into CSV tables with bootstrap CIs and McNemar tests. |
| [`analysis/transfer_figures.py`](pipeline_v1/analysis/transfer_figures.py) | Figures (heatmap, scaling, fertility, overlap, training curves, per-relation differences). |
| [`run_transfer_grid.sh`](run_transfer_grid.sh) / [`transfer_train.slurm`](transfer_train.slurm) | Launcher and SLURM array for the 30 transfer jobs. |
| [`scripts/probe.py`](scripts/probe.py) | Original X-FACTR probe — extended with `--soft_prompt_path` for seamless integration. |
| [`scripts/ana.py`](scripts/ana.py) | Alias-aware Exact Match analysis for baseline probing results. |
| [`baseline_probe.slurm`](baseline_probe.slurm) | SLURM script for baseline discrete cloze probing across models and languages. |

---

## References

- Lester et al. (2021). [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- Jiang et al. (2020). [X-FACTR: Multilingual Factual Knowledge Retrieval from Pretrained Language Models](https://arxiv.org/abs/2010.06189)
