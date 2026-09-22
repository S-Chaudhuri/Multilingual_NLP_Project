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
│   └── configs/
│       └── prompt_tuning_config.json     # Default hyperparameters
├── scripts/
│   ├── probe.py                          # [MODIFIED] Added --soft_prompt_path support
│   └── ...                               # Original X-FACTR scripts (unchanged)
├── data/                                 # X-FACTR datasets (mTREx, mTRExf, etc.)
├── checkpoints/                          # Saved soft prompt weights (created at runtime)
├── Project_fixedLM/                      # Data loading utilities
├── OLD_README.md                         # Original X-FACTR README
└── requirements.txt                      # Dependencies
```

---

## Quick Start

### Prerequisites

```bash
pip install torch "transformers<5.0.0"
```

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
| [`configs/prompt_tuning_config.json`](pipeline_v1/configs/prompt_tuning_config.json) | Default hyperparameters in JSON format. |
| [`scripts/probe.py`](scripts/probe.py) | Original X-FACTR probe — extended with `--soft_prompt_path` for seamless integration. |

---

## References

- Lester et al. (2021). [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- Jiang et al. (2020). [X-FACTR: Multilingual Factual Knowledge Retrieval from Pretrained Language Models](https://arxiv.org/abs/2010.06189)
