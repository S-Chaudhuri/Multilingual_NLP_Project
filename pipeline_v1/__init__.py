"""
pipeline_v1 — Fixed-LM Prompt Tuning for mBERT on X-FACTR.

Modules:
    soft_prompt          — SoftPromptEmbedding nn.Module
    prompt_tuning_data   — PromptTuningDataset + data loading helpers
    prompt_tuning_train  — Training pipeline (frozen model + trainable prompt)
    prompt_tuning_eval   — Evaluation with trained prompts + zero-shot comparison
"""
