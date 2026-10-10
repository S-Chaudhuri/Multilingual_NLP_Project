"""
Joint multilingual Fixed-LM prompt tuning with a fixed fact budget.

One soft prompt is trained on several source languages at the same time.
Every optimisation step uses a batch with exactly batch_size / L examples from
each of the L languages (BalancedLanguageBatchSampler), so the prompt is
optimised for all source languages in parallel instead of sequentially.

The facts per language come from a budget file (make_budget_partitions.py):
language subsets are disjoint and together contain exactly B facts, so every
configuration trains on the same number of unique facts.

Loss:
    --loss_reduction language (default): mean over languages of each
        language's token-level cross-entropy. Every language gets the same
        weight, even though e.g. Greek targets have more tokens than English.
    --loss_reduction token: plain token-level mean (as prompt_tuning_train.py).
    With a single language both are identical.

Model selection uses the validation split of the SOURCE languages only (mean
of the per-language validation losses, equal weights). Target-language data
is never seen during training or checkpoint selection.

Usage:
    python pipeline_v1/multilingual_train.py \\
        --langs en nl ru el \\
        --budget_file splits/budget/en-nl-ru-el_B8928_seed42.json \\
        --model_name bert-base-multilingual-cased \\
        --seed 42 \\
        --output_dir checkpoints/transfer/mbert_base/en-nl-ru-el/seed42
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from os.path import abspath, dirname, exists, join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

PIPELINE_DIR = dirname(abspath(__file__))
ROOT = dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from soft_prompt import SoftPromptEmbedding
from prompt_tuning_data import collate_fn
from prompt_tuning_train import (
    DEFAULTS as SINGLE_LANG_DEFAULTS,
    WarmupPlateauSchedule,
    evaluate,
    freeze_model,
    masked_lm_logits,
    verify_frozen,
)
from multilingual_data import (
    BalancedLanguageBatchSampler,
    MultilingualFactDataset,
    check_budget_against_split,
    load_budget,
    load_language_datasets,
    load_split,
    multilingual_collate_fn,
    split_subset,
)

logger = logging.getLogger("multilingual_train")

DEFAULT_CONFIG = join(PIPELINE_DIR, "configs", "transfer_config.json")
DEFAULT_SPLIT = "splits/shared_en_nl_ru_el_ko_rulc_seed42.json"


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else join(ROOT, path)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str) -> dict:
    config = dict(SINGLE_LANG_DEFAULTS)
    config.setdefault("loss_reduction", "language")
    if path and exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
        logger.info(f"Loaded config from {path}")
    return config


def batch_loss(logits, targets, example_index, langs, lang_to_id, reduction):
    """
    Cross-entropy over the target tokens of a mixed-language batch.

    Returns (loss, {lang: token-mean loss of that language}).
    """
    token_losses = F.cross_entropy(logits, targets, reduction="none")

    lang_ids = torch.tensor(
        [lang_to_id[lang] for lang in langs], device=token_losses.device
    )[example_index]

    per_lang = {}
    for lang, lid in lang_to_id.items():
        mask = lang_ids == lid
        if mask.any():
            per_lang[lang] = token_losses[mask].mean()

    if reduction == "language":
        loss = torch.stack(list(per_lang.values())).mean()
    else:
        loss = token_losses.mean()

    return loss, {lang: value.item() for lang, value in per_lang.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Joint multilingual Fixed-LM prompt tuning (fixed fact budget)"
    )
    parser.add_argument("--langs", nargs="+", required=True,
                        help="Source languages, e.g. en nl ru el")
    parser.add_argument("--budget_file", required=True,
                        help="Budget JSON from make_budget_partitions.py")
    parser.add_argument("--split_file", default=DEFAULT_SPLIT,
                        help="Shared train/val/test split (val is used for model selection)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="JSON with the frozen grid hyperparameters")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--num_prompt_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_schedule", default=None, choices=["linear", "plateau"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--loss_reduction", default=None, choices=["language", "token"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_incomplete_split", action="store_true",
                        help="Warn instead of failing when budget/val facts are missing "
                             "for a language (e.g. dropped by another tokenizer)")
    parser.add_argument("--no_inflect", action="store_true")
    parser.add_argument("--device", default=None)
    # Debugging / smoke tests only.
    parser.add_argument("--max_facts_per_lang", type=int, default=None,
                        help="DEBUG: use only the first N budget facts per language")
    parser.add_argument("--max_val_per_lang", type=int, default=None,
                        help="DEBUG: use only the first N validation facts per language")
    args = parser.parse_args()

    config = load_config(args.config)
    for key in ("model_name", "num_prompt_tokens", "seed", "learning_rate",
                "lr_schedule", "epochs", "batch_size", "loss_reduction"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    langs = list(args.langs)
    tag = "-".join(langs)
    k = config["num_prompt_tokens"]
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Budget and split
    # -----------------------------------------------------------------------
    budget_path = resolve(args.budget_file)
    split_path = resolve(args.split_file)

    budget = load_budget(budget_path)
    split = load_split(split_path)

    if set(budget["train"]) != set(langs):
        raise ValueError(
            f"--langs {langs} do not match the budget file languages "
            f"{list(budget['train'])}"
        )
    budget_split = budget["metadata"].get("split_file")
    if budget_split and os.path.normpath(resolve(budget_split)) != os.path.normpath(split_path):
        raise ValueError(
            f"Budget file was built from {budget_split}, but --split_file is {args.split_file}"
        )
    if budget["metadata"].get("seed") != config["seed"]:
        logger.warning(
            f"Budget seed {budget['metadata'].get('seed')} differs from training "
            f"seed {config['seed']}"
        )
    check_budget_against_split(budget, split)

    # -----------------------------------------------------------------------
    # Device and seed
    # -----------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # -----------------------------------------------------------------------
    # Frozen model and soft prompt (same order as prompt_tuning_train.py, so
    # a given seed gives the same initial prompt in every configuration)
    # -----------------------------------------------------------------------
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForMaskedLM.from_pretrained(config["model_name"])
    model.to(device)
    model.eval()
    freeze_model(model)

    config["embedding_dim"] = model.config.hidden_size

    soft_prompt = SoftPromptEmbedding(
        num_prompt_tokens=k,
        embedding_dim=config["embedding_dim"],
        init_from_vocab=config["init_from_vocab"],
        tokenizer=tokenizer,
        model=model,
    )
    soft_prompt.to(device)
    verify_frozen(model, soft_prompt)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    datasets = load_language_datasets(
        tokenizer,
        langs,
        probe=config["probe"],
        portion=config["portion"],
        num_mask=config["num_mask"],
        use_inflection=not args.no_inflect,
        max_seq_len=config["max_seq_len"],
    )

    train_dataset = MultilingualFactDataset(
        datasets,
        budget["train"],
        allow_incomplete=args.allow_incomplete_split,
        max_facts_per_lang=args.max_facts_per_lang,
    )

    sampler = BalancedLanguageBatchSampler(
        train_dataset.indices_by_lang,
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=multilingual_collate_fn,
        num_workers=0,
    )

    val_loaders = {}
    for lang in langs:
        subset = split_subset(
            datasets[lang], split["val"], lang, "validation",
            allow_incomplete=args.allow_incomplete_split,
            limit=args.max_val_per_lang,
        )
        val_loaders[lang] = DataLoader(
            subset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    num_train = {lang: len(idx) for lang, idx in train_dataset.indices_by_lang.items()}
    num_val = {lang: len(loader.dataset) for lang, loader in val_loaders.items()}
    logger.info(f"Train facts per language: {num_train}")
    logger.info(f"Val facts per language:   {num_val}")
    logger.info(
        f"Balanced batches: {sampler.num_batches}/epoch, "
        f"{sampler.per_lang} examples per language per batch"
    )

    # -----------------------------------------------------------------------
    # Optimiser and schedule
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        soft_prompt.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(sampler) * config["epochs"]
    if config.get("warmup_epochs") is not None:
        # Fixed warmup length, independent of the (maximum) number of epochs.
        warmup_steps = int(config["warmup_epochs"] * len(sampler))
    else:
        warmup_steps = int(config["warmup_ratio"] * total_steps)

    if config["lr_schedule"] == "plateau":
        scheduler = WarmupPlateauSchedule(
            optimizer,
            peak_lr=config["learning_rate"],
            warmup_steps=warmup_steps,
            factor=config["plateau_factor"],
            patience=config["plateau_patience"],
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lang_to_id = {lang: i for i, lang in enumerate(langs)}

    logger.info("=" * 60)
    logger.info("Joint multilingual prompt tuning")
    logger.info(f"  Source languages:  {langs}")
    logger.info(f"  Budget:            {budget['metadata'].get('budget')} facts")
    logger.info(f"  Prompt tokens (k): {k}")
    logger.info(f"  Epochs:            {config['epochs']}")
    logger.info(f"  Batch size:        {config['batch_size']}")
    logger.info(f"  Learning rate:     {config['learning_rate']} ({config['lr_schedule']})")
    logger.info(f"  Loss reduction:    {config['loss_reduction']}")
    logger.info(f"  Total steps:       {total_steps} (warmup {warmup_steps})")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    best_prompt = soft_prompt.soft_prompt.detach().clone()
    best_epoch = 0
    lr_reductions = 0
    history = []

    best_path = join(args.output_dir, f"soft_prompt_{tag}_P{k}_best.pt")

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        sampler.set_epoch(epoch)
        soft_prompt.train()

        sums = {"loss": 0.0, "acc": 0.0, "batches": 0}
        lang_sums = {lang: 0.0 for lang in langs}
        lang_counts = {lang: 0 for lang in langs}

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            target_mask = labels != -100
            logits = masked_lm_logits(model, soft_prompt, input_ids, attention_mask, target_mask)
            targets = labels[target_mask]
            # Row-major order of target_mask == order of the logits rows.
            example_index = target_mask.nonzero(as_tuple=True)[0]

            loss, per_lang = batch_loss(
                logits, targets, example_index, batch["lang"],
                lang_to_id, config["loss_reduction"],
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            sums["loss"] += loss.item()
            sums["acc"] += (logits.argmax(-1) == targets).float().mean().item()
            sums["batches"] += 1
            for lang, value in per_lang.items():
                lang_sums[lang] += value
                lang_counts[lang] += 1

            if (batch_idx + 1) % config["log_interval"] == 0:
                per_lang_str = " ".join(
                    f"{lang}={lang_sums[lang] / max(lang_counts[lang], 1):.3f}"
                    for lang in langs
                )
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss {sums['loss'] / sums['batches']:.4f} | per language: {per_lang_str} | "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}"
                )

        train_loss_by_lang = {
            lang: lang_sums[lang] / max(lang_counts[lang], 1) for lang in langs
        }

        # Validation on every source language; model selection on the mean.
        val_by_lang = {
            lang: evaluate(model, soft_prompt, loader, loss_fn, device)
            for lang, loader in val_loaders.items()
        }
        val_loss = sum(v["loss"] for v in val_by_lang.values()) / len(val_by_lang)
        val_acc = sum(v["accuracy"] for v in val_by_lang.values()) / len(val_by_lang)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{config['epochs']} ({elapsed:.1f}s) | "
            f"Train loss {sums['loss'] / max(sums['batches'], 1):.4f} | "
            f"Val loss (mean) {val_loss:.4f} | Val acc (mean) {val_acc:.4f} | "
            + " ".join(f"{lang}:{v['loss']:.3f}/{v['accuracy']:.3f}" for lang, v in val_by_lang.items())
        )

        history.append({
            "epoch": epoch,
            "train_loss": sums["loss"] / max(sums["batches"], 1),
            "train_acc": sums["acc"] / max(sums["batches"], 1),
            "train_loss_by_lang": train_loss_by_lang,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_loss_by_lang": {lang: v["loss"] for lang, v in val_by_lang.items()},
            "val_acc_by_lang": {lang: v["accuracy"] for lang, v in val_by_lang.items()},
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_prompt = soft_prompt.soft_prompt.detach().clone()
            soft_prompt.save(best_path)
            logger.info(f"  Saved best checkpoint -> {best_path}")

        if isinstance(scheduler, WarmupPlateauSchedule) and scheduler.end_epoch(val_loss):
            with torch.no_grad():
                soft_prompt.soft_prompt.copy_(best_prompt)
            optimizer.state.clear()
            lr_reductions += 1
            logger.info(
                f"  Val loss stalled: learning rate -> {scheduler.peak_lr:.4g}, "
                f"restored best prompt (val loss {best_val_loss:.4f})"
            )
            max_reductions = config.get("max_lr_reductions")
            if max_reductions is not None and lr_reductions >= max_reductions:
                logger.info(f"  Early stopping at epoch {epoch} after {lr_reductions} "
                            f"learning-rate reductions (best epoch {best_epoch})")
                break


    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    final_path = join(args.output_dir, f"soft_prompt_{tag}_P{k}_final.pt")
    soft_prompt.save(final_path)

    with open(join(args.output_dir, f"history_{tag}_P{k}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    run_config = dict(
        config,
        langs=langs,
        tag=tag,
        budget_file=args.budget_file,
        budget_sha256=file_sha256(budget_path),
        budget=budget["metadata"].get("budget"),
        budget_seed=budget["metadata"].get("seed"),
        split_file=args.split_file,
        num_train_facts=num_train,
        num_val_facts=num_val,
        batches_per_epoch=len(sampler),
        examples_per_lang_per_batch=sampler.per_lang,
        total_steps=total_steps,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        max_facts_per_lang=args.max_facts_per_lang,
        max_val_per_lang=args.max_val_per_lang,
        use_inflection=not args.no_inflect,
    )
    with open(join(args.output_dir, f"config_{tag}_P{k}.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"Best epoch {best_epoch} (mean source val loss {best_val_loss:.4f})")
    logger.info(f"Saved final checkpoint -> {final_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
