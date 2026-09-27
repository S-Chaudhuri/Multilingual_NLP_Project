"""
Evaluation script for Fixed-LM Prompt Tuning on X-FACTR.

Loads a trained soft prompt checkpoint and evaluates prompt-tuned mBERT.
Optionally compares against the zero-shot frozen mBERT baseline.

The script can evaluate either:
    1. the complete language-specific dataset, or
    2. one partition (train / val / test) from a precomputed shared
       multilingual split.

For the multilingual experiments, use --split_file together with
--split test so that all languages are evaluated on exactly the same
underlying factual triples.

A fact is identified by:
    (relation, sub_uri, obj_uri)

Example:
    python pipeline_v1/prompt_tuning_eval.py \
        --lang tr \
        --soft_prompt_path checkpoints/soft_prompt_tr_P20_best.pt \
        --split_file splits/shared_en_nl_tr_el_sw_seed42.json \
        --split test \
        --compare_zero_shot
"""

import sys
import os
import json
import argparse
import logging
from os.path import dirname, abspath

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForMaskedLM


# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------

PIPELINE_DIR = dirname(abspath(__file__))
ROOT = dirname(PIPELINE_DIR)

sys.path.insert(0, PIPELINE_DIR)

from soft_prompt import SoftPromptEmbedding
from prompt_tuning_data import PromptTuningDataset, collate_fn
from prompt_tuning_train import (
    forward_with_soft_prompt,
    compute_loss,
    compute_accuracy,
    freeze_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("prompt_tuning_eval")


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_zero_shot(
    model,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """
    Evaluate the frozen model WITHOUT soft prompts.

    Uses the standard masked-language-model forward pass.
    """

    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        vocab_size = logits.size(-1)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )

        preds = logits.argmax(dim=-1)

        mask = labels != -100

        if mask.sum() > 0:
            acc = (
                preds[mask] == labels[mask]
            ).float().mean().item()
        else:
            acc = 0.0

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": total_acc / max(total_batches, 1),
    }


@torch.no_grad()
def evaluate_with_prompt(
    model,
    soft_prompt: SoftPromptEmbedding,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """
    Evaluate the frozen model WITH trained soft prompts.
    """

    model.eval()
    soft_prompt.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = forward_with_soft_prompt(
            model,
            soft_prompt,
            input_ids,
            attention_mask,
        )

        loss = compute_loss(
            logits,
            labels,
            soft_prompt.num_prompt_tokens,
            loss_fn,
        )

        acc = compute_accuracy(
            logits,
            labels,
            soft_prompt.num_prompt_tokens,
        )

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": total_acc / max(total_batches, 1),
    }


@torch.no_grad()
def detailed_predictions(
    model,
    soft_prompt: SoftPromptEmbedding,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_examples: int = 50,
) -> list:
    """
    Generate detailed per-example predictions for qualitative analysis.

    Returns dictionaries containing:
        query
        gold answer
        gold tokens
        predicted tokens
        exact token-level correctness
    """

    model.eval()
    soft_prompt.eval()

    results = []
    count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = forward_with_soft_prompt(
            model,
            soft_prompt,
            input_ids,
            attention_mask,
        )

        seq_logits = logits[
            :,
            soft_prompt.num_prompt_tokens:,
            :,
        ]

        preds = seq_logits.argmax(dim=-1)

        for i in range(input_ids.size(0)):
            mask = labels[i] != -100

            if mask.sum() == 0:
                continue

            gold_ids = labels[i][mask].cpu().tolist()
            pred_ids = preds[i][mask].cpu().tolist()

            gold_tokens = tokenizer.convert_ids_to_tokens(
                gold_ids
            )

            pred_tokens = tokenizer.convert_ids_to_tokens(
                pred_ids
            )

            gold_str = tokenizer.convert_tokens_to_string(
                gold_tokens
            )

            pred_str = tokenizer.convert_tokens_to_string(
                pred_tokens
            )

            correct = gold_ids == pred_ids

            results.append(
                {
                    "query": batch["query_text"][i],
                    "gold_answer": batch["answer_text"][i],
                    "gold_tokens": gold_str,
                    "predicted_tokens": pred_str,
                    "correct": correct,
                }
            )

            count += 1

            if count >= max_examples:
                return results

    return results


# ---------------------------------------------------------------------------
# Shared multilingual split
# ---------------------------------------------------------------------------

def apply_shared_split(
    dataset,
    split_file: str,
    split_name: str,
    pids=None,
    allow_incomplete: bool = False,
):
    """
    Filter a language-specific PromptTuningDataset using a shared
    multilingual split manifest.

    Facts are matched by:
        (relation, sub_uri, obj_uri)

    This guarantees that the same underlying facts are used across
    languages.

    Returns:
        subset
        metadata
    """

    logger.info(
        f"Loading shared split: {split_file} "
        f"(partition={split_name})"
    )

    with open(split_file, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    required_keys = {"train", "val", "test"}

    missing_keys = required_keys - set(split_manifest)

    if missing_keys:
        raise ValueError(
            "Split file is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    if split_name not in required_keys:
        raise ValueError(
            f"Invalid split {split_name!r}. "
            "Expected one of: train, val, test."
        )

    # ------------------------------------------------------------
    # Convert JSON lists to fact tuples
    # ------------------------------------------------------------

    split_facts = {
        name: {
            tuple(fact)
            for fact in split_manifest[name]
        }
        for name in ("train", "val", "test")
    }

    # ------------------------------------------------------------
    # Check split integrity
    # ------------------------------------------------------------

    if split_facts["train"] & split_facts["val"]:
        raise ValueError(
            "Shared split has train/val overlap."
        )

    if split_facts["train"] & split_facts["test"]:
        raise ValueError(
            "Shared split has train/test overlap."
        )

    if split_facts["val"] & split_facts["test"]:
        raise ValueError(
            "Shared split has val/test overlap."
        )

    target_facts = split_facts[split_name]

    # ------------------------------------------------------------
    # Optional relation filtering
    # ------------------------------------------------------------

    if pids is not None:
        pid_set = set(pids)

        target_facts = {
            fact
            for fact in target_facts
            if fact[0] in pid_set
        }

    # ------------------------------------------------------------
    # Match language-specific dataset examples to shared facts
    # ------------------------------------------------------------

    indices = []
    found_facts = set()

    for idx, sample in enumerate(dataset.samples):
        fact = (
            sample["relation"],
            sample["sub_uri"],
            sample["obj_uri"],
        )

        if fact in target_facts:
            indices.append(idx)
            found_facts.add(fact)

    # ------------------------------------------------------------
    # Verify complete recovery
    # ------------------------------------------------------------

    missing = target_facts - found_facts
    extra = found_facts - target_facts

    if missing and not allow_incomplete:
        examples = sorted(missing)[:10]

        raise ValueError(
            f"{len(missing)} facts from shared {split_name} split "
            f"were not found in the language-specific dataset. "
            f"First missing facts: {examples}"
        )

    if missing:
        logger.warning(
            f"{len(missing)} expected facts were not found."
        )

    if extra:
        raise ValueError(
            f"Unexpected internal error: {len(extra)} extra facts "
            "were selected."
        )

    subset = Subset(
        dataset,
        indices,
    )

    metadata = {
        "split_file": split_file,
        "split": split_name,
        "expected_facts": len(target_facts),
        "found_facts": len(found_facts),
        "missing_facts": len(missing),
        "num_examples": len(subset),
    }

    logger.info(
        f"Applied shared split '{split_name}': "
        f"{len(subset)} examples "
        f"({len(found_facts)}/{len(target_facts)} facts recovered)"
    )

    return subset, metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Fixed-LM Prompt Tuning on X-FACTR"
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language code (e.g., en, nl, tr, el, sw)",
    )

    parser.add_argument(
        "--soft_prompt_path",
        type=str,
        required=True,
        help="Path to trained soft prompt checkpoint (.pt)",
    )

    parser.add_argument(
        "--num_prompt_tokens",
        type=int,
        default=20,
        help="Number of soft prompt tokens P (must match checkpoint)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Base model name",
    )

    parser.add_argument(
        "--probe",
        type=str,
        default="mlamaf",
        choices=["mlama", "mlamaf", "lama"],
        help="Dataset variant",
    )

    parser.add_argument(
        "--portion",
        type=str,
        default="trans",
        choices=["trans", "non", "all"],
    )

    parser.add_argument(
        "--pids",
        type=str,
        default=None,
        help="Comma-separated relation IDs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_mask",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of dataset examples before split filtering",
    )

    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=(
            "Path to shared multilingual train/val/test "
            "split JSON"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help=(
            "Partition from --split_file to evaluate "
            "(default: test)"
        ),
    )

    parser.add_argument(
        "--compare_zero_shot",
        action="store_true",
        help="Also evaluate zero-shot (no prompt) baseline",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-example predictions",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    parser.add_argument(
        "--no_inflect",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Safety checks
    # -----------------------------------------------------------------------

    if args.limit is not None and args.split_file is not None:
        raise ValueError(
            "--limit should not be used together with --split_file. "
            "The shared split requires the complete language dataset "
            "so that every expected fact can be recovered."
        )

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------

    if args.device:
        device = torch.device(args.device)

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Load model & tokenizer
    # -----------------------------------------------------------------------

    logger.info(
        f"Loading model: {args.model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name
    )

    model.to(device)
    model.eval()

    freeze_model(model)

    # -----------------------------------------------------------------------
    # Load soft prompt
    # -----------------------------------------------------------------------

    logger.info(
        f"Loading soft prompt from: "
        f"{args.soft_prompt_path}"
    )

    soft_prompt = SoftPromptEmbedding.load(
        path=args.soft_prompt_path,
        num_prompt_tokens=args.num_prompt_tokens,
        embedding_dim=768,
    )

    soft_prompt.to(device)

    logger.info(
        f"Loaded soft prompt with "
        f"P={args.num_prompt_tokens} "
        f"({soft_prompt.num_trainable_params:,} params)"
    )

    # -----------------------------------------------------------------------
    # Load complete language-specific dataset
    # -----------------------------------------------------------------------

    pids = (
        args.pids.split(",")
        if args.pids
        else None
    )

    dataset = PromptTuningDataset(
        tokenizer=tokenizer,
        lang=args.lang,
        probe=args.probe,
        portion=args.portion,
        pids=pids,
        num_mask=args.num_mask,
        use_inflection=not args.no_inflect,
        max_seq_len=args.max_seq_len,
        limit=args.limit,
    )

    logger.info(
        f"Loaded language dataset: "
        f"{len(dataset)} samples "
        f"(stats: {dataset.stats})"
    )

    if len(dataset) == 0:
        logger.error(
            "No evaluation samples found."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Apply shared multilingual split if supplied
    # -----------------------------------------------------------------------

    split_metadata = None

    if args.split_file:
        eval_dataset, split_metadata = apply_shared_split(
            dataset=dataset,
            split_file=args.split_file,
            split_name=args.split,
            pids=pids,
        )

    else:
        eval_dataset = dataset

        logger.warning(
            "No --split_file supplied. "
            "Evaluating the complete language-specific dataset."
        )

    if len(eval_dataset) == 0:
        raise ValueError(
            "Evaluation dataset is empty after filtering."
        )

    logger.info(
        f"Final evaluation set: "
        f"{len(eval_dataset)} samples"
    )

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    # -----------------------------------------------------------------------
    # Results metadata
    # -----------------------------------------------------------------------

    results = {
        "metadata": {
            "language": args.lang,
            "model_name": args.model_name,
            "probe": args.probe,
            "portion": args.portion,
            "num_prompt_tokens": args.num_prompt_tokens,
            "num_mask": args.num_mask,
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
            "soft_prompt_path": args.soft_prompt_path,
            "num_language_examples": len(dataset),
            "num_evaluation_examples": len(eval_dataset),
            "use_inflection": not args.no_inflect,
        }
    }

    if split_metadata is not None:
        results["metadata"]["shared_split"] = split_metadata

    # -----------------------------------------------------------------------
    # Evaluate with trained soft prompt
    # -----------------------------------------------------------------------

    logger.info(
        "Evaluating with trained soft prompts..."
    )

    prompt_metrics = evaluate_with_prompt(
        model,
        soft_prompt,
        eval_loader,
        loss_fn,
        device,
    )

    results["prompt_tuned"] = prompt_metrics

    logger.info(
        f"Prompt-tuned mBERT | "
        f"Loss: {prompt_metrics['loss']:.4f} | "
        f"Acc@1: {prompt_metrics['accuracy']:.4f}"
    )

    # -----------------------------------------------------------------------
    # Evaluate zero-shot baseline
    # -----------------------------------------------------------------------

    if args.compare_zero_shot:
        logger.info(
            "Evaluating zero-shot "
            "(no prompt) baseline..."
        )

        zero_metrics = evaluate_zero_shot(
            model,
            eval_loader,
            loss_fn,
            device,
        )

        results["zero_shot"] = zero_metrics

        logger.info(
            f"Zero-shot mBERT | "
            f"Loss: {zero_metrics['loss']:.4f} | "
            f"Acc@1: {zero_metrics['accuracy']:.4f}"
        )

        delta = (
            prompt_metrics["accuracy"]
            - zero_metrics["accuracy"]
        )

        results["accuracy_delta"] = delta

        logger.info(
            f"Improvement: {delta:+.4f} accuracy"
        )

    # -----------------------------------------------------------------------
    # Detailed predictions
    # -----------------------------------------------------------------------

    if args.detailed:
        logger.info(
            "Generating detailed predictions..."
        )

        predictions = detailed_predictions(
            model,
            soft_prompt,
            eval_loader,
            tokenizer,
            device,
            max_examples=50,
        )

        results["predictions"] = predictions

        logger.info(
            "--- Sample Predictions "
            "(showing up to 20) ---"
        )

        for pred in predictions[:20]:
            status = (
                "✓"
                if pred["correct"]
                else "✗"
            )

            logger.info(
                f"  {status} Query: "
                f"{pred['query']}\n"
                f"        Gold: "
                f"{pred['gold_answer']} "
                f"({pred['gold_tokens']})\n"
                f"        Pred: "
                f"{pred['predicted_tokens']}"
            )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    if args.output:
        output_dir = dirname(
            abspath(args.output)
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        with open(
            args.output,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            f"Results saved to {args.output}"
        )

    logger.info(
        "Evaluation complete."
    )


if __name__ == "__main__":
    main()

    