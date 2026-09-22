"""
Evaluation script for Fixed-LM Prompt Tuning on X-FACTR.

Loads a trained soft prompt checkpoint and evaluates against zero-shot
(no prompt) and prompt-tuned mBERT on the same data.

Usage:
    python pipeline_v1/prompt_tuning_eval.py --lang en --soft_prompt_path checkpoints/soft_prompt_en_P20_best.pt
    python pipeline_v1/prompt_tuning_eval.py --lang en --soft_prompt_path checkpoints/soft_prompt_en_P20_best.pt --compare_zero_shot
"""

import sys
import os
import json
import argparse
import logging
from os.path import dirname, abspath, join, exists

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('prompt_tuning_eval')


@torch.no_grad()
def evaluate_zero_shot(
    model,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """
    Evaluate the frozen model WITHOUT soft prompts (zero-shot baseline).

    Uses model's standard forward pass with input_ids and attention_mask.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Standard forward (no soft prompt)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Loss at [MASK] positions
        vocab_size = logits.size(-1)
        loss = loss_fn(logits.reshape(-1, vocab_size), labels.reshape(-1))

        # Accuracy
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).float().mean().item()
        else:
            acc = 0.0

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

    return {
        'loss': total_loss / max(total_batches, 1),
        'accuracy': total_acc / max(total_batches, 1),
    }


@torch.no_grad()
def evaluate_with_prompt(
    model,
    soft_prompt: SoftPromptEmbedding,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """Evaluate the frozen model WITH trained soft prompts."""
    soft_prompt.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = forward_with_soft_prompt(model, soft_prompt, input_ids, attention_mask)
        loss = compute_loss(logits, labels, soft_prompt.num_prompt_tokens, loss_fn)
        acc = compute_accuracy(logits, labels, soft_prompt.num_prompt_tokens)

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

    return {
        'loss': total_loss / max(total_batches, 1),
        'accuracy': total_acc / max(total_batches, 1),
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

    Returns a list of dicts with query, gold answer, predicted answer, and correctness.
    """
    soft_prompt.eval()
    results = []
    count = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = forward_with_soft_prompt(model, soft_prompt, input_ids, attention_mask)
        seq_logits = logits[:, soft_prompt.num_prompt_tokens:, :]
        preds = seq_logits.argmax(dim=-1)

        for i in range(input_ids.size(0)):
            mask = labels[i] != -100
            if mask.sum() == 0:
                continue

            gold_ids = labels[i][mask].cpu().tolist()
            pred_ids = preds[i][mask].cpu().tolist()

            gold_tokens = tokenizer.convert_ids_to_tokens(gold_ids)
            pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)

            gold_str = tokenizer.convert_tokens_to_string(gold_tokens)
            pred_str = tokenizer.convert_tokens_to_string(pred_tokens)
            correct = gold_ids == pred_ids

            results.append({
                'query': batch['query_text'][i],
                'gold_answer': batch['answer_text'][i],
                'gold_tokens': gold_str,
                'predicted_tokens': pred_str,
                'correct': correct,
            })
            count += 1
            if count >= max_examples:
                return results

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Fixed-LM Prompt Tuning on X-FACTR'
    )
    parser.add_argument('--lang', type=str, required=True,
                        help='Language code (e.g., en, fr, zh)')
    parser.add_argument('--soft_prompt_path', type=str, required=True,
                        help='Path to trained soft prompt checkpoint (.pt)')
    parser.add_argument('--num_prompt_tokens', type=int, default=20,
                        help='Number of soft prompt tokens P (must match checkpoint)')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased',
                        help='Base model name')
    parser.add_argument('--probe', type=str, default='mlamaf',
                        choices=['mlama', 'mlamaf', 'lama'],
                        help='Dataset variant')
    parser.add_argument('--portion', type=str, default='trans',
                        choices=['trans', 'non', 'all'])
    parser.add_argument('--pids', type=str, default=None,
                        help='Comma-separated relation IDs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--num_mask', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap number of eval examples')
    parser.add_argument('--compare_zero_shot', action='store_true',
                        help='Also evaluate zero-shot (no prompt) for comparison')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed per-example predictions')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--no_inflect', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    # -----------------------------------------------------------------------
    # Load model & tokenizer
    # -----------------------------------------------------------------------
    logger.info(f'Loading model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    freeze_model(model)

    # -----------------------------------------------------------------------
    # Load soft prompt
    # -----------------------------------------------------------------------
    logger.info(f'Loading soft prompt from: {args.soft_prompt_path}')
    soft_prompt = SoftPromptEmbedding.load(
        path=args.soft_prompt_path,
        num_prompt_tokens=args.num_prompt_tokens,
        embedding_dim=768,
    )
    soft_prompt.to(device)
    logger.info(f'Loaded soft prompt with P={args.num_prompt_tokens} '
                f'({soft_prompt.num_trainable_params:,} params)')

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    pids = args.pids.split(',') if args.pids else None
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
    logger.info(f'Evaluation set: {len(dataset)} samples')

    if len(dataset) == 0:
        logger.error('No evaluation samples found.')
        sys.exit(1)

    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    results = {}

    # -----------------------------------------------------------------------
    # Evaluate with soft prompt
    # -----------------------------------------------------------------------
    logger.info('Evaluating with trained soft prompts...')
    prompt_metrics = evaluate_with_prompt(model, soft_prompt, eval_loader, loss_fn, device)
    results['prompt_tuned'] = prompt_metrics
    logger.info(
        f'Prompt-tuned mBERT | Loss: {prompt_metrics["loss"]:.4f} | '
        f'Acc@1: {prompt_metrics["accuracy"]:.4f}'
    )

    # -----------------------------------------------------------------------
    # Evaluate zero-shot (optional comparison)
    # -----------------------------------------------------------------------
    if args.compare_zero_shot:
        logger.info('Evaluating zero-shot (no prompt) baseline...')
        zero_metrics = evaluate_zero_shot(model, eval_loader, loss_fn, device)
        results['zero_shot'] = zero_metrics
        logger.info(
            f'Zero-shot mBERT   | Loss: {zero_metrics["loss"]:.4f} | '
            f'Acc@1: {zero_metrics["accuracy"]:.4f}'
        )

        # Improvement
        delta = prompt_metrics['accuracy'] - zero_metrics['accuracy']
        logger.info(f'Improvement: {delta:+.4f} accuracy')

    # -----------------------------------------------------------------------
    # Detailed predictions (optional)
    # -----------------------------------------------------------------------
    if args.detailed:
        logger.info('Generating detailed predictions...')
        predictions = detailed_predictions(
            model, soft_prompt, eval_loader, tokenizer, device, max_examples=50
        )
        results['predictions'] = predictions

        logger.info(f'--- Sample Predictions (showing up to 20) ---')
        for i, pred in enumerate(predictions[:20]):
            status = '✓' if pred['correct'] else '✗'
            logger.info(
                f'  {status} Query: {pred["query"]}\n'
                f'        Gold: {pred["gold_answer"]} ({pred["gold_tokens"]})\n'
                f'        Pred: {pred["predicted_tokens"]}'
            )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    if args.output:
        os.makedirs(dirname(abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f'Results saved to {args.output}')

    logger.info('Evaluation complete.')


if __name__ == '__main__':
    main()
