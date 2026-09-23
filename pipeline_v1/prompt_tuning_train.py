"""
Fixed-LM Prompt Tuning training pipeline for mBERT on X-FACTR.

The entire pre-trained mBERT model is frozen. Only the soft prompt embeddings
are optimized via backpropagation.

Usage:
    python pipeline_v1/prompt_tuning_train.py --lang en --num_prompt_tokens 20
    python pipeline_v1/prompt_tuning_train.py --lang en --config configs/prompt_tuning_config.json
"""

import sys
import os
import json
import argparse
import logging
import time
from os.path import dirname, abspath, join, exists

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = dirname(abspath(__file__))
ROOT = dirname(PIPELINE_DIR)  # Multilingual_NLP_Project/

sys.path.insert(0, PIPELINE_DIR)
from soft_prompt import SoftPromptEmbedding
from prompt_tuning_data import PromptTuningDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('prompt_tuning')


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    'model_name': 'bert-base-multilingual-cased',
    'num_prompt_tokens': 20,
    'embedding_dim': 768,
    'init_from_vocab': True,
    'learning_rate': 0.3,
    'weight_decay': 1e-5,
    'warmup_ratio': 0.1,
    'batch_size': 32,
    'epochs': 10,
    'max_seq_len': 128,
    'num_mask': 5,
    'probe': 'mlamaf',
    'portion': 'trans',
    'val_split': 0.1,
    'test_split': 0.1,
    'log_interval': 50,
    'seed': 42,
}


def load_config(config_path: str) -> dict:
    """Load hyperparameters from a JSON config file, falling back to defaults."""
    config = dict(DEFAULTS)
    if config_path and exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        config.update(user_cfg)
        logger.info(f'Loaded config from {config_path}')
    return config


def freeze_model(model: nn.Module) -> None:
    """Freeze ALL parameters of the pretrained model."""
    for param in model.parameters():
        param.requires_grad = False


def verify_frozen(model: nn.Module, soft_prompt: SoftPromptEmbedding) -> None:
    """Verify that only soft prompt parameters are trainable."""
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prompt_trainable = sum(p.numel() for p in soft_prompt.parameters() if p.requires_grad)
    total_model = sum(p.numel() for p in model.parameters())

    assert model_trainable == 0, (
        f'Expected 0 trainable model params, got {model_trainable}'
    )
    assert prompt_trainable == soft_prompt.num_trainable_params, (
        f'Expected {soft_prompt.num_trainable_params} trainable prompt params, '
        f'got {prompt_trainable}'
    )

    logger.info(f'Model parameters (frozen): {total_model:,}')
    logger.info(f'Soft prompt parameters (trainable): {prompt_trainable:,}')
    logger.info(f'Trainable ratio: {prompt_trainable / total_model * 100:.4f}%')


def forward_with_soft_prompt(
    model,
    soft_prompt: SoftPromptEmbedding,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run the frozen MLM with learned soft-prompt embeddings prepended.

    The pretrained model remains frozen. Gradients flow only through the
    learned soft-prompt embeddings.
    """
    # Obtain frozen token embeddings.
    with torch.no_grad():
        word_embeds = model.get_input_embeddings()(input_ids)

    # Prepend the trainable soft prompt.
    combined_embeds = soft_prompt(word_embeds)

    # Extend the attention mask for the prompt positions.
    batch_size = input_ids.size(0)
    num_prompt = soft_prompt.num_prompt_tokens

    prompt_mask = torch.ones(
        batch_size,
        num_prompt,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    extended_attention_mask = torch.cat(
        [prompt_mask, attention_mask],
        dim=1,
    )

    # Use the public Hugging Face forward API for consistent embedding
    # processing during both training and evaluation.
    outputs = model(
        inputs_embeds=combined_embeds,
        attention_mask=extended_attention_mask,
    )

    return outputs.logits


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_prompt_tokens: int,
    loss_fn: nn.CrossEntropyLoss,
) -> torch.Tensor:
    """
    Compute cross-entropy loss at [MASK] positions only.

    Args:
        logits: (batch, P + seq_len, vocab_size) from the model.
        labels: (batch, seq_len) with -100 at non-MASK positions.
        num_prompt_tokens: P, to offset into the logits.
        loss_fn: CrossEntropyLoss with ignore_index=-100.

    Returns:
        Scalar loss tensor.
    """
    # Slice logits to remove the prompt prefix: keep only the original seq positions
    # logits[:, P:, :] corresponds to the original input tokens
    seq_logits = logits[:, num_prompt_tokens:, :]  # (batch, seq_len, vocab)

    # Flatten for cross-entropy
    vocab_size = seq_logits.size(-1)
    loss = loss_fn(seq_logits.reshape(-1, vocab_size), labels.reshape(-1))
    return loss


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_prompt_tokens: int,
) -> float:
    """Compute accuracy at [MASK] positions (Accuracy@1)."""
    seq_logits = logits[:, num_prompt_tokens:, :]  # (batch, seq_len, vocab)
    preds = seq_logits.argmax(dim=-1)  # (batch, seq_len)

    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).float().sum().item()
    total = mask.sum().item()
    return correct / total


def train_epoch(
    model,
    soft_prompt: SoftPromptEmbedding,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> dict:
    """Run one training epoch."""
    soft_prompt.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = forward_with_soft_prompt(model, soft_prompt, input_ids, attention_mask)

        # Loss
        loss = compute_loss(logits, labels, soft_prompt.num_prompt_tokens, loss_fn)

        # Backward (gradients flow only to soft prompt)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Metrics
        acc = compute_accuracy(logits, labels, soft_prompt.num_prompt_tokens)
        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | '
                f'Loss: {avg_loss:.4f} | Acc@1: {avg_acc:.4f} | LR: {lr:.6f}'
            )

    return {
        'loss': total_loss / max(total_batches, 1),
        'accuracy': total_acc / max(total_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model,
    soft_prompt: SoftPromptEmbedding,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """Evaluate on a validation set."""
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


def main():
    parser = argparse.ArgumentParser(
        description='Fixed-LM Prompt Tuning for mBERT on X-FACTR'
    )
    parser.add_argument('--lang', type=str, required=True,
                        help='Language code (e.g., en, fr, zh)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file')
    parser.add_argument('--num_prompt_tokens', type=int, default=None,
                        help='Number of soft prompt tokens P (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--probe', type=str, default=None,
                        choices=['mlama', 'mlamaf', 'lama'],
                        help='Dataset variant (overrides config)')
    parser.add_argument('--pids', type=str, default=None,
                        help='Comma-separated relation IDs to train on')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Path to a shared multilingual train/val/test split JSON')
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap number of training examples (for quick testing)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save checkpoints (default: checkpoints/)')
    parser.add_argument('--no_inflect', action='store_true',
                        help='Disable language-aware inflection')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda, cpu, or mps')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI overrides
    for key in ['num_prompt_tokens', 'learning_rate', 'epochs', 'batch_size', 'probe']:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Resolve output directory
    output_dir = args.output_dir or join(ROOT, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    # Seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # -----------------------------------------------------------------------
    # 1. Load frozen mBERT model
    # -----------------------------------------------------------------------
    logger.info(f'Loading model: {config["model_name"]}')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForMaskedLM.from_pretrained(config['model_name'])
    model.to(device)
    model.eval()

    # ❄️ Freeze ALL model parameters
    freeze_model(model)

    # -----------------------------------------------------------------------
    # 2. Create soft prompt
    # -----------------------------------------------------------------------
    soft_prompt = SoftPromptEmbedding(
        num_prompt_tokens=config['num_prompt_tokens'],
        embedding_dim=config['embedding_dim'],
        init_from_vocab=config['init_from_vocab'],
        tokenizer=tokenizer,
        model=model,
    )
    soft_prompt.to(device)

    # Verify freezing
    verify_frozen(model, soft_prompt)

    # -----------------------------------------------------------------------
    # 3. Load dataset
    # -----------------------------------------------------------------------
    logger.info(f'Loading data: lang={args.lang}, probe={config["probe"]}, portion={config["portion"]}')
    pids = args.pids.split(',') if args.pids else None

    dataset = PromptTuningDataset(
        tokenizer=tokenizer,
        lang=args.lang,
        probe=config['probe'],
        portion=config['portion'],
        pids=pids,
        num_mask=config['num_mask'],
        use_inflection=not args.no_inflect,
        max_seq_len=config['max_seq_len'],
        limit=args.limit,
    )
    logger.info(f'Loaded {len(dataset)} samples (stats: {dataset.stats})')

    if len(dataset) == 0:
        logger.error('No training samples found. Check language/probe/portion settings.')
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Train / validation / test split
    # -----------------------------------------------------------------------
    if args.split_file:
        # Use a precomputed multilingual split. Facts are identified by the
        # stable X-FACTR key (relation, sub_uri, obj_uri), so the same factual
        # partition is used across languages.
        logger.info(f'Loading shared split: {args.split_file}')

        with open(args.split_file, 'r', encoding='utf-8') as f:
            split_manifest = json.load(f)

        required_keys = {'train', 'val', 'test'}
        missing_keys = required_keys - set(split_manifest)
        if missing_keys:
            raise ValueError(
                f'Split file is missing required keys: {sorted(missing_keys)}'
            )

        # Convert JSON lists to stable fact tuples.
        split_facts = {
            split_name: {tuple(fact) for fact in split_manifest[split_name]}
            for split_name in ('train', 'val', 'test')
        }

        # Sanity check: the supplied split itself must be disjoint.
        if split_facts['train'] & split_facts['val']:
            raise ValueError('Shared split has train/val overlap')
        if split_facts['train'] & split_facts['test']:
            raise ValueError('Shared split has train/test overlap')
        if split_facts['val'] & split_facts['test']:
            raise ValueError('Shared split has val/test overlap')

        # Map the current language-specific dataset back to the shared
        # multilingual fact identities.
        indices = {
            'train': [],
            'val': [],
            'test': [],
        }

        dataset_facts = set()

        for idx, sample in enumerate(dataset.samples):
            fact = (
                sample['relation'],
                sample['sub_uri'],
                sample['obj_uri'],
            )
            dataset_facts.add(fact)

            for split_name in ('train', 'val', 'test'):
                if fact in split_facts[split_name]:
                    indices[split_name].append(idx)
                    break

        # If --pids was supplied, only expect facts from those relations.
        if pids is not None:
            expected_facts = {
                split_name: {
                    fact for fact in split_facts[split_name]
                    if fact[0] in set(pids)
                }
                for split_name in ('train', 'val', 'test')
            }
        else:
            expected_facts = split_facts

        # With --limit, the dataset is intentionally truncated, so exact
        # recovery of the complete shared split is not expected.
        if args.limit is None:
            for split_name in ('train', 'val', 'test'):
                found = {
                    (
                        dataset.samples[idx]['relation'],
                        dataset.samples[idx]['sub_uri'],
                        dataset.samples[idx]['obj_uri'],
                    )
                    for idx in indices[split_name]
                }

                missing = expected_facts[split_name] - found
                if missing:
                    raise ValueError(
                        f'{len(missing)} {split_name} facts from the shared '
                        f'split were not found for language {args.lang}'
                    )

        from torch.utils.data import Subset

        train_dataset = Subset(dataset, indices['train'])
        val_dataset = Subset(dataset, indices['val'])
        test_dataset = Subset(dataset, indices['test'])

        logger.info(
            f'Applied shared multilingual split from {args.split_file}'
        )

    else:
        # Fall back to a deterministic language-specific random split.
        num_samples = len(dataset)

        val_size = max(1, int(num_samples * config['val_split']))
        test_size = max(1, int(num_samples * config['test_split']))
        train_size = num_samples - val_size - test_size

        if train_size < 1:
            raise ValueError(
                f'Dataset too small for train/val/test split: '
                f'N={num_samples}, train={train_size}, '
                f'val={val_size}, test={test_size}'
            )

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config['seed']),
        )

        # Save the exact language-specific split for reproducibility.
        def fact_triples(subset):
            return [
                [
                    dataset.samples[idx]['relation'],
                    dataset.samples[idx]['sub_uri'],
                    dataset.samples[idx]['obj_uri'],
                ]
                for idx in subset.indices
            ]

        split_manifest = {
            'train': fact_triples(train_dataset),
            'val': fact_triples(val_dataset),
            'test': fact_triples(test_dataset),
        }

        split_path = join(
            output_dir,
            f'split_{args.lang}.json',
        )

        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_manifest, f, indent=2)

        logger.info(f'Saved data split → {split_path}')

    logger.info(
        f'Train: {len(train_dataset)} | '
        f'Val: {len(val_dataset)} | '
        f'Test: {len(test_dataset)}'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # -----------------------------------------------------------------------
    # 4. Optimizer & scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        soft_prompt.parameters(),  # ONLY soft prompt params
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(config['warmup_ratio'] * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # -----------------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------------
    logger.info('=' * 60)
    logger.info('Starting Fixed-LM Prompt Tuning')
    logger.info(f'  Prompt tokens (P): {config["num_prompt_tokens"]}')
    logger.info(f'  Trainable params:  {soft_prompt.num_trainable_params:,}')
    logger.info(f'  Epochs:            {config["epochs"]}')
    logger.info(f'  Batch size:        {config["batch_size"]}')
    logger.info(f'  Learning rate:     {config["learning_rate"]}')
    logger.info(f'  Total steps:       {total_steps}')
    logger.info(f'  Warmup steps:      {warmup_steps}')
    logger.info('=' * 60)

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, soft_prompt, train_loader, optimizer, scheduler,
            loss_fn, device, epoch, config['log_interval'],
        )

        # Validate
        val_metrics = evaluate(model, soft_prompt, val_loader, loss_fn, device)

        elapsed = time.time() - t0
        logger.info(
            f'Epoch {epoch}/{config["epochs"]} ({elapsed:.1f}s) | '
            f'Train Loss: {train_metrics["loss"]:.4f} | '
            f'Train Acc: {train_metrics["accuracy"]:.4f} | '
            f'Val Loss: {val_metrics["loss"]:.4f} | '
            f'Val Acc: {val_metrics["accuracy"]:.4f}'
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })

        # Save best checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            ckpt_path = join(output_dir, f'soft_prompt_{args.lang}_P{config["num_prompt_tokens"]}_best.pt')
            soft_prompt.save(ckpt_path)
            logger.info(f'  ✓ Saved best checkpoint → {ckpt_path}')

    # Save final checkpoint
    final_path = join(output_dir, f'soft_prompt_{args.lang}_P{config["num_prompt_tokens"]}_final.pt')
    soft_prompt.save(final_path)
    logger.info(f'Saved final checkpoint → {final_path}')

    # Save training history
    history_path = join(output_dir, f'history_{args.lang}_P{config["num_prompt_tokens"]}.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    logger.info(f'Saved training history → {history_path}')
    logger.info('Training complete.')


if __name__ == '__main__':
    main()
