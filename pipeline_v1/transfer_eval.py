"""
Cross-lingual evaluation of one soft prompt for the transfer-scaling grid.

A trained soft prompt (and, for reference, the frozen model without a prompt)
is evaluated on the shared TEST split of every evaluation language. Each
language is labelled "source" (seen in training) or "zero-shot" (never seen).

Metrics
-------
EM (primary): candidate-ranking Exact Match, exactly as in prompt_ensemble.py
    (build_candidates / score_member / length_normalise / em_metrics, which are
    imported unchanged). Every candidate object of a relation is scored with
    as many [MASK] tokens as it has tokens, scores are length-normalised with
    exponent alpha, and the top-ranked candidate must equal the gold answer.
    Macro-averaged over relations; also reported for single-/multi-token gold.

    Two adjustments for the transfer setting:
      * alpha is selected on the validation split of the SOURCE languages
        (mean macro EM over source languages) and then applied unchanged to
        every evaluation language, so no target-language labels are used.
        The per-language oracle alpha is reported as a diagnostic only.
      * candidate pool:
          shared (primary): only objects of facts in the shared 5-language
                            split, so every language ranks the same objects;
          full (secondary): every object of the relation in the language's
                            full dataset (the definition used in the ablation).
        Both are computed from one scoring pass (shared is a subset of full).

token_acc (secondary): teacher-forced token accuracy at the gold number of
    masks, averaged over batches as in prompt_tuning_eval.py.

Outputs (in --output_dir):
    metrics.json       all aggregate metrics
    predictions.jsonl  one line per (language, test fact) with the ranked
                       predictions, greedy token predictions and correctness,
                       for the fertility, overlap and error analyses

Example:
    python pipeline_v1/transfer_eval.py \\
        --soft_prompt_path checkpoints/transfer/mbert_base/en-nl/seed42/soft_prompt_en-nl_P20_best.pt \\
        --source_langs en nl \\
        --model_name bert-base-multilingual-cased \\
        --output_dir pipeline_v1/results/transfer/mbert_base/en-nl/seed42
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from os.path import abspath, dirname, join

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

PIPELINE_DIR = dirname(abspath(__file__))
ROOT = dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from soft_prompt import SoftPromptEmbedding
from prompt_tuning_data import collate_fn
from prompt_tuning_train import freeze_model, masked_lm_logits
from prompt_ensemble import build_candidates, em_metrics, length_normalise, score_member
from multilingual_data import fact_key, load_language_datasets, load_split, split_subset

logger = logging.getLogger("transfer_eval")

DEFAULT_SPLIT = "splits/shared_en_nl_ru_el_ko_rulc_seed42.json"
ALL_LANGS = ["en", "nl", "ru", "el", "ko"]
POOLS = ("shared", "full")
LENGTH_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 10 ** 6))


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else join(ROOT, path)


# ---------------------------------------------------------------------------
# Examples and candidates
# ---------------------------------------------------------------------------

def group_examples_with_meta(subset, candidates, tokenizer, sub_labels):
    """
    Same grouping as prompt_ensemble.group_examples, plus fact metadata.

    Returns relation -> {"queries", "gold", "gold_len", "facts", "answers",
    "sub_labels"} and the list of facts skipped because their gold answer is
    longer than the mask limit (not a candidate).
    """
    grouped = defaultdict(lambda: {
        "queries": [], "gold": [], "gold_len": [],
        "facts": [], "answers": [], "sub_labels": [],
    })
    skipped = []

    for idx in subset.indices:
        sample = subset.dataset.samples[idx]
        rel = sample["relation"]
        ids = tuple(tokenizer.encode(sample["answer_text"], add_special_tokens=False))
        key = fact_key(sample)

        if rel not in candidates or ids not in candidates[rel]["index"]:
            skipped.append(key)
            continue

        g = grouped[rel]
        g["queries"].append(sample["query_text"])
        g["gold"].append(candidates[rel]["index"][ids])
        g["gold_len"].append(len(ids))
        g["facts"].append(key)
        g["answers"].append(sample["answer_text"])
        g["sub_labels"].append(sub_labels.get(key, ""))

    result = {}
    for rel, g in sorted(grouped.items()):
        result[rel] = dict(g)
        result[rel]["gold"] = torch.tensor(g["gold"])
        result[rel]["gold_len"] = torch.tensor(g["gold_len"])
    return result, skipped


def shared_pool_masks(dataset, candidates, tokenizer, pool_facts):
    """relation -> bool tensor over candidates: True if the candidate is the
    object of at least one fact in the shared split."""
    in_pool = defaultdict(set)
    for sample in dataset.samples:
        if fact_key(sample) in pool_facts:
            ids = tuple(tokenizer.encode(sample["answer_text"], add_special_tokens=False))
            in_pool[sample["relation"]].add(ids)

    return {
        rel: torch.tensor([ids in in_pool[rel] for ids in cand["ids"]], dtype=torch.bool)
        for rel, cand in candidates.items()
    }


def predict(raw, candidates, pool_masks, alpha, pool):
    """relation -> predicted candidate index (argmax of normalised scores)."""
    predictions = {}
    for rel, scores in raw.items():
        normalised = length_normalise(scores, candidates[rel]["lengths"], alpha)
        if pool == "shared":
            normalised = normalised.masked_fill(~pool_masks[rel], float("-inf"))
        predictions[rel] = normalised.argmax(-1)
    return predictions


def chance_macro(examples, candidates, pool_masks, pool):
    """Macro-average over relations of 1 / |candidates|."""
    values = []
    for rel in examples:
        n = int(pool_masks[rel].sum()) if pool == "shared" else len(candidates[rel]["ids"])
        values.append(1.0 / max(n, 1))
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Teacher-forced token predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def token_predictions(model, soft_prompt, subset, tokenizer, device, batch_size):
    """
    Teacher-forced greedy predictions at the gold number of masks.

    Returns (batch-averaged token accuracy, micro token accuracy,
             fact -> {"greedy", "token_correct", "token_total"}).
    """
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    per_fact = {}
    acc_sum, batches, correct_total, token_total = 0.0, 0, 0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        target_mask = labels != -100

        logits = masked_lm_logits(model, soft_prompt, input_ids, attention_mask, target_mask)
        preds = logits.argmax(-1)
        targets = labels[target_mask]
        example_index = target_mask.nonzero(as_tuple=True)[0]

        hits = (preds == targets)
        if targets.numel():
            acc_sum += hits.float().mean().item()
            batches += 1
        correct_total += int(hits.sum())
        token_total += int(targets.numel())

        for i in range(input_ids.size(0)):
            rows = example_index == i
            pred_ids = preds[rows].tolist()
            key = (batch["relation"][i], batch["sub_uri"][i], batch["obj_uri"][i])
            per_fact[key] = {
                "greedy": tokenizer.decode(pred_ids).strip(),
                "token_correct": int(hits[rows].sum()),
                "token_total": int(rows.sum()),
            }

    return (
        acc_sum / max(batches, 1),
        correct_total / max(token_total, 1),
        per_fact,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-lingual soft-prompt evaluation")
    parser.add_argument("--soft_prompt_path", required=True)
    parser.add_argument("--source_langs", nargs="+", required=True,
                        help="Languages the prompt was trained on (used to select alpha)")
    parser.add_argument("--eval_langs", nargs="+", default=ALL_LANGS)
    parser.add_argument("--model_name", default="bert-base-multilingual-cased")
    parser.add_argument("--split_file", default=DEFAULT_SPLIT)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1",
                        help="Comma-separated length-normalisation exponents")
    parser.add_argument("--probe", default="mlamaf", choices=["mlama", "mlamaf", "lama"])
    parser.add_argument("--portion", default="trans", choices=["trans", "non", "all"])
    parser.add_argument("--num_mask", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_zero_shot", action="store_true",
                        help="Skip the frozen model without a prompt")
    parser.add_argument("--allow_incomplete_split", action="store_true")
    parser.add_argument("--no_inflect", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eval_per_lang", type=int, default=None,
                        help="DEBUG: evaluate only the first N val/test facts per language")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    source_langs = list(args.source_langs)
    eval_langs = list(dict.fromkeys(args.eval_langs))
    langs = list(dict.fromkeys(eval_langs + source_langs))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Model and members
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    freeze_model(model)

    weights = torch.load(args.soft_prompt_path, map_location="cpu")
    num_prompt_tokens, embedding_dim = weights.shape
    if embedding_dim != model.config.hidden_size:
        raise ValueError(
            f"{args.soft_prompt_path} has embedding size {embedding_dim}, but "
            f"{args.model_name} has hidden size {model.config.hidden_size}"
        )
    soft_prompt = SoftPromptEmbedding.load(
        args.soft_prompt_path, num_prompt_tokens, embedding_dim
    ).to(device)
    soft_prompt.eval()

    members = [("prompt", soft_prompt)]
    if not args.no_zero_shot:
        members.append(("zero_shot", None))

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    split = load_split(resolve(args.split_file))
    pool_facts = split["train"] | split["val"] | split["test"]

    datasets = load_language_datasets(
        tokenizer, langs, probe=args.probe, portion=args.portion,
        num_mask=args.num_mask, use_inflection=not args.no_inflect,
        max_seq_len=args.max_seq_len,
    )

    state = {}
    for lang in langs:
        dataset = datasets[lang]
        sub_labels = {
            (ex["relation"], ex["sub_uri"], ex["obj_uri"]): ex["sub_label"]
            for ex in dataset.examples
        }
        candidates = build_candidates(dataset, tokenizer, args.num_mask)
        pool_masks = shared_pool_masks(dataset, candidates, tokenizer, pool_facts)

        entry = {"candidates": candidates, "pool_masks": pool_masks, "subsets": {},
                 "examples": {}, "skipped": {}, "raw": {}}

        parts = ["val"] + (["test"] if lang in eval_langs else [])
        for part in parts:
            subset = split_subset(
                dataset, split[part], lang, part,
                allow_incomplete=args.allow_incomplete_split,
                limit=args.max_eval_per_lang,
            )
            examples, skipped = group_examples_with_meta(subset, candidates, tokenizer, sub_labels)

            # Every val/test gold answer is, by construction, in the shared pool.
            for rel, g in examples.items():
                assert bool(pool_masks[rel][g["gold"]].all()), f"gold outside shared pool ({lang}, {rel})"

            entry["subsets"][part] = subset
            entry["examples"][part] = examples
            entry["skipped"][part] = skipped

            for name, member_prompt in members:
                logger.info(f"Scoring {name} on {lang}/{part} "
                            f"({sum(len(g['gold']) for g in examples.values())} examples)")
                entry["raw"][(name, part)] = score_member(
                    model, member_prompt, examples, candidates, tokenizer,
                    device, args.batch_size, args.max_seq_len,
                )
        state[lang] = entry

    # -----------------------------------------------------------------------
    # Alpha selection on the SOURCE languages' validation split
    # -----------------------------------------------------------------------
    def val_em(lang, name, pool, alpha):
        e = state[lang]
        preds = predict(e["raw"][(name, "val")], e["candidates"], e["pool_masks"], alpha, pool)
        return em_metrics(preds, e["examples"]["val"])["all"]["macro"]

    alpha_selection = {}
    for name, _ in members:
        alpha_selection[name] = {}
        for pool in POOLS:
            table = {
                str(a): {lang: val_em(lang, name, pool, a) for lang in langs}
                for a in alphas
            }
            score = {a: sum(table[a][l] for l in source_langs) / len(source_langs) for a in table}
            selected = max(alphas, key=lambda a: score[str(a)])
            alpha_selection[name][pool] = {
                "selected_alpha": selected,
                "source_val_em_by_alpha": score,
                "val_em_by_alpha_and_lang": table,
            }
            logger.info(f"{name}/{pool}: alpha {selected} selected on source val {source_langs}")

    # -----------------------------------------------------------------------
    # Test metrics and per-fact predictions
    # -----------------------------------------------------------------------
    languages = {}
    prediction_rows = []

    for lang in eval_langs:
        e = state[lang]
        examples = e["examples"]["test"]
        role = "source" if lang in source_langs else "zero-shot"

        lang_result = {
            "role": role,
            "n_test_facts": len(e["subsets"]["test"]),
            "n_scored": sum(len(g["gold"]) for g in examples.values()),
            "n_skipped_too_long": len(e["skipped"]["test"]),
            "num_relations": len(examples),
            "n_candidates": {
                "shared": {rel: int(e["pool_masks"][rel].sum()) for rel in examples},
                "full": {rel: len(e["candidates"][rel]["ids"]) for rel in examples},
            },
            "chance_em_macro": {
                pool: chance_macro(examples, e["candidates"], e["pool_masks"], pool)
                for pool in POOLS
            },
            "members": {},
        }

        token_info = {}
        ranked = {}

        for name, member_prompt in members:
            acc, acc_micro, per_fact = token_predictions(
                model, member_prompt, e["subsets"]["test"], tokenizer, device, args.batch_size
            )
            token_info[name] = per_fact
            member_result = {"token_acc": acc, "token_acc_micro": acc_micro, "em": {}}

            for pool in POOLS:
                selected = alpha_selection[name][pool]["selected_alpha"]
                by_alpha = {}
                for a in alphas:
                    preds = predict(e["raw"][(name, "test")], e["candidates"], e["pool_masks"], a, pool)
                    by_alpha[str(a)] = em_metrics(preds, examples)
                    if a == selected:
                        ranked[(name, pool)] = preds

                oracle = max(alphas, key=lambda a: val_em(lang, name, pool, a))
                member_result["em"][pool] = {
                    "alpha": selected,
                    "selected": by_alpha[str(selected)],
                    "oracle_alpha": oracle,
                    "oracle": by_alpha[str(oracle)],
                    "by_alpha": by_alpha,
                }

            # EM by gold-length bucket (shared pool, selected alpha), micro.
            buckets = {}
            for label, lo, hi in LENGTH_BUCKETS:
                hits = total = 0
                for rel, g in examples.items():
                    keep = (g["gold_len"] >= lo) & (g["gold_len"] <= hi)
                    hits += int((ranked[(name, "shared")][rel][keep] == g["gold"][keep]).sum())
                    total += int(keep.sum())
                buckets[label] = {"em_micro": hits / total if total else None, "n": total}
            member_result["em_by_gold_length"] = buckets

            lang_result["members"][name] = member_result
            logger.info(
                f"{lang} ({role}) {name}: EM shared "
                f"{member_result['em']['shared']['selected']['all']['macro']:.4f} | "
                f"EM full {member_result['em']['full']['selected']['all']['macro']:.4f} | "
                f"token_acc {acc:.4f}"
            )

        languages[lang] = lang_result

        # Per-fact rows.
        for rel, g in examples.items():
            cand = e["candidates"][rel]
            for i, key in enumerate(g["facts"]):
                row = {
                    "lang": lang, "role": role, "relation": rel,
                    "sub_uri": key[1], "obj_uri": key[2],
                    "sub_label": g["sub_labels"][i], "query": g["queries"][i],
                    "gold": g["answers"][i], "gold_len": int(g["gold_len"][i]),
                    "n_candidates_shared": int(e["pool_masks"][rel].sum()),
                    "n_candidates_full": len(cand["ids"]),
                    "scored": True,
                }
                for name, _ in members:
                    for pool in POOLS:
                        p = int(ranked[(name, pool)][rel][i])
                        row[f"{name}_pred_{pool}"] = cand["texts"][p]
                        row[f"{name}_correct_{pool}"] = p == int(g["gold"][i])
                    info = token_info[name].get(key, {})
                    row[f"{name}_greedy"] = info.get("greedy")
                    row[f"{name}_token_correct"] = info.get("token_correct")
                    row[f"{name}_token_total"] = info.get("token_total")
                prediction_rows.append(row)

        for key in e["skipped"]["test"]:
            row = {"lang": lang, "role": role, "relation": key[0],
                   "sub_uri": key[1], "obj_uri": key[2], "scored": False}
            for name, _ in members:
                info = token_info[name].get(key, {})
                row[f"{name}_greedy"] = info.get("greedy")
                row[f"{name}_token_correct"] = info.get("token_correct")
                row[f"{name}_token_total"] = info.get("token_total")
            prediction_rows.append(row)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    train_config = None
    ckpt_dir = dirname(abspath(args.soft_prompt_path))
    for name in sorted(os.listdir(ckpt_dir)):
        if name.startswith("config_") and name.endswith(".json"):
            with open(join(ckpt_dir, name), "r", encoding="utf-8") as f:
                train_config = json.load(f)
            break

    metrics = {
        "metadata": {
            "soft_prompt_path": args.soft_prompt_path,
            "num_prompt_tokens": int(num_prompt_tokens),
            "model_name": args.model_name,
            "source_langs": source_langs,
            "eval_langs": eval_langs,
            "split_file": args.split_file,
            "alphas": alphas,
            "num_mask": args.num_mask,
            "members": [name for name, _ in members],
            "max_eval_per_lang": args.max_eval_per_lang,
            "use_inflection": not args.no_inflect,
            "train_config": train_config,
        },
        "alpha_selection": alpha_selection,
        "languages": languages,
    }

    with open(join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(join(args.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Saved {join(args.output_dir, 'metrics.json')} and predictions.jsonl "
                f"({len(prediction_rows)} rows)")


if __name__ == "__main__":
    main()
