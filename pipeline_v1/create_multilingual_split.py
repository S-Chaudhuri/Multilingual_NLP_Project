"""
Create a deterministic multilingual train/validation/test split for X-FACTR.

Only facts that exist in ALL requested languages are retained. Splitting is
performed independently within each relation so that train/val/test remain
approximately relation-stratified.

A fact is identified by:
    (relation, sub_uri, obj_uri)

The resulting split can therefore be reused across languages.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from pipeline_v1.prompt_tuning_data import PromptTuningDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["en", "fr", "nl"])
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--probe", default="mlamaf")
    parser.add_argument("--portion", default="trans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--output",
        default="splits/shared_en_fr_nl_seed42.json",
    )
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    facts_by_lang = {}

    # ------------------------------------------------------------
    # Load the fact universe for each language
    # ------------------------------------------------------------
    for lang in args.langs:
        print(f"Loading {lang}...")

        dataset = PromptTuningDataset(
            tokenizer=tokenizer,
            lang=lang,
            probe=args.probe,
            portion=args.portion,
            pids=None,
            num_mask=1,
            use_inflection=True,
            max_seq_len=128,
            limit=None,
        )

        facts = {
            (sample["relation"], sample["sub_uri"], sample["obj_uri"])
            for sample in dataset.samples
        }

        facts_by_lang[lang] = facts

        print(
            f"  {len(dataset)} samples | "
            f"{len(facts)} unique facts"
        )

    # ------------------------------------------------------------
    # Keep only facts available in every language
    # ------------------------------------------------------------
    common_facts = set.intersection(
        *(facts_by_lang[lang] for lang in args.langs)
    )

    print()
    print("=" * 60)
    print("SHARED FACT UNIVERSE")
    print("=" * 60)
    print(f"Languages: {args.langs}")
    print(f"Common facts: {len(common_facts)}")

    # ------------------------------------------------------------
    # Group common facts by relation
    # ------------------------------------------------------------
    by_relation = defaultdict(list)

    for fact in common_facts:
        relation = fact[0]
        by_relation[relation].append(fact)

    rng = random.Random(args.seed)

    train = []
    val = []
    test = []

    relation_counts = {}

    # ------------------------------------------------------------
    # Stratified split within each relation
    # ------------------------------------------------------------
    for relation in sorted(by_relation):
        facts = sorted(by_relation[relation])
        rng.shuffle(facts)

        n = len(facts)

        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        relation_train = facts[:n_train]
        relation_val = facts[n_train:n_train + n_val]
        relation_test = facts[n_train + n_val:]

        train.extend(relation_train)
        val.extend(relation_val)
        test.extend(relation_test)

        relation_counts[relation] = {
            "total": n,
            "train": len(relation_train),
            "val": len(relation_val),
            "test": len(relation_test),
        }

    # ------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set

    assert train_set | val_set | test_set == common_facts

    # Sort before serialization so output is stable/readable.
    train = sorted(train)
    val = sorted(val)
    test = sorted(test)

    result = {
        "metadata": {
            "languages": args.langs,
            "model": args.model,
            "probe": args.probe,
            "portion": args.portion,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": test_ratio,
            "num_common_facts": len(common_facts),
            "stratified_by_relation": True,
        },
        "relation_counts": relation_counts,
        "train": [list(x) for x in train],
        "val": [list(x) for x in val],
        "test": [list(x) for x in test],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 60)
    print("FINAL SPLIT")
    print("=" * 60)
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Total: {len(train) + len(val) + len(test)}")
    print()
    print("Overlap:")
    print(f"  Train ∩ Val:  {len(train_set & val_set)}")
    print(f"  Train ∩ Test: {len(train_set & test_set)}")
    print(f"  Val ∩ Test:   {len(val_set & test_set)}")
    print()
    print(f"Saved → {output_path}")
    print()
    print("SUCCESS: multilingual split created.")
    

if __name__ == "__main__":
    main()
