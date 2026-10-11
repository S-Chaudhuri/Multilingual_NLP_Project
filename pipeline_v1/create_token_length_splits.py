"""
Create language-specific single-token and multi-token evaluation subsets
from an existing shared multilingual X-FACTR test split.

The canonical multilingual split is NOT modified.

For each language, target length is determined using the tokenizer of the
specified model. A fact is classified as:

    single_token: target object -> exactly 1 model token
    multi_token:  target object -> more than 1 model token

The underlying test facts remain those from the shared multilingual split.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from pipeline_v1.prompt_tuning_data import PromptTuningDataset


def fact_key(sample):
    return (
        sample["relation"],
        sample["sub_uri"],
        sample["obj_uri"],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "nl", "tr", "el", "sw"],
    )
    parser.add_argument(
        "--model",
        default="bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--split_file",
        default="splits/shared_en_nl_tr_el_sw_seed42.json",
    )
    parser.add_argument(
        "--probe",
        default="mlamaf",
    )
    parser.add_argument(
        "--portion",
        default="trans",
    )
    parser.add_argument(
        "--num_mask",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--output",
        default="splits/token_length_test_en_nl_tr_el_sw.json",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.split_file, "r", encoding="utf-8") as f:
        shared_split = json.load(f)

    test_facts = {tuple(x) for x in shared_split["test"]}

    print("=" * 100)
    print("TOKEN-LENGTH EVALUATION SPLIT")
    print("=" * 100)
    print(f"Model:            {args.model}")
    print(f"Tokenizer:        {tokenizer.__class__.__name__}")
    print(f"Shared split:     {args.split_file}")
    print(f"Shared test size: {len(test_facts)}")
    print(f"Languages:        {args.langs}")

    output = {
        "metadata": {
            "model": args.model,
            "tokenizer": tokenizer.__class__.__name__,
            "source_split": args.split_file,
            "probe": args.probe,
            "portion": args.portion,
            "num_mask": args.num_mask,
            "max_seq_len": args.max_seq_len,
            "languages": args.langs,
            "num_shared_test_facts": len(test_facts),
            "definition": {
                "single_token": "target object tokenizes to exactly 1 model token",
                "multi_token": "target object tokenizes to more than 1 model token",
            },
        },
        "languages": {},
    }

    all_passed = True

    for lang in args.langs:
        print()
        print("=" * 100)
        print(f"LANGUAGE: {lang}")
        print("=" * 100)

        dataset = PromptTuningDataset(
            tokenizer=tokenizer,
            lang=lang,
            probe=args.probe,
            portion=args.portion,
            pids=None,
            num_mask=args.num_mask,
            use_inflection=True,
            max_seq_len=args.max_seq_len,
            limit=None,
        )

        # Map each fact to its dataset sample.
        sample_by_fact = {}

        for sample in dataset.samples:
            key = fact_key(sample)

            if key in test_facts:
                if key in sample_by_fact:
                    raise ValueError(
                        f"Duplicate shared test fact in {lang}: {key}"
                    )
                sample_by_fact[key] = sample

        found_facts = set(sample_by_fact)
        missing = test_facts - found_facts

        if missing:
            all_passed = False
            print(f"ERROR: {len(missing)} shared test facts are missing.")
            for fact in sorted(missing)[:10]:
                print("  ", fact)
            continue

        single_token = []
        multi_token = []
        token_length_counts = Counter()

        for fact in sorted(test_facts):
            sample = sample_by_fact[fact]

            # Use the final target string actually produced by the
            # language-aware dataset pipeline.
            target = sample["answer_text"]

            token_ids = tokenizer.encode(
                target,
                add_special_tokens=False,
            )

            token_length = len(token_ids)
            token_length_counts[token_length] += 1

            record = {
                "fact": list(fact),
                "target": target,
                "num_target_tokens": token_length,
                "target_tokens": tokenizer.convert_ids_to_tokens(token_ids),
            }

            if token_length == 1:
                single_token.append(record)
            elif token_length > 1:
                multi_token.append(record)
            else:
                raise ValueError(
                    f"Target produced zero tokens in {lang}: "
                    f"{fact} -> {target!r}"
                )

        # Safety checks.
        single_facts = {
            tuple(record["fact"])
            for record in single_token
        }
        multi_facts = {
            tuple(record["fact"])
            for record in multi_token
        }

        assert not single_facts & multi_facts
        assert single_facts | multi_facts == test_facts

        print(f"Dataset samples:   {len(dataset):,}")
        print(f"Shared test facts: {len(test_facts):,}")
        print(f"Single-token:      {len(single_token):,}")
        print(f"Multi-token:       {len(multi_token):,}")

        print()
        print("TOKEN LENGTH DISTRIBUTION")
        print("-" * 60)

        for length in sorted(token_length_counts):
            print(
                f"{length:>2} token(s): "
                f"{token_length_counts[length]:>5}"
            )

        print()
        print(
            "Partition check: "
            f"{len(single_token)} + {len(multi_token)} "
            f"= {len(single_token) + len(multi_token)}"
        )

        assert len(single_token) + len(multi_token) == len(test_facts)

        output["languages"][lang] = {
            "num_test_facts": len(test_facts),
            "num_single_token": len(single_token),
            "num_multi_token": len(multi_token),
            "token_length_distribution": {
                str(k): v
                for k, v in sorted(token_length_counts.items())
            },
            "single_token": single_token,
            "multi_token": multi_token,
        }

        print(f"PASS: {lang}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)

    if all_passed and len(output["languages"]) == len(args.langs):
        print("PASS: token-length evaluation subsets created for all languages.")
        print(f"Saved -> {output_path}")
    else:
        print("WARNING: one or more languages failed validation.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
