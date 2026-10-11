#!/usr/bin/env python3

"""
Analyze X-FACTR target tokenization across languages and multilingual models.

For each selected language, this script:
- identifies usable X-FACTR facts whose subject and object have translations;
- tokenizes the translated object label;
- counts single-token vs multi-token targets;
- reports token-length distributions and average token fertility;
- optionally writes fact-level tokenization metadata to JSONL.

The analysis is performed independently for:
    mBERT      -> bert-base-multilingual-cased
    XLM-R base -> xlm-roberta-base
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]

ENTITY_LANG_PATH = ROOT / "data" / "mTRExf_unicode_escape.txt"
FACT_ROOT = ROOT / "data" / "mTRExf" / "sub"

MODELS = {
    "mbert_base": "bert-base-multilingual-cased",
    "xlmr_base": "xlm-roberta-base",
}

# Initial balanced experimental subset.
DEFAULT_LANGUAGES = ["en", "nl", "tr", "el", "sw"]


def load_entity_languages(path):
    """Load entity labels as entity -> language -> label."""
    entity2lang = defaultdict(dict)

    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")
            entity = parts[0]

            for chunk in parts[1:]:
                try:
                    label, lang = chunk.rsplit("@", 1)
                except ValueError:
                    continue

                entity2lang[entity][lang] = label.strip('"')

    return entity2lang


def load_facts(root):
    """Load all base X-FACTR facts."""
    facts = []

    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="strict") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    fact = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON at {path}:{line_no}: {exc}"
                    ) from exc

                facts.append(fact)

    return facts


def analyze_language(lang, facts, entity2lang, tokenizer):
    """
    Analyze translated target labels for one language.

    A fact is considered usable when both subject and object have labels
    in the target language, matching the coverage definition used by the
    multilingual validation script.
    """
    rows = []
    length_counts = Counter()
    relation_counts = Counter()

    for fact in facts:
        sub_uri = fact["sub_uri"]
        obj_uri = fact["obj_uri"]

        sub_label = entity2lang.get(sub_uri, {}).get(lang)
        obj_label = entity2lang.get(obj_uri, {}).get(lang)

        if sub_label is None or obj_label is None:
            continue

        token_ids = tokenizer.encode(
            obj_label,
            add_special_tokens=False,
        )

        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        token_length = len(token_ids)

        length_counts[token_length] += 1
        relation_counts[fact["predicate_id"]] += 1

        rows.append({
            "uuid": fact.get("uuid"),
            "predicate_id": fact["predicate_id"],
            "sub_uri": sub_uri,
            "obj_uri": obj_uri,
            "language": lang,
            "subject_label": sub_label,
            "object_label": obj_label,
            "token_ids": token_ids,
            "tokens": tokens,
            "token_length": token_length,
            "target_type": (
                "single_token" if token_length == 1 else "multi_token"
            ),
        })

    return rows, length_counts, relation_counts


def print_summary(model_key, model_name, lang, rows, length_counts):
    total = len(rows)
    single = length_counts.get(1, 0)
    multi = total - single

    avg_length = (
        sum(length * count for length, count in length_counts.items()) / total
        if total
        else 0.0
    )

    print("\n" + "=" * 80)
    print(f"MODEL: {model_key} ({model_name})")
    print(f"LANGUAGE: {lang}")
    print("=" * 80)

    print(f"Usable facts:          {total:,}")
    print(
        f"Single-token targets:  {single:,} "
        f"({single / total:.1%})"
        if total else
        "Single-token targets:  0"
    )
    print(
        f"Multi-token targets:   {multi:,} "
        f"({multi / total:.1%})"
        if total else
        "Multi-token targets:   0"
    )
    print(f"Average target tokens: {avg_length:.3f}")

    print("\nTarget token-length distribution:")

    for length in sorted(length_counts):
        count = length_counts[length]
        fraction = count / total if total else 0.0
        print(
            f"  {length:>2} token(s): "
            f"{count:>7,} ({fraction:>6.1%})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze X-FACTR target tokenization."
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Languages to analyze.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=list(MODELS),
        help="Models/tokenizers to analyze.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for fact-level JSONL outputs. "
            "If omitted, only summary statistics are printed."
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print("X-FACTR TARGET TOKENIZATION ANALYSIS")
    print("=" * 80)
    print("Languages:", ", ".join(args.languages))
    print("Models:", ", ".join(args.models))

    print("\nLoading multilingual entity labels...")
    entity2lang = load_entity_languages(ENTITY_LANG_PATH)

    print("Loading X-FACTR facts...")
    facts = load_facts(FACT_ROOT)

    print(f"Base facts loaded: {len(facts):,}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        model_name = MODELS[model_key]

        print("\n" + "#" * 80)
        print(f"Loading tokenizer: {model_name}")
        print("#" * 80)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for lang in args.languages:
            rows, length_counts, relation_counts = analyze_language(
                lang=lang,
                facts=facts,
                entity2lang=entity2lang,
                tokenizer=tokenizer,
            )

            print_summary(
                model_key=model_key,
                model_name=model_name,
                lang=lang,
                rows=rows,
                length_counts=length_counts,
            )

            if args.output_dir is not None:
                output_path = (
                    args.output_dir /
                    f"{model_key}_{lang}_target_tokenization.jsonl"
                )

                with output_path.open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    for row in rows:
                        f.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                print(f"Saved: {output_path}")

    print("\n" + "=" * 80)
    print("TOKENIZATION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
