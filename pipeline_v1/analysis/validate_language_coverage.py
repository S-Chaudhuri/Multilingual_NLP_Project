#!/usr/bin/env python3

"""
Audit multilingual coverage of the X-FACTR mTRExf dataset.

The underlying multilingual entity-label resource contains labels for many
language codes, but the main analysis is restricted to the 23 languages
officially supported by the X-FACTR benchmark.

Reports, for each official X-FACTR language:
- entities with a label
- usable facts (subject + object both translated)
- coverage relative to all base facts
- number of represented relations
- min/max facts per relation
- missing relations

This is used to justify multilingual language selection.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


# Official X-FACTR benchmark languages.
# The multilingual entity-label metadata contains many more language codes,
# but X-FACTR provides manually created prompts for these 23 languages.
XFACTR_LANGUAGES = [
    "en", "fr", "nl", "ru", "es", "ja", "vi", "zh", "hu", "ko",
    "tr", "he", "el", "war", "mr", "mg", "bn", "tl", "sw", "pa",
    "ceb", "yo", "ilo",
]

ROOT = Path(__file__).resolve().parents[2]

ENTITY_LANG_PATH = ROOT / "data" / "mTRExf_unicode_escape.txt"
FACT_ROOT = ROOT / "data" / "mTRExf" / "sub"
PROMPT_PATH = ROOT / "data" / "TREx_prompts.csv"


def load_entity_languages(path):
    entity2lang = defaultdict(dict)
    malformed = 0

    # Explicit UTF-8 decoding also acts as an integrity check.
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
                    malformed += 1
                    continue

                entity2lang[entity][lang] = label.strip('"')

    return entity2lang, malformed


def load_prompt_languages(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            x for x in reader.fieldnames
            if x not in {"pid", "relation"}
        ]


def load_facts(root):
    facts = []
    relation_counts = Counter()

    for path in sorted(root.glob("*.jsonl")):
        pid = path.stem

        with path.open("r", encoding="utf-8", errors="strict") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    fact = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON: {path}:{line_no}: {exc}"
                    ) from exc

                required = {
                    "predicate_id",
                    "sub_uri",
                    "obj_uri",
                    "sub_label",
                    "obj_label",
                }

                missing = required - set(fact)

                if missing:
                    raise RuntimeError(
                        f"{path}:{line_no} missing fields: "
                        f"{sorted(missing)}"
                    )

                facts.append(fact)
                relation_counts[pid] += 1

    return facts, relation_counts


def main():
    print("=" * 90)
    print("X-FACTR mTRExf MULTILINGUAL DATA VALIDATION")
    print("=" * 90)

    print("\nLoading multilingual entity labels...")
    entity2lang, malformed = load_entity_languages(ENTITY_LANG_PATH)

    print("Loading factual triples...")
    facts, base_relation_counts = load_facts(FACT_ROOT)

    prompt_languages = set(load_prompt_languages(PROMPT_PATH))

    # The entity-label metadata contains substantially more languages than
    # those used by the X-FACTR benchmark. We count them for transparency,
    # but restrict the coverage analysis below to XFACTR_LANGUAGES.
    all_entity_label_languages = sorted({
        lang
        for labels in entity2lang.values()
        for lang in labels
    })

    print("\nDATA INTEGRITY")
    print("-" * 90)
    print(f"Base facts:                         {len(facts):,}")
    print(f"Relations:                          {len(base_relation_counts)}")
    print(f"Entities with language data:        {len(entity2lang):,}")
    print(
        f"Entity-label language codes:        "
        f"{len(all_entity_label_languages)}"
    )
    print(f"Official X-FACTR languages:         {len(XFACTR_LANGUAGES)}")
    print(f"Malformed language entries:         {malformed}")
    print("UTF-8 decoding:                      PASS")

    # Check that the official benchmark languages are represented in the
    # prompt file. This guards against accidentally analysing unsupported
    # languages or a mismatched prompt file.
    missing_prompt_languages = [
        lang
        for lang in XFACTR_LANGUAGES
        if lang not in prompt_languages
    ]

    if missing_prompt_languages:
        print(
            "WARNING: Official X-FACTR languages missing from prompt file: "
            + ", ".join(missing_prompt_languages)
        )

    print("\nLANGUAGE COVERAGE — OFFICIAL X-FACTR LANGUAGES")
    print("-" * 90)

    header = (
        f"{'LANG':<7}"
        f"{'ENTITIES':>11}"
        f"{'FACTS':>11}"
        f"{'COVERAGE':>11}"
        f"{'RELS':>8}"
        f"{'MIN/REL':>10}"
        f"{'MAX/REL':>10}"
    )

    print(header)
    print("-" * len(header))

    results = []

    # Restrict the main analysis to the 23 official X-FACTR languages.
    # We preserve the benchmark language order rather than sorting all
    # metadata languages by coverage.
    for lang in XFACTR_LANGUAGES:
        entity_count = sum(
            lang in labels for labels in entity2lang.values()
        )

        relation_counts = Counter()

        for fact in facts:
            if (
                lang in entity2lang.get(fact["sub_uri"], {})
                and lang in entity2lang.get(fact["obj_uri"], {})
            ):
                relation_counts[fact["predicate_id"]] += 1

        usable = sum(relation_counts.values())
        coverage = usable / len(facts) if facts else 0.0

        represented = len(relation_counts)
        min_rel = min(relation_counts.values()) if relation_counts else 0
        max_rel = max(relation_counts.values()) if relation_counts else 0

        results.append(
            (
                lang,
                entity_count,
                usable,
                coverage,
                represented,
                min_rel,
                max_rel,
            )
        )

    for (
        lang,
        entity_count,
        usable,
        coverage,
        represented,
        min_rel,
        max_rel,
    ) in results:

        print(
            f"{lang:<7}"
            f"{entity_count:>11,}"
            f"{usable:>11,}"
            f"{coverage:>10.1%}"
            f"{represented:>8}"
            f"{min_rel:>10}"
            f"{max_rel:>10}"
        )

    print("\nMISSING RELATIONS — OFFICIAL X-FACTR LANGUAGES")
    print("-" * 90)

    all_relations = set(base_relation_counts)

    for lang in XFACTR_LANGUAGES:
        represented = set()

        for fact in facts:
            if (
                lang in entity2lang.get(fact["sub_uri"], {})
                and lang in entity2lang.get(fact["obj_uri"], {})
            ):
                represented.add(fact["predicate_id"])

        missing = sorted(all_relations - represented)

        print(
            f"{lang:<7}: "
            f"{len(missing):>2} missing"
            + (f" -> {', '.join(missing)}" if missing else "")
        )

    print("\n" + "=" * 90)
    print("VALIDATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()