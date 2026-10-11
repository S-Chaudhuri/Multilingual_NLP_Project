"""
Remove facts with malformed UniMorph inflections from a shared split.

UniMorph sometimes corrupts the entity names it inflects (Israel in the
Russian genitive: Израиль -> рзраиля). For each language that uses UniMorph,
this script re-fills every fact's template, records the entity inflections
that fact triggers, and flags an output as malformed when more than the word
ending changed (stem check below). A fact flagged in any language is removed
from train/val/test, so the split stays shared across languages.

Run it with the same XFACTR_RU_LOWERCASE setting as training, because that
setting changes the Russian inflections.

Example:
    XFACTR_RU_LOWERCASE=1 python pipeline_v1/filter_inflection_noise.py \\
        --split splits/shared_en_nl_ru_el_ko_rulc_seed42.json \\
        --output splits/shared_en_nl_ru_el_ko_rulc_clean_seed42.json
"""

import os
import sys
import json
import argparse
import unicodedata
from collections import Counter, defaultdict
from os.path import dirname, abspath

from transformers import AutoTokenizer

PIPELINE_DIR = dirname(abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from prompt_tuning_data import (
    DATASET,
    Gender,
    UNIMORPH_LANGS,
    _load_entity_gender,
    _load_entity_instance,
    load_examples,
)
import prompt


def strip(text):
    text = unicodedata.normalize("NFD", text.lower().replace("ё", "е"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def is_malformed(label, output):
    """True when the inflection changed more than the word endings."""
    if strip(output) == strip(label):
        return False
    words_in, words_out = strip(label).split(" "), strip(output).split(" ")
    if len(words_in) != len(words_out):
        return True
    for w_in, w_out in zip(words_in, words_out):
        common = 0
        while common < min(len(w_in), len(w_out)) and w_in[common] == w_out[common]:
            common += 1
        # Keep all but at most the last 3 characters; add at most 4.
        if common < max(1, len(w_in) - 3) or len(w_out) - common > 4:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, help="Shared split JSON to filter")
    parser.add_argument("--output", required=True, help="Filtered split JSON")
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--probe", default="mlamaf")
    parser.add_argument("--portion", default="trans")
    parser.add_argument("--num_mask", type=int, default=10)
    args = parser.parse_args()

    with open(args.split, encoding="utf-8") as f:
        split = json.load(f)

    langs = [l for l in split["metadata"]["languages"] if l in UNIMORPH_LANGS]
    print(f"Languages using UniMorph: {langs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    paths = DATASET[args.probe]
    entity2gender = defaultdict(
        lambda: Gender.NONE, _load_entity_gender(paths["entity_gender_path"])
    )
    entity2instance = defaultdict(
        str, _load_entity_instance(paths["entity_instance_path"])
    )

    # Record the entity inflections (tags "N;...") made for one fact at a time.
    calls = []
    original_inflect = prompt.cache_inflect

    def observing_inflect(label, tag, language):
        output = original_inflect(label, tag, language=language)
        if tag.startswith("N;"):
            calls.append((label, output[0]))
        return output

    flagged = set()
    per_lang = {}

    for lang in langs:
        examples, _ = load_examples(
            lang=lang,
            tokenizer=tokenizer,
            probe=args.probe,
            portion=args.portion,
            num_mask=args.num_mask,
            use_inflection=True,
        )

        # Re-fill each fact's template (cached inflections, so this is cheap)
        # and attribute the inflections it triggers to that fact only.
        prompt_model = prompt.Prompt.from_lang(lang, entity2gender, entity2instance)
        prompt.cache_inflect = observing_inflect
        lang_flagged = set()
        malformed = Counter()

        for ex in examples:
            calls.clear()
            filled, _ = prompt_model.fill_x(ex["template"], ex["sub_uri"], ex["sub_label"])
            prompt_model.fill_y(filled, ex["obj_uri"], ex["obj_label"], num_mask=0)
            bad = [(label, out) for label, out in calls if is_malformed(label, out)]
            if bad:
                lang_flagged.add((ex["relation"], ex["sub_uri"], ex["obj_uri"]))
                malformed.update(bad)

        prompt.cache_inflect = original_inflect

        per_lang[lang] = {
            "malformed_forms": len(malformed),
            "facts_loaded": len(examples),
            "facts_flagged": len(lang_flagged),
            "examples": [f"{label} -> {out}" for (label, out), _ in malformed.most_common(10)],
        }
        flagged |= lang_flagged
        print(f"{lang}: {len(malformed)} malformed forms, "
              f"{len(lang_flagged)}/{len(examples)} facts flagged")

    removed = {}
    for part in ("train", "val", "test"):
        kept = [fact for fact in split[part] if tuple(fact) not in flagged]
        removed[part] = len(split[part]) - len(kept)
        split[part] = kept

    by_relation = Counter(fact[0] for part in ("train", "val", "test") for fact in split[part])
    for relation, counts in split.get("relation_counts", {}).items():
        counts["after_inflection_filter"] = by_relation.get(relation, 0)

    split["metadata"]["inflection_filter"] = {
        "source_split": args.split,
        "ru_lowercase_inflection": prompt.RU_LOWERCASE_INFLECTION,
        "languages_checked": per_lang,
        "removed": removed,
        "kept": {part: len(split[part]) for part in ("train", "val", "test")},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"Removed {removed} -> kept "
          f"{ {p: len(split[p]) for p in ('train', 'val', 'test')} }")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
