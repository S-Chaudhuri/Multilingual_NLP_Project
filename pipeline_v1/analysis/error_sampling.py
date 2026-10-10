"""
Sample incorrect predictions for manual failure-mode annotation.

Follows the protocol of `scripts/ana.py --task error` (a random sample of
incorrect predictions per relation), applied to the per-fact predictions
written by transfer_eval.py. scripts/ana.py itself is not used or modified,
because it reads probe.py's CSV output format.

For every selected language, up to --per_relation incorrect predictions are
drawn per relation (seeded). Each row shows both
    ranked_prediction  the top-ranked candidate (what EM scores), always an
                       object entity of that relation, and
    greedy_prediction  the teacher-forced argmax tokens at the gold length,
                       i.e. what the model "writes" freely; failure modes such
                       as subject repetition or non-informative tokens are
                       visible here.

Annotation codebook (fill in the empty `label` column):
    subject_repetition       the prediction repeats (part of) the subject
    wrong_entity_type        an entity of the wrong type (e.g. a city for a language)
    language_specific_bias   a plausible entity favoured by the query language
                             (e.g. a country/city associated with that language)
    non_informative          punctuation, function words, subword fragments
    other                    anything else (related concept, alias of the gold, ...)

`auto_label` pre-fills subject_repetition and non_informative with simple
heuristics; check every row by hand.

Example:
    python pipeline_v1/analysis/error_sampling.py \\
        --model mbert_base --run en-nl-ru-el --seed 42 --per_relation 2
    # -> pipeline_v1/results/transfer/annotation/errors_mbert_base_en-nl-ru-el_seed42.csv
"""

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

COLUMNS = [
    "model", "run", "seed", "lang", "role", "relation", "sub_uri", "obj_uri",
    "sub_label", "query", "gold", "ranked_prediction", "greedy_prediction",
    "auto_label", "label", "notes",
]


def normalise(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = text.replace("##", "")
    return re.sub(r"[\W_]+", " ", text).strip()


def auto_label(row):
    greedy = normalise(row.get("prompt_greedy"))
    ranked = normalise(row.get("prompt_pred_shared"))
    subject = normalise(row.get("sub_label"))

    if len(greedy.replace(" ", "")) <= 1:
        return "non_informative"
    if subject and (greedy in subject or subject in greedy or ranked == subject):
        return "subject_repetition"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Sample errors for annotation")
    parser.add_argument("--results_root", default="pipeline_v1/results/transfer")
    parser.add_argument("--model", required=True, help="e.g. mbert_base or xlmr_base")
    parser.add_argument("--run", required=True, help="e.g. en-nl-ru-el")
    parser.add_argument("--seed", type=int, default=42, help="Training seed of the run")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Languages to sample (default: the run's zero-shot targets)")
    parser.add_argument("--per_relation", type=int, default=2,
                        help="Incorrect predictions per relation and language "
                             "(scripts/ana.py uses 10)")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.results_root) / args.model / args.run / f"seed{args.seed}" / "predictions.jsonl"
    with path.open("r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]

    langs = args.langs or sorted({p["lang"] for p in preds if p["role"] == "zero-shot"})
    rng = random.Random(args.sample_seed)

    errors = defaultdict(list)  # (lang, relation) -> rows
    for p in preds:
        if p.get("scored") and p["lang"] in langs and p.get("prompt_correct_shared") is False:
            errors[(p["lang"], p["relation"])].append(p)

    rows = []
    for (lang, rel) in sorted(errors):
        candidates = sorted(errors[(lang, rel)], key=lambda p: (p["sub_uri"], p["obj_uri"]))
        for p in rng.sample(candidates, min(args.per_relation, len(candidates))):
            rows.append({
                "model": args.model, "run": args.run, "seed": args.seed,
                "lang": lang, "role": p["role"], "relation": rel,
                "sub_uri": p["sub_uri"], "obj_uri": p["obj_uri"],
                "sub_label": p.get("sub_label", ""), "query": p.get("query", ""),
                "gold": p.get("gold", ""),
                "ranked_prediction": p.get("prompt_pred_shared", ""),
                "greedy_prediction": p.get("prompt_greedy", ""),
                "auto_label": auto_label(p), "label": "", "notes": "",
            })

    output = Path(args.output) if args.output else (
        Path(args.results_root) / "annotation"
        / f"errors_{args.model}_{args.run}_seed{args.seed}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so that spreadsheet programs show Cyrillic/Greek/Hangul correctly.
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    per_lang = defaultdict(int)
    for r in rows:
        per_lang[r["lang"]] += 1
    print(f"Sampled {len(rows)} errors {dict(per_lang)} -> {output}")


if __name__ == "__main__":
    main()
