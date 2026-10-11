"""
Create fixed-budget, language-disjoint training partitions for the
multilingual transfer-scaling experiments.

Every run trains on exactly B unique facts in total. With L source languages,
each language receives B / L facts and no fact is assigned to more than one
language. A multilingual run therefore sees the same number of unique facts as
the monolingual control, only spread over more languages.

Procedure (deterministic given --seed):

    1. Select B facts from the TRAIN part of the shared split, stratified by
       relation (largest-remainder quotas). This step depends only on the seed,
       not on the languages, so for a given seed every run (R1-R5) trains on
       the same B unique facts.
    2. Shuffle the selected facts within each relation and deal them
       round-robin to the source languages with one global counter, so every
       language receives exactly B / L facts and per-relation counts differ by
       at most one between languages.

Safety checks (the script fails if any is violated):
    - the shared split itself has disjoint train / val / test parts;
    - every assigned fact comes from the train part;
    - no assigned fact appears in the val or test part
      (no fact is ever used for both training and evaluation);
    - language subsets are pairwise disjoint;
    - every language has exactly B / L facts and their union has B facts.

A fact is identified by (relation, sub_uri, obj_uri), as everywhere else in
the pipeline.

Default budget: B = 8,928 = 32 x 279, the largest multiple of the batch size
(32, itself divisible by L = 1, 2, 4) that fits in the 8,958 train facts of
the keep split. Every run therefore has 279 exactly balanced batches per epoch
(8,928 / 4,464 / 2,232 facts per language for L = 1 / 2 / 4).

Example:
    python pipeline_v1/make_budget_partitions.py \\
        --langs en nl ru el \\
        --budget 8928 \\
        --seed 42
    # -> splits/budget/en-nl-ru-el_B8928_seed42.json
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPLIT = "splits/shared_en_nl_ru_el_ko_rulc_seed42.json"

# Offset for the dealing RNG so that fact selection (step 1) and the
# fact-to-language assignment (step 2) use independent random streams.
DEAL_SEED_OFFSET = 1_000_003


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path):
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    missing = {"train", "val", "test"} - set(manifest)
    if missing:
        raise ValueError(f"{path} is missing split parts: {sorted(missing)}")

    parts = {
        name: {tuple(fact) for fact in manifest[name]}
        for name in ("train", "val", "test")
    }

    if parts["train"] & parts["val"]:
        raise ValueError("Shared split has train/val overlap")
    if parts["train"] & parts["test"]:
        raise ValueError("Shared split has train/test overlap")
    if parts["val"] & parts["test"]:
        raise ValueError("Shared split has val/test overlap")

    return manifest, parts


def stratified_quotas(counts: dict, budget: int) -> dict:
    """
    Largest-remainder allocation of `budget` facts over relations,
    proportional to each relation's number of train facts.
    """
    total = sum(counts.values())
    if budget > total:
        raise ValueError(
            f"Budget {budget} exceeds the {total} train facts in the split."
        )

    exact = {rel: budget * n / total for rel, n in counts.items()}
    quotas = {rel: int(x) for rel, x in exact.items()}
    remainder = budget - sum(quotas.values())

    # Ties are broken by relation id so the allocation is deterministic.
    by_fraction = sorted(
        counts,
        key=lambda rel: (-(exact[rel] - quotas[rel]), rel),
    )
    for rel in by_fraction[:remainder]:
        quotas[rel] += 1

    assert sum(quotas.values()) == budget
    assert all(quotas[rel] <= counts[rel] for rel in counts)
    return quotas


def make_partitions(train_facts, langs, budget, seed):
    """Return (selected_facts, {lang: [facts]})."""

    by_relation = defaultdict(list)
    for fact in train_facts:
        by_relation[fact[0]].append(fact)

    # Step 1: stratified selection of `budget` facts (independent of langs).
    select_rng = random.Random(seed)
    quotas = stratified_quotas(
        {rel: len(facts) for rel, facts in by_relation.items()},
        budget,
    )

    selected_by_relation = {}
    for rel in sorted(by_relation):
        facts = sorted(by_relation[rel])
        select_rng.shuffle(facts)
        selected_by_relation[rel] = sorted(facts[:quotas[rel]])

    # Step 2: deal the selected facts round-robin to the languages.
    deal_rng = random.Random(seed + DEAL_SEED_OFFSET)
    assignment = {lang: [] for lang in langs}
    counter = 0

    for rel in sorted(selected_by_relation):
        facts = list(selected_by_relation[rel])
        deal_rng.shuffle(facts)
        for fact in facts:
            assignment[langs[counter % len(langs)]].append(fact)
            counter += 1

    selected = {f for facts in selected_by_relation.values() for f in facts}
    return selected, {lang: sorted(facts) for lang, facts in assignment.items()}


def verify(parts, selected, assignment, langs, budget):
    per_lang = budget // len(langs)
    held_out = parts["val"] | parts["test"]

    assert len(selected) == budget, (len(selected), budget)
    assert selected <= parts["train"], "Selected facts outside the train split"

    seen = set()
    for lang in langs:
        facts = set(assignment[lang])

        if len(facts) != len(assignment[lang]):
            raise AssertionError(f"Duplicate facts within language {lang}")
        if len(facts) != per_lang:
            raise AssertionError(
                f"{lang} has {len(facts)} facts, expected exactly {per_lang}"
            )
        if facts & seen:
            raise AssertionError(f"{lang} shares facts with another language")
        if facts & held_out:
            raise AssertionError(
                f"{lang} has {len(facts & held_out)} facts that are also in "
                "the val/test split"
            )
        if not facts <= parts["train"]:
            raise AssertionError(f"{lang} has facts outside the train split")

        seen |= facts

    assert seen == selected, "Union of language subsets != selected facts"


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-budget, language-disjoint training partitions"
    )
    parser.add_argument("--langs", nargs="+", required=True,
                        help="Source languages, e.g. en nl ru el")
    parser.add_argument("--budget", type=int, default=8928,
                        help="Total number of unique training facts B "
                             "(default 8928 = 32 x 279, max for the keep split)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Partition seed (use the training seed)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size, used to check that batches "
                             "can be exactly balanced")
    parser.add_argument("--split_file", default=DEFAULT_SPLIT,
                        help="Shared train/val/test split (relative to project root)")
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: splits/budget/<langs>_B<B>_seed<seed>.json)")
    args = parser.parse_args()

    langs = list(args.langs)
    num_langs = len(langs)

    if len(set(langs)) != num_langs:
        parser.error(f"Duplicate languages in --langs: {langs}")
    if args.budget % num_langs:
        parser.error(f"--budget {args.budget} is not divisible by {num_langs} languages")
    if args.batch_size % num_langs:
        parser.error(
            f"--batch_size {args.batch_size} is not divisible by {num_langs} "
            "languages, so batches cannot be exactly balanced"
        )
    if args.budget % args.batch_size:
        print(
            f"WARNING: budget {args.budget} is not a multiple of batch size "
            f"{args.batch_size}; the last incomplete batch of every epoch is dropped.",
            file=sys.stderr,
        )

    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = ROOT / split_path

    manifest, parts = load_split(split_path)

    selected, assignment = make_partitions(
        parts["train"], langs, args.budget, args.seed
    )
    verify(parts, selected, assignment, langs, args.budget)

    relation_counts = {
        lang: dict(sorted(Counter(f[0] for f in facts).items()))
        for lang, facts in assignment.items()
    }
    split_relations = Counter(f[0] for f in parts["train"])
    covered = {f[0] for f in selected}

    tag = "-".join(langs)
    output = Path(args.output) if args.output else (
        ROOT / "splits" / "budget" / f"{tag}_B{args.budget}_seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "langs": langs,
            "tag": tag,
            "budget": args.budget,
            "facts_per_lang": args.budget // num_langs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "split_file": str(Path(args.split_file).as_posix()),
            "split_sha256": file_sha256(split_path),
            "split_train_size": len(parts["train"]),
            "split_val_size": len(parts["val"]),
            "split_test_size": len(parts["test"]),
            "num_relations_in_train": len(split_relations),
            "num_relations_selected": len(covered),
            "checks": {
                "languages_disjoint": True,
                "subset_of_train": True,
                "no_overlap_with_val_or_test": True,
                "exact_facts_per_lang": True,
            },
        },
        "relation_counts": relation_counts,
        "train": {lang: [list(f) for f in facts] for lang, facts in assignment.items()},
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print(f"Split:       {args.split_file} (train {len(parts['train'])}, "
          f"val {len(parts['val'])}, test {len(parts['test'])})")
    print(f"Languages:   {langs}")
    print(f"Budget:      {args.budget} facts ({args.budget // num_langs} per language)")
    print(f"Relations:   {len(covered)}/{len(split_relations)} covered")
    print("Checks:      disjoint languages, train-only, no val/test overlap -> OK")
    print(f"Saved:       {output}")


if __name__ == "__main__":
    main()
