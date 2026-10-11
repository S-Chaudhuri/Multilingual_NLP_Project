"""
Multilingual data pipeline for the transfer-scaling experiments.

Kept separate from prompt_tuning_data.py so that the existing monolingual
pipeline and ablation runs are not affected. It wraps one PromptTuningDataset
per source language and adds:

    MultilingualFactDataset       training examples from several languages,
                                  restricted to each language's assigned facts
                                  (a budget file from make_budget_partitions.py)
    BalancedLanguageBatchSampler  every batch holds exactly batch_size / L
                                  examples from each of the L languages
    multilingual_collate_fn       collate_fn plus the language of each example

Facts are identified by (relation, sub_uri, obj_uri).
"""

import json
import logging
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset, Sampler, Subset

from prompt_tuning_data import PromptTuningDataset, collate_fn


logger = logging.getLogger("multilingual_data")

Fact = Tuple[str, str, str]


def fact_key(sample: Dict) -> Fact:
    return (sample["relation"], sample["sub_uri"], sample["obj_uri"])


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def load_split(path: str) -> Dict[str, Set[Fact]]:
    """Load a shared train/val/test split as sets of fact tuples."""
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    parts = {
        name: {tuple(fact) for fact in manifest[name]}
        for name in ("train", "val", "test")
    }

    if parts["train"] & parts["val"] or parts["train"] & parts["test"] \
            or parts["val"] & parts["test"]:
        raise ValueError(f"Split {path} has overlapping train/val/test parts")

    return parts


def load_budget(path: str) -> Dict:
    """Load a budget file written by make_budget_partitions.py."""
    with open(path, "r", encoding="utf-8") as f:
        budget = json.load(f)

    budget["train"] = {
        lang: [tuple(fact) for fact in facts]
        for lang, facts in budget["train"].items()
    }
    return budget


def check_budget_against_split(budget: Dict, split: Dict[str, Set[Fact]]) -> None:
    """Re-verify, at training time, that no budget fact is held out."""
    held_out = split["val"] | split["test"]
    seen: Set[Fact] = set()

    for lang, facts in budget["train"].items():
        facts = set(facts)
        if facts & held_out:
            raise ValueError(
                f"Budget facts for {lang} overlap the val/test split "
                f"({len(facts & held_out)} facts)"
            )
        if not facts <= split["train"]:
            raise ValueError(f"Budget facts for {lang} are not all in the train split")
        if facts & seen:
            raise ValueError(f"Budget facts for {lang} overlap another language")
        seen |= facts


# ---------------------------------------------------------------------------
# Language datasets
# ---------------------------------------------------------------------------

def load_language_datasets(
    tokenizer,
    langs: Sequence[str],
    probe: str = "mlamaf",
    portion: str = "trans",
    num_mask: int = 10,
    use_inflection: bool = True,
    max_seq_len: int = 128,
) -> Dict[str, PromptTuningDataset]:
    """One complete PromptTuningDataset per language."""
    datasets = {}
    for lang in langs:
        logger.info(f"Loading language dataset: {lang}")
        datasets[lang] = PromptTuningDataset(
            tokenizer=tokenizer,
            lang=lang,
            probe=probe,
            portion=portion,
            pids=None,
            num_mask=num_mask,
            use_inflection=use_inflection,
            max_seq_len=max_seq_len,
            limit=None,
        )
        logger.info(f"  {lang}: {len(datasets[lang])} samples (stats: {datasets[lang].stats})")
    return datasets


def select_indices(
    dataset: PromptTuningDataset,
    facts: Set[Fact],
    lang: str,
    what: str,
    allow_incomplete: bool = False,
) -> List[int]:
    """
    Indices of `dataset` whose fact is in `facts`.

    Raises if some facts cannot be found (e.g. the language's tokenizer drops
    them), unless allow_incomplete is set.
    """
    indices = []
    found = set()

    for idx, sample in enumerate(dataset.samples):
        key = fact_key(sample)
        if key in facts and key not in found:
            indices.append(idx)
            found.add(key)

    missing = facts - found
    if missing:
        message = (
            f"{len(missing)} of {len(facts)} {what} facts were not found for "
            f"language {lang}; first missing: {sorted(missing)[:5]}"
        )
        if not allow_incomplete:
            raise ValueError(message + " (use --allow_incomplete_split to continue)")
        logger.warning(message)

    return indices


def split_subset(
    dataset: PromptTuningDataset,
    facts: Set[Fact],
    lang: str,
    what: str,
    allow_incomplete: bool = False,
    limit: Optional[int] = None,
) -> Subset:
    """Subset of one language dataset restricted to `facts` (e.g. the val split)."""
    indices = select_indices(dataset, facts, lang, what, allow_incomplete)
    if limit is not None:
        indices = indices[:limit]
    return Subset(dataset, indices)


# ---------------------------------------------------------------------------
# Multilingual training dataset
# ---------------------------------------------------------------------------

class MultilingualFactDataset(Dataset):
    """
    Training examples from several languages. Language `lang` contributes only
    the facts assigned to it in the budget file.

    Every item is the language dataset's sample dict plus a "lang" key.
    """

    def __init__(
        self,
        datasets: Dict[str, PromptTuningDataset],
        assignment: Dict[str, Sequence[Fact]],
        allow_incomplete: bool = False,
        max_facts_per_lang: Optional[int] = None,
    ):
        self.langs = list(assignment)
        self.items: List[Tuple[str, int]] = []           # (lang, index in language dataset)
        self.indices_by_lang: Dict[str, List[int]] = {}  # lang -> positions in self.items
        self.datasets = datasets

        for lang in self.langs:
            facts = list(assignment[lang])
            if max_facts_per_lang is not None:
                facts = facts[:max_facts_per_lang]

            local = select_indices(
                datasets[lang], set(facts), lang, "training", allow_incomplete
            )

            start = len(self.items)
            self.items.extend((lang, idx) for idx in local)
            self.indices_by_lang[lang] = list(range(start, len(self.items)))

        sizes = {lang: len(idx) for lang, idx in self.indices_by_lang.items()}
        logger.info(f"Training examples per language: {sizes}")

        if len(set(sizes.values())) > 1:
            logger.warning(
                "Languages have different numbers of training examples; the "
                "balanced sampler uses only the first min(size) examples per "
                f"language per epoch: {sizes}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        lang, local = self.items[idx]
        sample = dict(self.datasets[lang][local])
        sample["lang"] = lang
        return sample

    def facts_by_lang(self) -> Dict[str, Set[Fact]]:
        result = {lang: set() for lang in self.langs}
        for lang, local in self.items:
            result[lang].add(fact_key(self.datasets[lang].samples[local]))
        return result


class BalancedLanguageBatchSampler(Sampler):
    """
    Yields batches with exactly batch_size / L examples from every language.

    Each epoch, every language's examples are shuffled independently with a
    generator seeded by (seed, epoch); call set_epoch() before each epoch.
    With equally sized languages every example is used once per epoch, and
    every configuration with the same budget gets the same number of batches.
    """

    def __init__(
        self,
        indices_by_lang: Dict[str, List[int]],
        batch_size: int,
        seed: int,
    ):
        self.langs = list(indices_by_lang)
        self.indices_by_lang = indices_by_lang
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        num_langs = len(self.langs)
        if batch_size % num_langs:
            raise ValueError(
                f"batch_size {batch_size} is not divisible by {num_langs} languages"
            )

        self.per_lang = batch_size // num_langs
        smallest = min(len(idx) for idx in indices_by_lang.values())
        self.num_batches = smallest // self.per_lang

        if self.num_batches == 0:
            raise ValueError(
                f"Not enough examples for one balanced batch: smallest language "
                f"has {smallest}, need {self.per_lang}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1000 + self.epoch)

        shuffled = {}
        for lang in self.langs:
            indices = self.indices_by_lang[lang]
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled[lang] = [indices[i] for i in order]

        for b in range(self.num_batches):
            batch = []
            for lang in self.langs:
                start = b * self.per_lang
                batch.extend(shuffled[lang][start:start + self.per_lang])
            yield batch


def multilingual_collate_fn(batch: List[Dict]) -> Dict:
    """collate_fn from prompt_tuning_data plus a list of languages."""
    result = collate_fn(batch)
    result["lang"] = [b.get("lang") for b in batch]
    return result
