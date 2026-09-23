"""
Dataset class for Fixed-LM Prompt Tuning on the X-FACTR benchmark.

Converts X-FACTR factual triples into (input_with_mask, gold_label) pairs
suitable for training soft prompts with a frozen mBERT model.

Example:
    Triple:  (Q302, P19, Q430776)  =  (Jesus, place_of_birth, Nazareth)
    Template: "[X] was born in [Y] ."
    Input:    "Jesus was born in [MASK] ."
    Gold:     "Nazareth"  → tokenized to token IDs
"""

import sys
import os
import json
import csv
from os.path import dirname, abspath, join, exists
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Path resolution — point to the project root (Multilingual_NLP_Project/)
# so we can reuse the existing data directory structure.
# ---------------------------------------------------------------------------
PIPELINE_DIR = dirname(abspath(__file__))
ROOT = dirname(PIPELINE_DIR)  # Multilingual_NLP_Project/

# Add scripts/ to path so we can import the Prompt helper (optional)
sys.path.insert(0, join(ROOT, 'scripts'))

try:
    from prompt import Prompt
    HAS_INFLECTION = True
except ImportError:
    HAS_INFLECTION = False


# ---------------------------------------------------------------------------
# Lightweight helpers (mirrors from Project_fixedLM/load_data.py)
# ---------------------------------------------------------------------------

class Gender:
    NONE = 'none'
    MALE = 'male'
    FEMALE = 'female'

    @staticmethod
    def parse(text: str):
        if text.lower() == Gender.MALE:
            return Gender.MALE
        if text.lower() == Gender.FEMALE:
            return Gender.FEMALE
        return Gender.NONE


class SimplePrompt:
    """Plain [X]/[Y] placeholder filler, used when scripts/prompt.py's
    dependencies aren't available."""

    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        return prompt.replace('[X]', label), label

    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int = 1, mask_sym: str = '[MASK]') -> Tuple[str, str]:
        target = label if num_mask <= 0 else ' '.join([mask_sym] * num_mask)
        return prompt.replace('[Y]', target), label


def _load_entity_lang(filename: str) -> Dict[str, Dict[str, str]]:
    entity2lang: Dict[str, Dict[str, str]] = defaultdict(dict)
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split('\t')
            entity = parts[0]
            for chunk in parts[1:]:
                label, lang = chunk.rsplit('@', 1)
                entity2lang[entity][lang] = label.strip('"')
    return entity2lang


def _load_entity_gender(filename: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            uri, gender = line.strip().split('\t')
            result[uri] = Gender.parse(gender)
    return result


def _load_entity_instance(filename: str) -> Dict[str, str]:
    entity2instance: Dict[str, str] = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split('\t')
            entity2instance[parts[0]] = ','.join(parts[1:])
    return entity2instance


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET = {
    'mlama': dict(
        entity_path=join(ROOT, 'data', 'mTREx', 'sub', '{}.jsonl'),
        entity_lang_path=join(ROOT, 'data', 'mTREx_unicode_escape.txt'),
        entity_gender_path=join(ROOT, 'data', 'mTREx_gender.txt'),
        entity_instance_path=join(ROOT, 'data', 'mTREx_instanceof.txt'),
    ),
    'mlamaf': dict(
        entity_path=join(ROOT, 'data', 'mTRExf', 'sub', '{}.jsonl'),
        entity_lang_path=join(ROOT, 'data', 'mTRExf_unicode_escape.txt'),
        entity_gender_path=join(ROOT, 'data', 'mTRExf_gender.txt'),
        entity_instance_path=join(ROOT, 'data', 'mTRExf_instanceof.txt'),
    ),
    'lama': dict(
        entity_path=join(ROOT, 'data', 'TREx', '{}.jsonl'),
        entity_lang_path=join(ROOT, 'data', 'TREx_unicode_escape.txt'),
        entity_gender_path=join(ROOT, 'data', 'TREx_gender.txt'),
        entity_instance_path=join(ROOT, 'data', 'TREx_instanceof.txt'),
    ),
}
RELATION_PATH = join(ROOT, 'data', 'TREx-relations.jsonl')
PROMPT_LANG_PATH = join(ROOT, 'data', 'TREx_prompts.csv')


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_relations() -> Dict[str, Dict]:
    """Load relation metadata from TREx-relations.jsonl."""
    relations = {}
    with open(RELATION_PATH, 'r', encoding='utf-8') as fin:
        for line in fin:
            r = json.loads(line)
            relations[r['relation']] = r
    return relations


def load_relation_templates(lang: str) -> Dict[str, str]:
    """Load per-relation prompt templates for a given language from TREx_prompts.csv."""
    with open(PROMPT_LANG_PATH, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        if lang not in (reader.fieldnames or []):
            available = [f for f in reader.fieldnames if f not in ('pid', 'relation')]
            raise ValueError(
                f'Language "{lang}" has no prompt templates. '
                f'Available languages: {", ".join(sorted(available))}'
            )
        templates = {}
        for row in reader:
            if row[lang]:
                templates[row['pid']] = row[lang]
    return templates


def load_examples(
    lang: str,
    tokenizer,
    probe: str = 'mlamaf',
    portion: str = 'trans',
    pids: Optional[List[str]] = None,
    num_mask: int = 1,
    mask_token: str = '[MASK]',
    use_inflection: bool = True,
    limit: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Load factual triple examples for prompt tuning.

    Returns:
        examples: list of dicts, each with keys
            {relation, template, sub_uri, sub_label, obj_uri, obj_label,
             query, answer, translated}
        stats: summary statistics dict
    """
    paths = DATASET[probe]
    relations = load_relations()
    templates = load_relation_templates(lang)
    entity2lang = _load_entity_lang(paths['entity_lang_path'])
    entity2gender = defaultdict(lambda: Gender.NONE, _load_entity_gender(paths['entity_gender_path']))
    entity2instance = defaultdict(str, _load_entity_instance(paths['entity_instance_path']))

    if use_inflection and HAS_INFLECTION:
        prompt_model = Prompt.from_lang(lang, entity2gender, entity2instance)
    else:
        prompt_model = SimplePrompt()

    pid_filter = set(pids) if pids else None
    examples: List[Dict] = []
    stats = {'num_relations': 0, 'num_facts': 0, 'num_skipped_portion': 0, 'num_errors': 0}

    for pid, template in templates.items():
        if pid_filter is not None and pid not in pid_filter:
            continue
        fact_path = paths['entity_path'].format(pid)
        if not exists(fact_path):
            continue
        stats['num_relations'] += 1

        with open(fact_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                fact = json.loads(line)
                sub_uri, obj_uri = fact['sub_uri'], fact['obj_uri']
                sub_exist = lang in entity2lang[sub_uri]
                obj_exist = lang in entity2lang[obj_uri]
                exist = sub_exist and obj_exist

                if portion == 'trans' and not exist:
                    stats['num_skipped_portion'] += 1
                    continue
                if portion == 'non' and exist:
                    stats['num_skipped_portion'] += 1
                    continue

                sub_label = (entity2lang[sub_uri].get(lang)
                             or entity2lang[sub_uri].get('en')
                             or fact['sub_label'])
                obj_label = (entity2lang[obj_uri].get(lang)
                             or entity2lang[obj_uri].get('en')
                             or fact['obj_label'])

                try:
                    filled_x, _ = prompt_model.fill_x(template, sub_uri, sub_label)

                    # Match the number of masks to the number of mBERT
                    # subword tokens in the gold object, capped by num_mask.
                    answer_token_ids = tokenizer.encode(
                        obj_label,
                        add_special_tokens=False,
                    )

                    if len(answer_token_ids) == 0:
                        stats['num_errors'] += 1
                        continue

                    example_num_masks = min(
                        len(answer_token_ids),
                        num_mask,
                    )

                    query, answer = prompt_model.fill_y(
                        filled_x, obj_uri, obj_label,
                        num_mask=example_num_masks,
                        mask_sym=mask_token,
                    )
                except Exception:
                    stats['num_errors'] += 1
                    continue

                examples.append({
                    'relation': pid,
                    'template': template,
                    'sub_uri': sub_uri,
                    'sub_label': sub_label,
                    'obj_uri': obj_uri,
                    'obj_label': obj_label,
                    'query': query,
                    'answer': answer,
                    'translated': exist,
                })
                stats['num_facts'] += 1

                if limit is not None and len(examples) >= limit:
                    return examples, stats

    return examples, stats


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PromptTuningDataset(Dataset):
    """
    PyTorch Dataset for Fixed-LM Prompt Tuning on X-FACTR.

    Each sample is a dict with:
        - input_ids:      tokenized query with [MASK] (LongTensor)
        - attention_mask:  1s for real tokens, 0s for padding (LongTensor)
        - mask_positions:  indices of [MASK] tokens in input_ids (LongTensor)
        - label_ids:       tokenized gold answer token IDs (LongTensor)
        - answer_text:     raw answer string (str)
        - query_text:      raw query string (str)
    """

    def __init__(
        self,
        tokenizer,
        lang: str = 'en',
        probe: str = 'mlamaf',
        portion: str = 'trans',
        pids: Optional[List[str]] = None,
        num_mask: int = 1,
        mask_token: str = '[MASK]',
        use_inflection: bool = True,
        max_seq_len: int = 128,
        limit: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer (e.g. BertTokenizer for mBERT).
            lang: Language code (e.g. 'en', 'fr', 'zh').
            probe: Dataset variant ('mlama', 'mlamaf', 'lama').
            portion: 'trans' (only translated facts), 'non', or 'all'.
            pids: Optional list of relation IDs to filter.
            num_mask: Number of [MASK] tokens per answer slot.
            mask_token: Token string used for masking (default '[MASK]').
            use_inflection: Whether to use language-aware inflection.
            max_seq_len: Maximum sequence length for tokenization.
            limit: Cap number of examples (for quick testing).
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load raw examples
        self.examples, self.stats = load_examples(
            lang=lang,
            tokenizer=tokenizer,
            probe=probe,
            portion=portion,
            pids=pids,
            num_mask=num_mask,
            mask_token=mask_token,
            use_inflection=use_inflection,
            limit=limit,
        )

        # Pre-tokenize all examples
        self.samples = []
        for ex in self.examples:
            sample = self._tokenize_example(ex)
            if sample is not None:
                self.samples.append(sample)

    def _tokenize_example(self, example: Dict) -> Optional[Dict]:
        """Tokenize a single example into model-ready tensors."""
        query = example['query']
        answer = example['answer']

        # Tokenize the query (which already contains [MASK])
        encoding = self.tokenizer(
            query,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze(0)       # (max_seq_len,)
        attention_mask = encoding['attention_mask'].squeeze(0)  # (max_seq_len,)

        # Find [MASK] positions
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False).squeeze(-1)

        if mask_positions.numel() == 0:
            return None  # skip examples where [MASK] was truncated away

        # Tokenize the gold answer
        answer_token_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        if len(answer_token_ids) == 0:
            return None

        # The query contains one mask per gold answer token, up to the
        # configured maximum number of masks. Therefore the number of
        # supervised answer tokens must equal the number of masks.
        num_masks = mask_positions.size(0)
        expected_num_masks = min(len(answer_token_ids), num_masks)

        if expected_num_masks != num_masks:
            raise ValueError(
                f'Mask/answer mismatch: {num_masks} masks for '
                f'{len(answer_token_ids)} answer tokens. '
                f'Query={query!r}, answer={answer!r}'
            )

        # Build label tensor: -100 everywhere except at mask positions.
        # Answers longer than the configured mask maximum are explicitly
        # supervised on the prefix represented by those masks.
        labels = torch.full_like(input_ids, -100)
        for i in range(num_masks):
            labels[mask_positions[i]] = answer_token_ids[i]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mask_positions': mask_positions,
            'labels': labels,
            'answer_text': answer,
            'query_text': query,

            # Stable X-FACTR fact identifiers.
            'relation': example['relation'],
            'sub_uri': example['sub_uri'],
            'obj_uri': example['obj_uri'],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that stacks tensors and keeps string fields as lists.
    """

    result = {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'answer_text': [b['answer_text'] for b in batch],
        'query_text': [b['query_text'] for b in batch],
        'relation': [b['relation'] for b in batch],
        'sub_uri': [b['sub_uri'] for b in batch],
        'obj_uri': [b['obj_uri'] for b in batch],

    }
    
    return result
