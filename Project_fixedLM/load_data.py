"""
Usage:
    python scripts/load_lang_data.py --lang fr --out data/query_fact/fr.csv
    python scripts/load_lang_data.py --lang ja --pids P19,P20 --out 

    Language codes come from data/TREx_prompts.csv's header:
    en, zh, fr, el, nl, ru, ko, es, mr, vi, he, tr, yo, ja, hu, bn,
    war, tl, sw, mg, pa, ilo, ceb

    Flags:
    --lang         language code (required, see list above)
    --probe        dataset to load facts from: mlama (default), mlamaf, lama
    --portion      trans (default): only facts translated into --lang
                   non: only facts NOT translated into --lang
                   all: every fact, falling back to the English label
    --pids         comma-separated relation ids to include, e.g. P19,P20
                   (names are in data/TREx-relations.jsonl; default: all)
    --num_mask     number of mask tokens to substitute for the answer (default: 1)
    --mask_token   token used to mask the answer in the query (default: [MASK])
    --no_inflect   disable language-aware inflection (case/gender/articles),
                   use plain [X]/[Y] substitution even if available
    --limit        cap the number of examples, for quick testing
    --format       csv (default, one readable file per language) or jsonl
    --out          output path (default: data/query_fact/<lang>.<format>);
                   pass - to print to the terminal instead of writing a file
"""
import sys
import os
import csv
import json
import argparse
from os.path import dirname, abspath, join, exists
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(ROOT, 'scripts'))

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


def load_entity_gender(filename: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for l in fin:
            uri, gender = l.strip().split('\t')
            result[uri] = Gender.parse(gender)
    return result


def load_entity_instance(filename: str) -> Dict[str, str]:
    entity2instance: Dict[str, str] = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for l in fin:
            parts = l.strip().split('\t')
            entity2instance[parts[0]] = ','.join(parts[1:])
    return entity2instance

try:
    from prompt import Prompt
    HAS_INFLECTION = True
except ImportError:
    HAS_INFLECTION = False

#prompt filling
class SimplePrompt:
    """Plain [X]/[Y] placeholder filler, used when scripts/prompt.py's
    dependencies aren't available."""

    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        return prompt.replace('[X]', label), label

    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int = 1, mask_sym: str = '[MASK]') -> Tuple[str, str]:
        target = label if num_mask <= 0 else ' '.join([mask_sym] * num_mask)
        return prompt.replace('[Y]', target), label


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


def load_entity_lang(filename: str) -> Dict[str, Dict[str, str]]:
    entity2lang: Dict[str, Dict[str, str]] = defaultdict(dict)
    with open(filename, 'r', encoding='utf-8') as fin:
        for l in fin:
            parts = l.strip().split('\t')
            entity = parts[0]
            for chunk in parts[1:]:
                label, lang = chunk.rsplit('@', 1)
                entity2lang[entity][lang] = label.strip('"')
    return entity2lang


def load_relations() -> Dict[str, Dict]:
    relations = {}
    with open(RELATION_PATH, 'r', encoding='utf-8') as fin:
        for l in fin:
            r = json.loads(l)
            relations[r['relation']] = r
    return relations


def load_relation_templates(lang: str) -> Dict[str, str]:
    with open(PROMPT_LANG_PATH, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        if lang not in (reader.fieldnames or []):
            available = [f for f in reader.fieldnames if f not in ('pid', 'relation')]
            raise ValueError(f'language "{lang}" has no prompt templates. '
                              f'available languages: {", ".join(sorted(available))}')
        templates = {}
        for row in reader:
            if row[lang]:
                templates[row['pid']] = row[lang]
    return templates


def load_language_data(lang: str,
                        probe: str = 'mlama',
                        portion: str = 'trans',
                        pids: Optional[List[str]] = None,
                        num_mask: int = 1,
                        mask_token: str = '[MASK]',
                        use_inflection: bool = True,
                        limit: Optional[int] = None) -> Tuple[List[Dict], Dict]:

    paths = DATASET[probe]
    relations = load_relations()
    templates = load_relation_templates(lang)
    entity2lang = load_entity_lang(paths['entity_lang_path'])
    entity2gender = defaultdict(lambda: Gender.NONE, load_entity_gender(paths['entity_gender_path']))
    entity2instance = defaultdict(str, load_entity_instance(paths['entity_instance_path']))

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
        relation = relations.get(pid, {})

        with open(fact_path, 'r', encoding='utf-8') as fin:
            for l in fin:
                fact = json.loads(l)
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

                sub_label = entity2lang[sub_uri].get(lang) or entity2lang[sub_uri].get('en') or fact['sub_label']
                obj_label = entity2lang[obj_uri].get(lang) or entity2lang[obj_uri].get('en') or fact['obj_label']

                try:
                    instance_x, _ = prompt_model.fill_x(template, sub_uri, sub_label)
                    query, answer = prompt_model.fill_y(
                        instance_x, obj_uri, obj_label, num_mask=num_mask, mask_sym=mask_token)
                except Exception as e:
                    stats['num_errors'] += 1
                    continue

                examples.append({
                    'relation': pid,
                    'relation_label': relation.get('label'),
                    'template': template,
                    'uuid': fact.get('uuid'),
                    'sub_uri': sub_uri,
                    'sub_label': sub_label,
                    'obj_uri': obj_uri,
                    'obj_label': obj_label,
                    'translated': exist,
                    'query': query,
                    'answer': answer,
                })
                stats['num_facts'] += 1

                if limit is not None and len(examples) >= limit:
                    return examples, stats

    return examples, stats


CSV_FIELDS = [
    'relation', 'relation_label',
    'sub_uri', 'sub_label',
    'obj_uri', 'obj_label',
    'query', 'answer',
    'translated',
]


def write_examples(examples: List[Dict], out_path: str, fmt: str) -> None:
    """Write examples as one readable CSV (default) or as JSONL. out_path
    of '-' streams to stdout instead of writing a file."""
    to_stdout = out_path == '-'
    if not to_stdout:
        out_dir = dirname(abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    if fmt == 'csv':
        fout = sys.stdout if to_stdout else open(out_path, 'w', encoding='utf-8', newline='')
        try:
            writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for ex in examples:
                writer.writerow({k: ex[k] for k in CSV_FIELDS})
        finally:
            if not to_stdout:
                fout.close()
    else:
        if to_stdout:
            for ex in examples:
                print(json.dumps(ex, ensure_ascii=False))
        else:
            with open(out_path, 'w', encoding='utf-8') as fout:
                for ex in examples:
                    fout.write(json.dumps(ex, ensure_ascii=False) + '\n')


def main():
    # avoid UnicodeEncodeError when printing non-Latin scripts on consoles
    # whose default codepage isn't UTF-8 (e.g. Windows cp1252)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description='load mLAMA query/fact data for a language')
    parser.add_argument('--lang', type=str, required=True, help='language code, e.g. en, fr, zh, ja')
    parser.add_argument('--probe', type=str, choices=list(DATASET.keys()), default='mlama',
                         help='which dataset to load facts from')
    parser.add_argument('--portion', type=str, choices=['all', 'trans', 'non'], default='trans',
                         help='trans: only facts translated into --lang; non: only untranslated; '
                              'all: everything (falls back to English label)')
    parser.add_argument('--pids', type=str, default=None, help='comma-separated relation ids to include')
    parser.add_argument('--num_mask', type=int, default=1, help='number of mask tokens to substitute for the answer')
    parser.add_argument('--mask_token', type=str, default='[MASK]', help='token used to mask the answer in the query')
    parser.add_argument('--no_inflect', action='store_true',
                         help='disable language-aware inflection even if available, use plain [X]/[Y] substitution')
    parser.add_argument('--limit', type=int, default=None, help='cap the number of examples (for quick testing)')
    parser.add_argument('--format', type=str, choices=['csv', 'jsonl'], default='csv',
                         help='output format: one readable csv per language (default), or jsonl')
    parser.add_argument('--out', type=str, default=None,
                         help='output path (default: data/query_fact/<lang>.<format>); use - for stdout')
    args = parser.parse_args()

    examples, stats = load_language_data(
        lang=args.lang,
        probe=args.probe,
        portion=args.portion,
        pids=args.pids.split(',') if args.pids else None,
        num_mask=args.num_mask,
        mask_token=args.mask_token,
        use_inflection=not args.no_inflect,
        limit=args.limit,
    )

    out_path = args.out
    if out_path is None:
        out_dir = join(ROOT, 'data', 'query_fact')
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{args.lang}.{args.format}')

    write_examples(examples, out_path, args.format)
    if out_path != '-':
        print(f'wrote {len(examples)} examples to {out_path}')

    print(f'#relations {stats["num_relations"]}\t#facts {stats["num_facts"]}\t'
          f'#skipped(portion) {stats["num_skipped_portion"]}\t#errors {stats["num_errors"]}\t'
          f'inflection {"on" if (not args.no_inflect and HAS_INFLECTION) else "off (plain substitution)"}',
          file=sys.stderr)


if __name__ == '__main__':
    main()