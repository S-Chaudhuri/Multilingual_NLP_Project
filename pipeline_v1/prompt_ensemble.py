"""
Soft-prompt ensembling with length-normalised candidate scoring.

Every "member" (a trained soft prompt, optionally plus the zero-shot model with
no prompt) scores every candidate object of a relation. A candidate c with
tokens t_1..t_L is scored with L [MASK] tokens in the query:

    logp(c) = sum_i log p(t_i | query with L masks)

Raw sums decay exponentially with L, so multi-token candidates are almost
never ranked first. Scores are therefore length-normalised:

    score_alpha(c) = logp(c) / L ** alpha

alpha = 0 is the raw sum, alpha = 1 is the per-token mean. alpha is selected
on the validation split from --alphas.

Members are combined with learnable scalar weights w = softmax(theta):

    ensemble(c) = tau * sum_m w_m * score_alpha_m(c)

theta and the temperature tau are fit on the validation split by minimising
cross-entropy over each relation's candidate set (tau does not change the
argmax; it only calibrates the loss). The learned ensemble is compared with
every single member and with a uniform-weight ensemble on the test split.

Candidates for a relation are all distinct gold objects of that relation in
the language's dataset. A prediction is an Exact Match (EM) when its token IDs
equal the gold answer's. EM is reported micro- and macro-averaged (over
relations) for all / single-token / multi-token gold answers.

Example:
    python pipeline_v1/prompt_ensemble.py \\
        --lang en \\
        --model_name bert-base-multilingual-cased \\
        --soft_prompts ckpt/P5/soft_prompt_en_P5_best.pt \\
                       ckpt/P10/soft_prompt_en_P10_best.pt \\
                       ckpt/P20/soft_prompt_en_P20_best.pt \\
        --names P5 P10 P20 \\
        --include_zero_shot \\
        --split_file splits/shared_en_nl_tr_el_sw_seed42.json \\
        --output results/ablation/mbert_base/en/ensemble.json
"""

import sys
import os
import re
import json
import random
import argparse
import logging
from collections import defaultdict
from os.path import dirname, abspath
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


PIPELINE_DIR = dirname(abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from soft_prompt import SoftPromptEmbedding
from prompt_tuning_data import PromptTuningDataset
from prompt_tuning_train import masked_lm_logits, freeze_model
from prompt_tuning_eval import apply_shared_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("prompt_ensemble")


# ---------------------------------------------------------------------------
# Candidates and examples
# ---------------------------------------------------------------------------

def build_candidates(dataset, tokenizer, max_len: int) -> Dict[str, Dict]:
    """
    Collect each relation's candidate objects from the full language dataset.

    Candidates are deduplicated by token IDs; candidates longer than max_len
    tokens are dropped because they cannot be scored with max_len masks.

    Returns:
        relation -> {
            "texts":  [str],                 # one surface form per candidate
            "ids":    [tuple[int]],          # token IDs per candidate
            "index":  {tuple[int]: int},     # token IDs -> candidate position
            "lengths": LongTensor (C,),
        }
    """

    candidates: Dict[str, Dict] = {}

    for sample in dataset.samples:
        rel = sample["relation"]
        entry = candidates.setdefault(
            rel, {"texts": [], "ids": [], "index": {}}
        )

        ids = tuple(
            tokenizer.encode(sample["answer_text"], add_special_tokens=False)
        )

        if not 0 < len(ids) <= max_len or ids in entry["index"]:
            continue

        entry["index"][ids] = len(entry["ids"])
        entry["ids"].append(ids)
        entry["texts"].append(sample["answer_text"])

    for entry in candidates.values():
        entry["lengths"] = torch.tensor([len(ids) for ids in entry["ids"]])

    return candidates


def group_examples(subset, candidates, tokenizer) -> Dict[str, Dict]:
    """
    Group a split's examples by relation and attach the gold candidate index.

    Examples whose gold answer is not a candidate (longer than the mask limit)
    are skipped and counted.

    Returns:
        relation -> {"queries": [str], "gold": LongTensor (N,),
                     "gold_len": LongTensor (N,)}
        number of skipped examples
    """

    grouped = defaultdict(lambda: {"queries": [], "gold": [], "gold_len": []})
    skipped = 0

    for idx in subset.indices:
        sample = subset.dataset.samples[idx]
        rel = sample["relation"]

        ids = tuple(
            tokenizer.encode(sample["answer_text"], add_special_tokens=False)
        )

        if ids not in candidates[rel]["index"]:
            skipped += 1
            continue

        grouped[rel]["queries"].append(sample["query_text"])
        grouped[rel]["gold"].append(candidates[rel]["index"][ids])
        grouped[rel]["gold_len"].append(len(ids))

    return {
        rel: {
            "queries": g["queries"],
            "gold": torch.tensor(g["gold"]),
            "gold_len": torch.tensor(g["gold_len"]),
        }
        for rel, g in sorted(grouped.items())
    }, skipped


def with_n_masks(query: str, mask_token: str, n: int) -> str:
    """Replace the run of mask tokens in a query with exactly n masks."""

    mask = re.escape(mask_token)
    pattern = mask + r"(?:\s*" + mask + r")*"

    new_query, count = re.subn(
        pattern,
        lambda _: " ".join([mask_token] * n),
        query,
        count=1,
    )

    if count != 1:
        raise ValueError(f"No mask tokens found in query {query!r}")

    return new_query


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_member(
    model,
    soft_prompt: Optional[SoftPromptEmbedding],
    examples: Dict[str, Dict],
    candidates: Dict[str, Dict],
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Summed log-probability of every candidate for every example.

    One forward pass per (example, candidate length L) covers all candidates
    of that length, so the cost is independent of the number of candidates.

    Returns:
        relation -> FloatTensor (N_r, C_r)
    """

    model.eval()
    if soft_prompt is not None:
        soft_prompt.eval()

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id

    scores: Dict[str, torch.Tensor] = {}

    for rel, group in examples.items():
        cand = candidates[rel]
        rel_scores = torch.empty(len(group["queries"]), len(cand["ids"]))

        for length in sorted(set(cand["lengths"].tolist())):
            # Candidates of this length: (C_L,) positions and (C_L, L) token IDs.
            cand_pos = (cand["lengths"] == length).nonzero(as_tuple=True)[0]
            cand_ids = torch.tensor(
                [cand["ids"][c] for c in cand_pos.tolist()], device=device
            )
            steps = torch.arange(length, device=device).unsqueeze(0)

            queries = [
                with_n_masks(q, mask_token, length) for q in group["queries"]
            ]

            for start in range(0, len(queries), batch_size):
                enc = tokenizer(
                    queries[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                is_mask = input_ids == mask_id
                if not (is_mask.sum(-1) == length).all():
                    raise ValueError(
                        f"Masks were truncated for relation {rel} with "
                        f"{length} masks; increase --max_seq_len."
                    )

                # SHAPE: (B, L, V) — masks in left-to-right order per example.
                mask_logprobs = masked_lm_logits(
                    model, soft_prompt, input_ids, attention_mask, is_mask
                ).view(input_ids.size(0), length, -1).float().log_softmax(-1)

                # SHAPE: (B, C_L) = sum_i log p(token_i of candidate at mask_i)
                batch_scores = mask_logprobs[:, steps, cand_ids].sum(-1)

                rel_scores[start:start + batch_size, cand_pos] = batch_scores.cpu()

        scores[rel] = rel_scores

    return scores


def length_normalise(
    sum_logprob: torch.Tensor,
    lengths: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """score = sum log p / L ** alpha (alpha=0: raw sum, alpha=1: mean)."""

    return sum_logprob / lengths.float().pow(alpha)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def em_metrics(
    predictions: Dict[str, torch.Tensor],
    examples: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Exact Match for all / single-token / multi-token gold answers.

    macro: mean over relations that have examples in the subset.
    micro: mean over examples.
    """

    subsets = {
        "all": lambda n: torch.ones_like(n, dtype=torch.bool),
        "single": lambda n: n == 1,
        "multi": lambda n: n > 1,
    }

    metrics = {}

    for name, select in subsets.items():
        per_relation = []
        correct = total = 0

        for rel, group in examples.items():
            keep = select(group["gold_len"])
            if keep.sum() == 0:
                continue

            hits = (predictions[rel][keep] == group["gold"][keep]).float()
            per_relation.append(hits.mean().item())
            correct += hits.sum().item()
            total += int(keep.sum())

        metrics[name] = {
            "macro": sum(per_relation) / len(per_relation) if per_relation else 0.0,
            "micro": correct / total if total else 0.0,
            "n": total,
            "num_relations": len(per_relation),
        }

    return metrics


# ---------------------------------------------------------------------------
# Learnable scalar-weight ensemble
# ---------------------------------------------------------------------------

class ScalarWeightEnsemble(nn.Module):
    """ensemble(c) = tau * sum_m softmax(theta)_m * score_m(c)"""

    def __init__(self, num_members: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(num_members))
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def weights(self) -> torch.Tensor:
        return self.theta.softmax(-1)

    def forward(self, member_scores: torch.Tensor) -> torch.Tensor:
        # SHAPE: (M, N, C) -> (N, C)
        combined = torch.einsum("m,mnc->nc", self.weights, member_scores)
        return combined * self.log_temperature.exp()


def fit_ensemble(
    member_scores: Dict[str, torch.Tensor],
    examples: Dict[str, Dict],
    steps: int,
    lr: float,
) -> ScalarWeightEnsemble:
    """Fit the scalar weights by candidate cross-entropy on one split."""

    num_members = next(iter(member_scores.values())).size(0)
    ensemble = ScalarWeightEnsemble(num_members)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    total = sum(len(g["gold"]) for g in examples.values())

    for step in range(steps):
        optimizer.zero_grad()

        loss = sum(
            F.cross_entropy(
                ensemble(member_scores[rel]),
                examples[rel]["gold"],
                reduction="sum",
            )
            for rel in examples
        ) / total

        loss.backward()
        optimizer.step()

        if (step + 1) % max(steps // 5, 1) == 0:
            logger.info(
                f"  step {step + 1}/{steps} | val CE {loss.item():.4f} | "
                f"weights {[round(w, 3) for w in ensemble.weights.tolist()]}"
            )

    return ensemble


def stack_members(
    raw_scores: List[Dict[str, torch.Tensor]],
    candidates: Dict[str, Dict],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """relation -> (M, N, C) length-normalised member scores."""

    return {
        rel: torch.stack([
            length_normalise(member[rel], candidates[rel]["lengths"], alpha)
            for member in raw_scores
        ])
        for rel in raw_scores[0]
    }


def argmax_predictions(scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {rel: s.argmax(-1) for rel, s in scores.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_metrics(m: Dict) -> str:
    return (
        f"EM macro all {m['all']['macro']:.4f} | "
        f"single {m['single']['macro']:.4f} | "
        f"multi {m['multi']['macro']:.4f} "
        f"(micro all {m['all']['micro']:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Length-normalised soft-prompt ensembling on X-FACTR"
    )

    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="bert-base-multilingual-cased")
    parser.add_argument("--soft_prompts", type=str, nargs="+", default=[],
                        help="Soft prompt checkpoints (.pt); P is read from each file")
    parser.add_argument("--names", type=str, nargs="+", default=None,
                        help="Display names for --soft_prompts (default: P<k>)")
    parser.add_argument("--include_zero_shot", action="store_true",
                        help="Add the frozen model without a soft prompt as a member")
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--fit_split", type=str, default="val",
                        choices=["train", "val"],
                        help="Split used to fit weights and select alpha")
    parser.add_argument("--fit_limit", type=int, default=None,
                        help="Fit on a random subset of this many examples")
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--alphas", type=str, default="0,0.25,0.5,0.75,1",
                        help="Comma-separated length-normalisation exponents")
    parser.add_argument("--weight_steps", type=int, default=300)
    parser.add_argument("--weight_lr", type=float, default=0.05)
    parser.add_argument("--probe", type=str, default="mlamaf",
                        choices=["mlama", "mlamaf", "lama"])
    parser.add_argument("--portion", type=str, default="trans",
                        choices=["trans", "non", "all"])
    parser.add_argument("--num_mask", type=int, default=10,
                        help="Maximum candidate length in tokens")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--allow_incomplete_split", action="store_true")
    parser.add_argument("--no_inflect", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not args.soft_prompts and not args.include_zero_shot:
        parser.error("Nothing to evaluate: pass --soft_prompts and/or --include_zero_shot.")

    if args.names is not None and len(args.names) != len(args.soft_prompts):
        parser.error("--names must have one entry per --soft_prompts path.")

    alphas = [float(a) for a in args.alphas.split(",")]

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Model and members
    # -----------------------------------------------------------------------

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    freeze_model(model)

    members = []  # (name, SoftPromptEmbedding or None, path)

    if args.include_zero_shot:
        members.append(("zero_shot", None, None))

    for i, path in enumerate(args.soft_prompts):
        weights = torch.load(path, map_location="cpu")
        num_prompt_tokens, embedding_dim = weights.shape

        if embedding_dim != model.config.hidden_size:
            raise ValueError(
                f"{path} has embedding size {embedding_dim}, but "
                f"{args.model_name} has hidden size {model.config.hidden_size}."
            )

        soft_prompt = SoftPromptEmbedding.load(
            path=path,
            num_prompt_tokens=num_prompt_tokens,
            embedding_dim=embedding_dim,
        ).to(device)

        name = args.names[i] if args.names else f"P{num_prompt_tokens}"
        members.append((name, soft_prompt, path))
        logger.info(f"Member {name}: P={num_prompt_tokens} from {path}")

    member_names = [name for name, _, _ in members]
    if len(set(member_names)) != len(member_names):
        raise ValueError(f"Member names must be unique: {member_names}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    dataset = PromptTuningDataset(
        tokenizer=tokenizer,
        lang=args.lang,
        probe=args.probe,
        portion=args.portion,
        num_mask=args.num_mask,
        use_inflection=not args.no_inflect,
        max_seq_len=args.max_seq_len,
    )
    logger.info(f"Loaded language dataset: {len(dataset)} samples")

    candidates = build_candidates(dataset, tokenizer, args.num_mask)
    logger.info(
        f"Candidates: {sum(len(c['ids']) for c in candidates.values())} "
        f"across {len(candidates)} relations"
    )

    split_meta = {}
    examples = {}

    for role, split in (("fit", args.fit_split), ("eval", args.eval_split)):
        subset, split_meta[role] = apply_shared_split(
            dataset=dataset,
            split_file=args.split_file,
            split_name=split,
            allow_incomplete=args.allow_incomplete_split,
        )

        if role == "fit" and args.fit_limit is not None \
                and args.fit_limit < len(subset.indices):
            subset.indices = sorted(random.sample(list(subset.indices), args.fit_limit))

        examples[role], skipped = group_examples(subset, candidates, tokenizer)
        split_meta[role]["num_scored"] = sum(len(g["gold"]) for g in examples[role].values())
        split_meta[role]["num_skipped_too_long"] = skipped

        logger.info(
            f"{role} ({split}): {split_meta[role]['num_scored']} examples scored, "
            f"{skipped} skipped (gold longer than {args.num_mask} tokens)"
        )

    # -----------------------------------------------------------------------
    # Score every member once; alpha is applied afterwards.
    # -----------------------------------------------------------------------

    raw = {"fit": [], "eval": []}

    for name, soft_prompt, _ in members:
        for role in ("fit", "eval"):
            logger.info(f"Scoring {name} on {role} split...")
            raw[role].append(score_member(
                model, soft_prompt, examples[role], candidates, tokenizer,
                device, args.batch_size, args.max_seq_len,
            ))

    # -----------------------------------------------------------------------
    # Per alpha: single members, uniform ensemble, learned ensemble
    # -----------------------------------------------------------------------

    results_by_alpha = {}

    for alpha in alphas:
        logger.info("=" * 60)
        logger.info(f"alpha = {alpha}")

        stacked = {
            role: stack_members(raw[role], candidates, alpha)
            for role in ("fit", "eval")
        }

        entry = {"members": {}}

        for m, name in enumerate(member_names):
            entry["members"][name] = {
                role: em_metrics(
                    argmax_predictions({r: s[m] for r, s in stacked[role].items()}),
                    examples[role],
                )
                for role in ("fit", "eval")
            }
            logger.info(f"  {name:>10} | test {format_metrics(entry['members'][name]['eval'])}")

        if len(members) > 1:
            entry["uniform"] = {
                role: em_metrics(
                    argmax_predictions({r: s.mean(0) for r, s in stacked[role].items()}),
                    examples[role],
                )
                for role in ("fit", "eval")
            }
            logger.info(f"  {'uniform':>10} | test {format_metrics(entry['uniform']['eval'])}")

            ensemble = fit_ensemble(
                stacked["fit"], examples["fit"], args.weight_steps, args.weight_lr
            )

            with torch.no_grad():
                entry["learned"] = {
                    role: em_metrics(
                        argmax_predictions({r: ensemble(s) for r, s in stacked[role].items()}),
                        examples[role],
                    )
                    for role in ("fit", "eval")
                }

            entry["learned"]["weights"] = dict(zip(member_names, ensemble.weights.tolist()))
            entry["learned"]["temperature"] = ensemble.log_temperature.exp().item()
            logger.info(f"  {'learned':>10} | test {format_metrics(entry['learned']['eval'])}")

        results_by_alpha[str(alpha)] = entry

    # -----------------------------------------------------------------------
    # Model selection on the fit split
    # -----------------------------------------------------------------------

    def fit_em(alpha_key: str, system: str, name: str = None) -> float:
        entry = results_by_alpha[alpha_key]
        result = entry["members"][name] if system == "member" else entry[system]
        return result["fit"]["all"]["macro"]

    # Final system: learned ensemble when there are several members,
    # otherwise the single member.
    final_system = "learned" if len(members) > 1 else "member"
    selected_alpha = max(
        results_by_alpha,
        key=lambda a: fit_em(a, final_system, member_names[0]),
    )

    best_member = max(
        member_names,
        key=lambda n: fit_em(selected_alpha, "member", n),
    )

    selected = results_by_alpha[selected_alpha]
    summary = {
        "selected_alpha": float(selected_alpha),
        "best_member": best_member,
        "best_member_eval": selected["members"][best_member]["eval"],
        # The alpha=0 (raw sum) scores show what length normalisation fixes.
        "best_member_eval_raw_sum": results_by_alpha["0.0"]["members"][best_member]["eval"]
        if "0.0" in results_by_alpha else None,
    }

    if len(members) > 1:
        summary["uniform_eval"] = selected["uniform"]["eval"]
        summary["learned_eval"] = selected["learned"]["eval"]
        summary["learned_weights"] = selected["learned"]["weights"]

    logger.info("=" * 60)
    logger.info(f"Selected alpha on {args.fit_split}: {selected_alpha}")
    logger.info(f"Best single member ({best_member}): {format_metrics(summary['best_member_eval'])}")
    if len(members) > 1:
        logger.info(f"Uniform ensemble:  {format_metrics(summary['uniform_eval'])}")
        logger.info(f"Learned ensemble:  {format_metrics(summary['learned_eval'])}")
        logger.info(f"Learned weights:   {summary['learned_weights']}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    results = {
        "metadata": {
            "language": args.lang,
            "model_name": args.model_name,
            "probe": args.probe,
            "portion": args.portion,
            "split_file": args.split_file,
            "fit_split": args.fit_split,
            "eval_split": args.eval_split,
            "fit_limit": args.fit_limit,
            "num_mask": args.num_mask,
            "alphas": alphas,
            "weight_steps": args.weight_steps,
            "weight_lr": args.weight_lr,
            "seed": args.seed,
            "members": [
                {
                    "name": name,
                    "path": path,
                    "num_prompt_tokens": sp.num_prompt_tokens if sp is not None else 0,
                }
                for name, sp, path in members
            ],
            "num_candidates": {rel: len(c["ids"]) for rel, c in candidates.items()},
            "splits": split_meta,
        },
        "summary": summary,
        "by_alpha": results_by_alpha,
    }

    if args.output:
        os.makedirs(dirname(abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
