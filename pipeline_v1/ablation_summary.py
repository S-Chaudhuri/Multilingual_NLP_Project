"""
Aggregate the prompt-length ablation and ensembling results into tables.

Expects the layout written by run_ablation.sh:

    <results_root>/<model>/<lang>/eval_P<k>_seed<s>.json   (prompt_tuning_eval.py)
    <results_root>/<model>/<lang>/ensemble.json            (prompt_ensemble.py)

Writes to <results_root>:

    capacity.csv          one row per (model, lang, k, seed): zero-shot vs prompt
    model_comparison.csv  gains per (model, k), averaged over languages and seeds
    ensemble.csv          one row per (model, lang): single prompts vs ensembles

and prints the same tables as Markdown.

Two metrics are reported:
    token_acc  teacher-forced token accuracy at the gold number of masks
               (prompt_tuning_eval.py)
    em         Exact Match of the top-ranked candidate object, macro-averaged
               over relations, at the length-normalisation alpha selected on
               validation (prompt_ensemble.py)

Usage:
    python pipeline_v1/ablation_summary.py --results_root results/ablation
"""

import os
import re
import csv
import json
import argparse
from collections import defaultdict
from statistics import mean


EVAL_FILE = re.compile(r"^eval_P(\d+)_seed(\d+)\.json$")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def em(metrics, subset="all"):
    return metrics[subset]["macro"] if metrics else None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_table(rows, columns, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: fmt(row.get(c)) for c in columns})


def print_markdown(title, rows, columns):
    print(f"\n### {title}\n")
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")


def main():
    parser = argparse.ArgumentParser(description="Summarise ablation results")
    parser.add_argument("--results_root", type=str, default="results/ablation")
    args = parser.parse_args()

    capacity = []
    ensembles = []

    for model in sorted(os.listdir(args.results_root)):
        model_dir = os.path.join(args.results_root, model)
        if not os.path.isdir(model_dir):
            continue

        for lang in sorted(os.listdir(model_dir)):
            lang_dir = os.path.join(model_dir, lang)
            if not os.path.isdir(lang_dir):
                continue

            ensemble_path = os.path.join(lang_dir, "ensemble.json")
            ensemble = load_json(ensemble_path) if os.path.exists(ensemble_path) else None

            selected = None
            if ensemble is not None:
                alpha_key = str(ensemble["summary"]["selected_alpha"])
                selected = ensemble["by_alpha"][alpha_key]

            def member_em(name, subset="all"):
                if selected is None or name not in selected["members"]:
                    return None
                return em(selected["members"][name]["eval"], subset)

            # ---------------------------------------------------------------
            # Capacity: one row per trained prompt
            # ---------------------------------------------------------------

            for file in sorted(os.listdir(lang_dir)):
                match = EVAL_FILE.match(file)
                if not match:
                    continue

                k, seed = int(match.group(1)), int(match.group(2))
                result = load_json(os.path.join(lang_dir, file))
                member = f"P{k}_s{seed}"

                zero_acc = result.get("zero_shot", {}).get("accuracy")
                prompt_acc = result["prompt_tuned"]["accuracy"]
                zero_em = member_em("zero_shot")
                prompt_em = member_em(member)

                capacity.append({
                    "model": model,
                    "lang": lang,
                    "k": k,
                    "seed": seed,
                    "token_acc_zero_shot": zero_acc,
                    "token_acc_prompt": prompt_acc,
                    "token_acc_gain": prompt_acc - zero_acc if zero_acc is not None else None,
                    "em_zero_shot": zero_em,
                    "em_prompt": prompt_em,
                    "em_gain": prompt_em - zero_em
                    if prompt_em is not None and zero_em is not None else None,
                    "em_prompt_single": member_em(member, "single"),
                    "em_prompt_multi": member_em(member, "multi"),
                })

            # ---------------------------------------------------------------
            # Ensembling: one row per (model, lang)
            # ---------------------------------------------------------------

            if ensemble is not None:
                summary = ensemble["summary"]
                best = summary["best_member"]
                raw = summary.get("best_member_eval_raw_sum")
                weights = summary.get("learned_weights") or {}

                ensembles.append({
                    "model": model,
                    "lang": lang,
                    "alpha": summary["selected_alpha"],
                    "em_zero_shot": member_em("zero_shot"),
                    "best_member": best,
                    "em_best_member": em(summary["best_member_eval"]),
                    "em_uniform": em(summary.get("uniform_eval")),
                    "em_learned": em(summary.get("learned_eval")),
                    "learned_minus_best": em(summary["learned_eval"]) - em(summary["best_member_eval"])
                    if summary.get("learned_eval") else None,
                    "best_multi_raw_sum": em(raw, "multi") if raw else None,
                    "best_multi_normalised": em(summary["best_member_eval"], "multi"),
                    "learned_multi": em(summary.get("learned_eval"), "multi"),
                    "weights": " ".join(f"{n}={w:.2f}" for n, w in weights.items()),
                })

    capacity.sort(key=lambda r: (r["model"], r["lang"], r["k"], r["seed"]))

    if not capacity and not ensembles:
        raise SystemExit(f"No results found under {args.results_root}")

    # -----------------------------------------------------------------------
    # Model comparison: WordPiece (mBERT) vs SentencePiece (XLM-R) per k
    # -----------------------------------------------------------------------

    groups = defaultdict(list)
    for row in capacity:
        groups[(row["model"], row["k"])].append(row)

    def mean_of(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return mean(values) if values else None

    comparison = [
        {
            "model": model,
            "k": k,
            "num_runs": len(rows),
            "langs": ",".join(sorted({r["lang"] for r in rows})),
            "token_acc_prompt": mean_of(rows, "token_acc_prompt"),
            "token_acc_gain": mean_of(rows, "token_acc_gain"),
            "em_prompt": mean_of(rows, "em_prompt"),
            "em_gain": mean_of(rows, "em_gain"),
            "em_prompt_multi": mean_of(rows, "em_prompt_multi"),
        }
        for (model, k), rows in sorted(groups.items())
    ]

    # -----------------------------------------------------------------------
    # Write and print
    # -----------------------------------------------------------------------

    tables = [
        ("Capacity: prompt length k (per language)", capacity, "capacity.csv",
         ["model", "lang", "k", "seed", "token_acc_zero_shot", "token_acc_prompt",
          "token_acc_gain", "em_zero_shot", "em_prompt", "em_gain",
          "em_prompt_single", "em_prompt_multi"]),
        ("Soft prompt gains: mBERT (WordPiece) vs XLM-R (SentencePiece)", comparison,
         "model_comparison.csv",
         ["model", "k", "num_runs", "langs", "token_acc_prompt", "token_acc_gain",
          "em_prompt", "em_gain", "em_prompt_multi"]),
        ("Ensembling and length normalisation (test EM, macro)", ensembles, "ensemble.csv",
         ["model", "lang", "alpha", "em_zero_shot", "best_member", "em_best_member",
          "em_uniform", "em_learned", "learned_minus_best", "best_multi_raw_sum",
          "best_multi_normalised", "learned_multi", "weights"]),
    ]

    for title, rows, filename, columns in tables:
        if not rows:
            continue
        path = os.path.join(args.results_root, filename)
        write_table(rows, columns, path)
        print_markdown(title, rows, columns)
        print(f"\n(saved to {path})")


if __name__ == "__main__":
    main()
