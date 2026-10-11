"""
Aggregate the transfer-scaling results into tidy CSV tables.

Expects the layout written by run_transfer_grid.sh / transfer_train.slurm:

    <results_root>/<model>/<run>/seed<seed>/metrics.json
    <results_root>/<model>/<run>/seed<seed>/predictions.jsonl

where <run> is the "-"-joined list of source languages (e.g. en-nl-ru-el).

Writes to <output_dir> (default: <results_root>/summary):

    runs.csv            one row per (model, run, seed, language)
    runs_mean.csv       mean / std over seeds
    significance.csv    paired comparisons of runs on every zero-shot target
                        (stratified bootstrap of the macro-EM difference over
                        the shared test facts, and an exact McNemar test on the
                        per-fact majority vote over seeds)
    em_by_length.csv    EM by gold token-length bucket
    fertility.csv       mean gold length and share of multi-token targets
    paired_same_fact.csv  P(correct in X | correct in English) by length in X
    overlap_hist.csv    number of languages in which each fact is correct
    overlap_jaccard.csv pairwise Jaccard overlap of correct facts
    relation_em.csv     EM per relation (for per-relation differences)

The primary metric is EM with the shared candidate pool ("em_shared"),
macro-averaged over relations, for the trained prompt ("prompt"); the frozen
model without a prompt ("zero") is the reference.

Usage:
    python pipeline_v1/analysis/transfer_summary.py --results_root pipeline_v1/results/transfer
"""

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LANGS = ["en", "nl", "ru", "el", "ko"]
BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 10 ** 6))
SEED_DIR = re.compile(r"^seed(\d+)$")


def bucket_of(n):
    for label, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return label
    return None


def discover(results_root: Path):
    """Yield (model, run, seed, run_dir)."""
    for metrics in sorted(results_root.glob("*/*/seed*/metrics.json")):
        match = SEED_DIR.match(metrics.parent.name)
        if not match:
            continue
        yield (metrics.parent.parent.parent.name, metrics.parent.parent.name,
               int(match.group(1)), metrics.parent)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    rows = []
    with (run_dir / "predictions.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def metrics_rows(model, run, seed, metrics):
    rows = []
    sources = metrics["metadata"]["source_langs"]
    for lang, res in metrics["languages"].items():
        prompt = res["members"]["prompt"]
        zero = res["members"].get("zero_shot")

        def em(member, pool, subset="all"):
            if member is None:
                return None
            return member["em"][pool]["selected"][subset]["macro"]

        row = {
            "model": model, "run": run, "n_sources": len(sources), "seed": seed,
            "lang": lang, "role": res["role"],
            "em_shared_prompt": em(prompt, "shared"),
            "em_shared_zero": em(zero, "shared"),
            "em_full_prompt": em(prompt, "full"),
            "em_full_zero": em(zero, "full"),
            "em_shared_single_prompt": em(prompt, "shared", "single"),
            "em_shared_multi_prompt": em(prompt, "shared", "multi"),
            "em_shared_micro_prompt": prompt["em"]["shared"]["selected"]["all"]["micro"],
            "em_shared_prompt_oracle_alpha": prompt["em"]["shared"]["oracle"]["all"]["macro"],
            "token_acc_prompt": prompt["token_acc"],
            "token_acc_zero": zero["token_acc"] if zero else None,
            "chance_shared": res["chance_em_macro"]["shared"],
            "chance_full": res["chance_em_macro"]["full"],
            "alpha_shared": prompt["em"]["shared"]["alpha"],
            "alpha_full": prompt["em"]["full"]["alpha"],
            "oracle_alpha_shared": prompt["em"]["shared"]["oracle_alpha"],
            "n_scored": res["n_scored"],
            "n_skipped_too_long": res["n_skipped_too_long"],
        }
        for key in ("shared", "full"):
            p, z = row[f"em_{key}_prompt"], row[f"em_{key}_zero"]
            row[f"gain_{key}"] = p - z if z is not None else None
        rows.append(row)
    return rows


def stratified_bootstrap(a_by_rel, b_by_rel, n_boot, rng):
    """
    Bootstrap distribution of macro(A) - macro(B) over facts, resampling
    facts within each relation. Inputs: relation -> (array_a, array_b) of
    per-fact scores (seed-averaged correctness) on the same facts.
    """
    diffs = np.zeros(n_boot)
    for rel in a_by_rel:
        a, b = a_by_rel[rel], b_by_rel[rel]
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        diffs += a[idx].mean(1) - b[idx].mean(1)
    return diffs / len(a_by_rel)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def significance_table(preds, n_boot, seed):
    """Paired comparisons between runs on every language that is zero-shot in both."""
    rng = np.random.default_rng(seed)
    rows = []
    col = "prompt_correct_shared"

    scored = preds[preds["scored"] == True]  # noqa: E712
    for (model, lang), group in scored.groupby(["model", "lang"]):
        zs = group[group["role"] == "zero-shot"]
        runs = sorted(zs["run"].unique(), key=lambda r: (len(r.split("-")), r))

        # Per-fact correctness averaged over seeds, per run.
        per_run = {
            run: zs[zs["run"] == run].groupby(["relation", "sub_uri", "obj_uri"])[col]
            .agg(["mean", "count"])
            for run in runs
        }

        for run_a, run_b in itertools.combinations(runs, 2):
            joined = per_run[run_a].join(per_run[run_b], lsuffix="_a", rsuffix="_b", how="inner")
            if joined.empty:
                continue

            a_by_rel, b_by_rel = {}, {}
            for rel, g in joined.groupby(level="relation"):
                a_by_rel[rel] = g["mean_a"].to_numpy(dtype=float)
                b_by_rel[rel] = g["mean_b"].to_numpy(dtype=float)

            macro_a = np.mean([v.mean() for v in a_by_rel.values()])
            macro_b = np.mean([v.mean() for v in b_by_rel.values()])
            diffs = stratified_bootstrap(a_by_rel, b_by_rel, n_boot, rng)

            # Majority vote over seeds for McNemar.
            maj_a = joined["mean_a"] > 0.5
            maj_b = joined["mean_b"] > 0.5
            b_count = int((maj_a & ~maj_b).sum())
            c_count = int((~maj_a & maj_b).sum())

            rows.append({
                "model": model, "target": lang, "run_a": run_a, "run_b": run_b,
                "n_facts": len(joined), "n_relations": len(a_by_rel),
                "em_a": macro_a, "em_b": macro_b, "diff_a_minus_b": macro_a - macro_b,
                "ci_low": float(np.percentile(diffs, 2.5)),
                "ci_high": float(np.percentile(diffs, 97.5)),
                "p_bootstrap": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
                "mcnemar_a_only": b_count, "mcnemar_b_only": c_count,
                "p_mcnemar": mcnemar_exact(b_count, c_count),
            })
    return pd.DataFrame(rows)


def em_by_length(preds):
    scored = preds[preds["scored"] == True].copy()  # noqa: E712
    scored["bucket"] = scored["gold_len"].map(bucket_of)
    per_seed = (scored.groupby(["model", "run", "seed", "lang", "role", "bucket"])
                .agg(em=("prompt_correct_shared", "mean"), n=("prompt_correct_shared", "size"))
                .reset_index())
    return (per_seed.groupby(["model", "run", "lang", "role", "bucket"])
            .agg(em_mean=("em", "mean"), em_std=("em", "std"), n=("n", "mean"))
            .reset_index())


def fertility(preds):
    one = preds.drop_duplicates(["model", "lang", "relation", "sub_uri", "obj_uri"])
    rows = []
    for (model, lang), g in one.groupby(["model", "lang"]):
        s = g[g["scored"] == True]  # noqa: E712
        rows.append({
            "model": model, "lang": lang,
            "mean_gold_tokens": s["gold_len"].mean(),
            "share_multi_token": (s["gold_len"] > 1).mean(),
            "n_scored": len(s),
            "n_skipped_too_long": int((g["scored"] == False).sum()),  # noqa: E712
        })
    return pd.DataFrame(rows)


def paired_same_fact(preds):
    scored = preds[preds["scored"] == True]  # noqa: E712
    keys = ["relation", "sub_uri", "obj_uri"]
    rows = []
    for (model, run, seed), g in scored.groupby(["model", "run", "seed"]):
        en = g[g["lang"] == "en"].set_index(keys)["prompt_correct_shared"]
        en_correct = en[en == True].index  # noqa: E712
        for lang in sorted(set(g["lang"]) - {"en"}):
            x = g[g["lang"] == lang].set_index(keys)
            x = x.loc[x.index.intersection(en_correct)]
            role = x["role"].iloc[0] if len(x) else None
            for label, lo, hi in BUCKETS:
                sel = x[(x["gold_len"] >= lo) & (x["gold_len"] <= hi)]
                rows.append({
                    "model": model, "run": run, "seed": seed, "lang": lang, "role": role,
                    "bucket": label, "n": len(sel),
                    "p_correct_given_en": sel["prompt_correct_shared"].mean() if len(sel) else None,
                })
    df = pd.DataFrame(rows)
    return (df.groupby(["model", "run", "lang", "role", "bucket"])
            .agg(p_mean=("p_correct_given_en", "mean"), p_std=("p_correct_given_en", "std"),
                 n=("n", "mean"))
            .reset_index())


def overlap_tables(preds):
    scored = preds[preds["scored"] == True]  # noqa: E712
    keys = ["relation", "sub_uri", "obj_uri"]
    hist_rows, jac_rows = [], []

    for (model, run, seed), g in scored.groupby(["model", "run", "seed"]):
        pivot = g.pivot_table(index=keys, columns="lang", values="prompt_correct_shared",
                              aggfunc="first")
        pivot = pivot.dropna()  # facts scored in every evaluated language
        if pivot.empty:
            continue
        counts = pivot.astype(bool).sum(axis=1)
        for k in range(pivot.shape[1] + 1):
            hist_rows.append({"model": model, "run": run, "seed": seed,
                              "num_langs_correct": k, "share": float((counts == k).mean()),
                              "n_facts": len(pivot)})
        for a, b in itertools.combinations(pivot.columns, 2):
            ca, cb = pivot[a].astype(bool), pivot[b].astype(bool)
            union = int((ca | cb).sum())
            jac_rows.append({"model": model, "run": run, "seed": seed, "lang_a": a, "lang_b": b,
                             "jaccard": int((ca & cb).sum()) / union if union else None})

    hist = (pd.DataFrame(hist_rows).groupby(["model", "run", "num_langs_correct"])
            .agg(share_mean=("share", "mean"), share_std=("share", "std")).reset_index()
            if hist_rows else pd.DataFrame())
    jac = (pd.DataFrame(jac_rows).groupby(["model", "run", "lang_a", "lang_b"])
           .agg(jaccard_mean=("jaccard", "mean")).reset_index()
           if jac_rows else pd.DataFrame())
    return hist, jac


def relation_em(preds):
    scored = preds[preds["scored"] == True]  # noqa: E712
    per_seed = (scored.groupby(["model", "run", "seed", "lang", "role", "relation"])
                ["prompt_correct_shared"].mean().reset_index())
    return (per_seed.groupby(["model", "run", "lang", "role", "relation"])
            ["prompt_correct_shared"].agg(["mean", "std", "count"]).reset_index()
            .rename(columns={"mean": "em_mean", "std": "em_std", "count": "n_seeds"}))


def main():
    parser = argparse.ArgumentParser(description="Summarise the transfer-scaling grid")
    parser.add_argument("--results_root", default="pipeline_v1/results/transfer")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.results_root)
    out = Path(args.output_dir) if args.output_dir else root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    metric_rows, frames = [], []
    for model, run, seed, run_dir in discover(root):
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        metric_rows.extend(metrics_rows(model, run, seed, metrics))

        preds = load_predictions(run_dir)
        preds["model"], preds["run"], preds["seed"] = model, run, seed
        frames.append(preds)

    if not metric_rows:
        raise SystemExit(f"No results found under {root}/<model>/<run>/seed*/metrics.json")

    runs = pd.DataFrame(metric_rows).sort_values(["model", "n_sources", "run", "lang", "seed"])
    runs.to_csv(out / "runs.csv", index=False)

    numeric = [c for c in runs.columns
               if c not in ("model", "run", "n_sources", "seed", "lang", "role")]
    grouped = runs.groupby(["model", "run", "n_sources", "lang", "role"])
    mean = grouped[numeric].mean().add_suffix("_mean")
    std = grouped[numeric].std().add_suffix("_std")
    runs_mean = mean.join(std).join(grouped.size().rename("n_seeds")).reset_index()
    runs_mean.to_csv(out / "runs_mean.csv", index=False)

    preds = pd.concat(frames, ignore_index=True)
    significance_table(preds, args.n_boot, args.seed).to_csv(out / "significance.csv", index=False)
    em_by_length(preds).to_csv(out / "em_by_length.csv", index=False)
    fertility(preds).to_csv(out / "fertility.csv", index=False)
    paired_same_fact(preds).to_csv(out / "paired_same_fact.csv", index=False)
    hist, jac = overlap_tables(preds)
    hist.to_csv(out / "overlap_hist.csv", index=False)
    jac.to_csv(out / "overlap_jaccard.csv", index=False)
    relation_em(preds).to_csv(out / "relation_em.csv", index=False)

    # Console overview: primary metric per run and language.
    view = runs_mean.pivot_table(index=["model", "run"], columns="lang",
                                 values="em_shared_prompt_mean")
    print("\nEM (shared candidates, macro over relations), mean over seeds:\n")
    print(view.reindex(columns=[l for l in LANGS if l in view.columns]).round(4).to_string())
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
