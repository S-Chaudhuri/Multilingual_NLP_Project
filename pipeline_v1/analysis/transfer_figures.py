"""
Figures for the transfer-scaling grid.

Reads the tables written by transfer_summary.py, the training histories
written by multilingual_train.py and (optionally) annotated error CSVs from
error_sampling.py, and writes PNG figures:

    F1_transfer_heatmap_<model>.png    runs x languages: EM, and gain over no prompt
    F2_scaling_<model>.png             EM on each zero-shot target vs number of source languages
    F3_transfer_bars_<model>.png       per target: no prompt vs every run where it is zero-shot
    F4_fertility_<model>.png           (a) gold length vs EM drop, (b) EM by gold-length bucket
    F5_same_fact_<model>.png           P(correct in X | correct in English) by gold length in X
    F6_overlap_<model>.png             number of languages a fact is correct in + Jaccard overlap
    F7_training_curves_<model>.png     validation loss per source language and epoch
    F8_error_taxonomy.png              annotated failure modes per language (if labels exist)
    F9_relation_diff_<model>.png       per-relation EM difference, max diversity vs control (Korean)

Usage:
    python pipeline_v1/analysis/transfer_figures.py \
        --summary_dir pipeline_v1/results/transfer/summary \
        --ckpt_root checkpoints/transfer \
        --annotation_dir pipeline_v1/results/transfer/annotation \
        --fig_dir pipeline_v1/results/transfer/figures
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

LANGS = ["en", "nl", "ru", "el", "ko"]
LANG_COLORS = {"en": "#4C72B0", "nl": "#DD8452", "ru": "#55A868", "el": "#C44E52", "ko": "#8172B3"}
RUN_ORDER = ["en", "en-nl", "en-ru", "en-nl-ru-el", "en-ru-el-ko"]
RUN_LABELS = {
    "en": "R1 en",
    "en-nl": "R2 en+nl",
    "en-ru": "R3 en+ru",
    "en-nl-ru-el": "R4 en+nl+ru+el",
    "en-ru-el-ko": "R5 en+ru+el+ko",
}
MODEL_LABELS = {"mbert_base": "mBERT", "xlmr_base": "XLM-R"}
BUCKET_ORDER = ["1", "2", "3-4", "5+"]
CONTROL_RUN, DIVERSE_RUN = "en", "en-nl-ru-el"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def ordered_runs(runs):
    known = [r for r in RUN_ORDER if r in set(runs)]
    return known + sorted(set(runs) - set(known), key=lambda r: (len(r.split("-")), r))


def run_label(run):
    return RUN_LABELS.get(run, run)


def model_label(model):
    return MODEL_LABELS.get(model, model)


def save(fig, fig_dir: Path, name: str):
    path = fig_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# F1 / F2 / F3: transfer results
# ---------------------------------------------------------------------------

def fig_heatmap(runs_mean, model, fig_dir):
    df = runs_mean[runs_mean["model"] == model]
    runs = ordered_runs(df["run"].unique())
    langs = [l for l in LANGS if l in set(df["lang"])]

    panels = [
        ("em_shared_prompt", "EM (shared candidates)", "viridis"),
        ("gain_shared", "Gain over no prompt", "RdBu"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(4.2 + 1.1 * len(langs) * 2, 0.6 * len(runs) + 1.6))

    for ax, (metric, title, cmap) in zip(axes, panels):
        mean = np.full((len(runs), len(langs)), np.nan)
        std = np.full_like(mean, np.nan)
        source = np.zeros_like(mean, dtype=bool)
        for i, run in enumerate(runs):
            for j, lang in enumerate(langs):
                row = df[(df["run"] == run) & (df["lang"] == lang)]
                if len(row):
                    mean[i, j] = row[f"{metric}_mean"].iloc[0]
                    std[i, j] = row[f"{metric}_std"].iloc[0]
                    source[i, j] = row["role"].iloc[0] == "source"

        if cmap == "RdBu":
            bound = np.nanmax(np.abs(mean)) or 1.0
            im = ax.imshow(mean, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
        else:
            im = ax.imshow(mean, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(runs)):
            for j in range(len(langs)):
                if np.isnan(mean[i, j]):
                    continue
                text = f"{mean[i, j]:.3f}" + ("" if np.isnan(std[i, j]) else f"\n±{std[i, j]:.3f}")
                r, g, b, _ = im.cmap(im.norm(mean[i, j]))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if luminance < 0.5 else "black")
                if source[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor="grey", lw=0))

        ax.set_xticks(range(len(langs)), langs)
        ax.set_yticks(range(len(runs)), [run_label(r) for r in runs])
        ax.set_title(f"{model_label(model)}: {title}")

    fig.text(0.5, -0.02, "Hatched cells: language was a source (training) language. "
             "Mean ± std over seeds.", ha="center", fontsize=8, color="grey")
    fig.tight_layout()
    save(fig, fig_dir, f"F1_transfer_heatmap_{model}")


def fig_scaling(runs_mean, model, fig_dir):
    df = runs_mean[runs_mean["model"] == model]
    targets = [l for l in LANGS if ((df["lang"] == l) & (df["role"] == "zero-shot")).any()]
    if not targets:
        return

    fig, axes = plt.subplots(1, len(targets), figsize=(3.1 * len(targets), 3.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, lang in zip(axes, targets):
        zs = df[(df["lang"] == lang) & (df["role"] == "zero-shot")]
        for run in ordered_runs(zs["run"].unique()):
            row = zs[zs["run"] == run].iloc[0]
            ax.errorbar(row["n_sources"], row["em_shared_prompt_mean"],
                        yerr=row["em_shared_prompt_std"], fmt="o", capsize=3,
                        color=LANG_COLORS[lang])
            ax.annotate(run_label(run).split()[0], (row["n_sources"], row["em_shared_prompt_mean"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        base = df[df["lang"] == lang]["em_shared_zero_mean"].mean()
        ax.axhline(base, ls="--", color="grey", lw=1, label="no prompt")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("number of source languages")
        ax.set_title(f"target: {lang}")
    axes[0].set_ylabel("zero-shot EM (shared candidates)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model_label(model)}: transfer vs source diversity (fixed budget)", y=1.02)
    fig.tight_layout()
    save(fig, fig_dir, f"F2_scaling_{model}")


def fig_transfer_bars(runs_mean, model, fig_dir):
    df = runs_mean[runs_mean["model"] == model]
    targets = [l for l in LANGS if ((df["lang"] == l) & (df["role"] == "zero-shot")).any()]
    if not targets:
        return

    fig, ax = plt.subplots(figsize=(1.9 * len(targets) + 2, 3.2))
    x = 0
    ticks, tick_labels = [], []
    runs_all = ordered_runs(df["run"].unique())
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(runs_all)))
    run_color = dict(zip(runs_all, colors))
    legend_done = set()

    for lang in targets:
        sub = df[df["lang"] == lang]
        start = x
        ax.bar(x, sub["em_shared_zero_mean"].mean(), color="lightgrey",
               label="no prompt" if "zero" not in legend_done else None)
        legend_done.add("zero")
        x += 1
        for run in runs_all:
            row = sub[(sub["run"] == run) & (sub["role"] == "zero-shot")]
            if not len(row):
                continue
            ax.bar(x, row["em_shared_prompt_mean"].iloc[0], yerr=row["em_shared_prompt_std"].iloc[0],
                   color=run_color[run], capsize=2,
                   label=run_label(run) if run not in legend_done else None)
            legend_done.add(run)
            x += 1
        ticks.append((start + x - 1) / 2)
        tick_labels.append(lang)
        x += 1

    ax.set_xticks(ticks, tick_labels)
    ax.set_ylabel("EM (shared candidates)")
    ax.set_title(f"{model_label(model)}: zero-shot EM per target language")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, fig_dir, f"F3_transfer_bars_{model}")


# ---------------------------------------------------------------------------
# F4 / F5: tokenisation
# ---------------------------------------------------------------------------

def fig_fertility(runs_mean, fertility, em_len, model, fig_dir):
    fert = fertility[fertility["model"] == model].set_index("lang")
    df = runs_mean[(runs_mean["model"] == model) & (runs_mean["run"] == CONTROL_RUN)]
    if df.empty or fert.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    en = df[df["lang"] == "en"]["em_shared_prompt_mean"]
    ax = axes[0]
    if len(en):
        for _, row in df[df["lang"] != "en"].iterrows():
            lang = row["lang"]
            if lang not in fert.index:
                continue
            drop = en.iloc[0] - row["em_shared_prompt_mean"]
            ax.scatter(fert.loc[lang, "mean_gold_tokens"], drop, color=LANG_COLORS[lang], s=40)
            ax.annotate(lang, (fert.loc[lang, "mean_gold_tokens"], drop),
                        textcoords="offset points", xytext=(5, 3))
    ax.set_xlabel("mean gold tokens (test targets)")
    ax.set_ylabel(f"EM drop from en ({run_label(CONTROL_RUN)})")
    ax.set_title("(a) fertility vs transfer drop (n = 4 languages)")

    ax = axes[1]
    sub = em_len[em_len["model"] == model]
    for run, style in ((CONTROL_RUN, "-"), (DIVERSE_RUN, "--")):
        for lang in LANGS:
            line = sub[(sub["run"] == run) & (sub["lang"] == lang)]
            if line.empty:
                continue
            line = line.set_index("bucket").reindex(BUCKET_ORDER)
            ax.plot(BUCKET_ORDER, line["em_mean"], style, marker="o", ms=3,
                    color=LANG_COLORS[lang],
                    label=f"{lang} ({run_label(run).split()[0]})")
    ax.set_xlabel("gold length (tokens)")
    ax.set_ylabel("EM (micro, shared candidates)")
    ax.set_title(f"(b) EM by gold length: {run_label(CONTROL_RUN)} (solid) vs "
                 f"{run_label(DIVERSE_RUN).split()[0]} (dashed)")
    ax.legend(frameon=False, fontsize=6, ncol=2)

    fig.suptitle(model_label(model), y=1.02)
    fig.tight_layout()
    save(fig, fig_dir, f"F4_fertility_{model}")


def fig_same_fact(paired, model, fig_dir):
    sub = paired[paired["model"] == model]
    runs = [r for r in ordered_runs(sub["run"].unique())]
    if not runs:
        return

    fig, axes = plt.subplots(1, len(runs), figsize=(2.8 * len(runs), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, runs):
        for lang in LANGS[1:]:
            line = sub[(sub["run"] == run) & (sub["lang"] == lang)]
            if line.empty:
                continue
            role = line["role"].iloc[0]
            line = line.set_index("bucket").reindex(BUCKET_ORDER)
            ax.plot(BUCKET_ORDER, line["p_mean"], "-" if role == "zero-shot" else ":",
                    marker="o", ms=3, color=LANG_COLORS[lang], label=lang)
        ax.set_title(run_label(run), fontsize=8)
        ax.set_xlabel("gold length in X")
    axes[0].set_ylabel("P(correct in X | correct in en)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model_label(model)}: same facts across languages "
                 "(solid = zero-shot, dotted = source)", y=1.03)
    fig.tight_layout()
    save(fig, fig_dir, f"F5_same_fact_{model}")


# ---------------------------------------------------------------------------
# F6: overlap of correct facts
# ---------------------------------------------------------------------------

def fig_overlap(hist, jac, model, fig_dir):
    if hist.empty:
        return
    h = hist[hist["model"] == model]
    j = jac[jac["model"] == model] if not jac.empty else jac
    runs = ordered_runs(h["run"].unique())
    if not runs:
        return

    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(3.0 * (len(runs) + 1), 2.9))
    ax = axes[0]
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        r = h[h["run"] == run]
        ax.bar(r["num_langs_correct"] + i * width - 0.4 + width / 2, r["share_mean"],
               width=width, label=run_label(run))
    ax.set_xlabel("number of languages correct")
    ax.set_ylabel("share of facts")
    ax.set_title("correct facts across languages")
    ax.legend(frameon=False, fontsize=6)

    for ax, run in zip(axes[1:], runs):
        mat = np.full((len(LANGS), len(LANGS)), np.nan)
        for _, row in j[j["run"] == run].iterrows():
            a, b = LANGS.index(row["lang_a"]), LANGS.index(row["lang_b"])
            mat[a, b] = mat[b, a] = row["jaccard_mean"]
        np.fill_diagonal(mat, 1.0)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma")
        for a in range(len(LANGS)):
            for b in range(len(LANGS)):
                if not np.isnan(mat[a, b]) and a != b:
                    ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if mat[a, b] < 0.5 else "black")
        ax.set_xticks(range(len(LANGS)), LANGS)
        ax.set_yticks(range(len(LANGS)), LANGS)
        ax.set_title(f"Jaccard: {run_label(run)}", fontsize=8)
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(model_label(model), y=1.03)
    fig.tight_layout()
    save(fig, fig_dir, f"F6_overlap_{model}")


# ---------------------------------------------------------------------------
# F7: training curves
# ---------------------------------------------------------------------------

def load_histories(ckpt_root: Path):
    rows = []
    for path in ckpt_root.glob("*/*/seed*/history_*.json"):
        model, run = path.parts[-4], path.parts[-3]
        seed = int(re.sub(r"\D", "", path.parent.name) or 0)
        with path.open("r", encoding="utf-8") as f:
            for entry in json.load(f):
                for lang, loss in entry.get("val_loss_by_lang", {}).items():
                    rows.append({"model": model, "run": run, "seed": seed,
                                 "epoch": entry["epoch"], "lang": lang, "val_loss": loss,
                                 "train_loss": entry.get("train_loss_by_lang", {}).get(lang)})
    return pd.DataFrame(rows)


def fig_training_curves(histories, model, fig_dir):
    if histories.empty:
        return
    h = histories[histories["model"] == model]
    runs = ordered_runs(h["run"].unique())
    if not runs:
        return

    fig, axes = plt.subplots(2, len(runs), figsize=(2.7 * len(runs), 4.8), sharex=True)
    axes = np.array(axes).reshape(2, len(runs))
    for col, run in enumerate(runs):
        r = h[h["run"] == run].groupby(["lang", "epoch"])[["val_loss", "train_loss"]].mean().reset_index()
        for lang in LANGS:
            line = r[r["lang"] == lang]
            if line.empty:
                continue
            axes[0, col].plot(line["epoch"], line["train_loss"], color=LANG_COLORS[lang], label=lang)
            axes[1, col].plot(line["epoch"], line["val_loss"], color=LANG_COLORS[lang], label=lang)
        axes[0, col].set_title(run_label(run), fontsize=8)
        axes[1, col].set_xlabel("epoch")
    axes[0, 0].set_ylabel("train loss per language")
    axes[1, 0].set_ylabel("val loss per language")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model_label(model)}: training curves (mean over seeds)", y=1.01)
    fig.tight_layout()
    save(fig, fig_dir, f"F7_training_curves_{model}")


# ---------------------------------------------------------------------------
# F8: error taxonomy, F9: per-relation differences
# ---------------------------------------------------------------------------

def fig_error_taxonomy(annotation_dir: Path, fig_dir):
    files = sorted(annotation_dir.glob("*.csv")) if annotation_dir.exists() else []
    frames = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    frames = [f for f in frames if "label" in f and f["label"].notna().any()]
    if not frames:
        print("  F8 skipped: no annotated error CSVs (fill in the `label` column first)")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
    df["group"] = df["model"].map(model_label) + " " + df["run"].map(run_label).str.split().str[0] \
        + " / " + df["lang"]
    table = pd.crosstab(df["group"], df["label"], normalize="index")

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1.5))
    table.plot(kind="barh", stacked=True, ax=ax, colormap="Set2", width=0.75)
    ax.set_xlabel("share of annotated errors")
    ax.set_ylabel("")
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Failure modes of incorrect predictions")
    fig.tight_layout()
    save(fig, fig_dir, "F8_error_taxonomy")


def fig_relation_diff(relation_em, model, fig_dir, target="ko"):
    sub = relation_em[(relation_em["model"] == model) & (relation_em["lang"] == target)]
    a = sub[sub["run"] == DIVERSE_RUN].set_index("relation")["em_mean"]
    b = sub[sub["run"] == CONTROL_RUN].set_index("relation")["em_mean"]
    diff = (a - b).dropna().sort_values()
    if diff.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(len(diff)), diff.values,
           color=["#C44E52" if v < 0 else "#55A868" for v in diff.values])
    ax.set_xticks(range(len(diff)), diff.index, rotation=90, fontsize=6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel(f"EM {run_label(DIVERSE_RUN).split()[0]}\n- {run_label(CONTROL_RUN).split()[0]}")
    ax.set_title(f"{model_label(model)}: per-relation change on {target} "
                 f"({run_label(DIVERSE_RUN)} vs {run_label(CONTROL_RUN)})")
    fig.tight_layout()
    save(fig, fig_dir, f"F9_relation_diff_{model}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_csv(path: Path):
    if path.exists() and path.stat().st_size > 1:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame()


def make_all(summary_dir, ckpt_root, annotation_dir, fig_dir):
    summary_dir, fig_dir = Path(summary_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs_mean = read_csv(summary_dir / "runs_mean.csv")
    if runs_mean.empty:
        raise SystemExit(f"{summary_dir}/runs_mean.csv not found: run transfer_summary.py first")

    fertility = read_csv(summary_dir / "fertility.csv")
    em_len = read_csv(summary_dir / "em_by_length.csv")
    paired = read_csv(summary_dir / "paired_same_fact.csv")
    hist = read_csv(summary_dir / "overlap_hist.csv")
    jac = read_csv(summary_dir / "overlap_jaccard.csv")
    rel = read_csv(summary_dir / "relation_em.csv")
    histories = load_histories(Path(ckpt_root))

    for model in sorted(runs_mean["model"].unique()):
        print(f"{model_label(model)}:")
        fig_heatmap(runs_mean, model, fig_dir)
        fig_scaling(runs_mean, model, fig_dir)
        fig_transfer_bars(runs_mean, model, fig_dir)
        if not fertility.empty and not em_len.empty:
            fig_fertility(runs_mean, fertility, em_len, model, fig_dir)
        if not paired.empty:
            fig_same_fact(paired, model, fig_dir)
        fig_overlap(hist, jac, model, fig_dir)
        fig_training_curves(histories, model, fig_dir)
        if not rel.empty:
            fig_relation_diff(rel, model, fig_dir)

    fig_error_taxonomy(Path(annotation_dir), fig_dir)


def main():
    parser = argparse.ArgumentParser(description="Figures for the transfer-scaling grid")
    parser.add_argument("--summary_dir", default="pipeline_v1/results/transfer/summary")
    parser.add_argument("--ckpt_root", default="checkpoints/transfer")
    parser.add_argument("--annotation_dir", default="pipeline_v1/results/transfer/annotation")
    parser.add_argument("--fig_dir", default="pipeline_v1/results/transfer/figures")
    args = parser.parse_args()
    make_all(args.summary_dir, args.ckpt_root, args.annotation_dir, args.fig_dir)


if __name__ == "__main__":
    main()