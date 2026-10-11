"""
Validate the failure-mode labels of the error CSVs written by error_sampling.py.

Two steps:

  sample  Draw a blind, stratified sample (per_block rows per file x language)
          from the labelled CSVs. Writes
              <annotation_dir>/validation/validation_blind.csv  (give to annotators;
                                                                no model/run/labels)
              <annotation_dir>/validation/validation_key.csv    (keep; holds the labels)

  score   Compare annotator copies of validation_blind.csv with the key:
          raw agreement, Cohen's kappa (overall and per language), confusion
          matrix, human-human kappa on doubly annotated rows. Disagreements are
          written to <annotation_dir>/validation/disagreements.csv.

The validation/ subfolder is not read by transfer_figures.py (it only globs
<annotation_dir>/*.csv).

Example:
    python pipeline_v1/analysis/annotation_agreement.py sample --per_block 20
    python pipeline_v1/analysis/annotation_agreement.py score annotator_a.csv annotator_b.csv
"""

import argparse
from pathlib import Path

import pandas as pd

LABELS = ["subject_repetition", "wrong_entity_type", "language_specific_bias",
          "non_informative", "other"]
BLIND_COLUMNS = ["id", "lang", "relation", "sub_label", "query", "gold",
                 "ranked_prediction", "greedy_prediction"]


def cohen_kappa(a, b, labels=LABELS):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    n = len(a)
    if n == 0:
        return float("nan")
    p_obs = (a.values == b.values).mean()
    pa = a.value_counts(normalize=True).reindex(labels, fill_value=0)
    pb = b.value_counts(normalize=True).reindex(labels, fill_value=0)
    p_exp = float((pa * pb).sum())
    return float("nan") if p_exp == 1 else (p_obs - p_exp) / (1 - p_exp)


def make_sample(annotation_dir: Path, per_block: int, seed: int):
    frames = []
    for path in sorted(annotation_dir.glob("errors_*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["source_file"] = path.name
        df["source_line"] = df.index + 2  # header = line 1
        frames.append(df)
    if not frames:
        raise SystemExit(f"no errors_*.csv in {annotation_dir}")
    df = pd.concat(frames, ignore_index=True)

    sample = df.groupby(["source_file", "lang"]).sample(n=per_block, random_state=seed)
    # grouped by language (annotators can split by language), shuffled within language
    sample = (sample.sample(frac=1, random_state=seed + 1)
                    .sort_values("lang", kind="stable")
                    .reset_index(drop=True))
    sample.insert(0, "id", range(1, len(sample) + 1))

    out = annotation_dir / "validation"
    out.mkdir(exist_ok=True)
    blind = sample[BLIND_COLUMNS].copy()
    blind["label"] = ""
    blind["notes"] = ""
    blind["annotator"] = ""
    blind.to_csv(out / "validation_blind.csv", index=False, encoding="utf-8-sig")

    key = sample[["id", "source_file", "source_line", "model", "run", "lang", "label", "notes"]]
    key = key.rename(columns={"label": "ref_label", "notes": "ref_notes"})
    key.to_csv(out / "validation_key.csv", index=False, encoding="utf-8-sig")

    print(f"{len(sample)} of {len(df)} rows ({len(sample) / len(df):.0%}) -> {out}")
    print(sample.groupby("lang").size().to_string())


def score(annotation_dir: Path, filled_paths):
    vdir = annotation_dir / "validation"
    key = pd.read_csv(vdir / "validation_key.csv", encoding="utf-8-sig")
    filled = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in filled_paths],
                       ignore_index=True)
    filled["label"] = filled["label"].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(filled["label"]) - set(LABELS) - {"nan", ""})
    if unknown:
        print("unknown labels ignored:", unknown)
    filled = filled[filled["label"].isin(LABELS)]

    m = filled.merge(key, on="id", suffixes=("", "_key"))
    print(f"rows compared: {len(m)}")
    print(f"raw agreement: {(m.label == m.ref_label).mean():.3f}")
    print(f"Cohen's kappa: {cohen_kappa(m.label, m.ref_label):.3f}")
    for lang, g in m.groupby("lang"):
        print(f"  {lang}: n={len(g):3d}  agree={(g.label == g.ref_label).mean():.2f}  "
              f"kappa={cohen_kappa(g.label, g.ref_label):.2f}")

    cm = pd.crosstab(pd.Categorical(m.label, LABELS), pd.Categorical(m.ref_label, LABELS),
                     rownames=["annotator"], colnames=["reference"], dropna=False)
    print(cm.to_string())

    if "annotator" in m and m["annotator"].nunique() >= 2:
        wide = m.pivot_table(index="id", columns="annotator", values="label", aggfunc="first")
        names = list(wide.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                both = wide[[names[i], names[j]]].dropna()
                if len(both) >= 10:
                    k = cohen_kappa(both[names[i]], both[names[j]])
                    print(f"human-human {names[i]} vs {names[j]}: n={len(both)} kappa={k:.3f}")

    m[m.label != m.ref_label].to_csv(vdir / "disagreements.csv", index=False, encoding="utf-8-sig")
    print("disagreements ->", vdir / "disagreements.csv")


def main():
    parser = argparse.ArgumentParser(description="Validate failure-mode annotations")
    parser.add_argument("--annotation_dir", default="pipeline_v1/results/transfer/annotation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sample = sub.add_parser("sample", help="write a blind validation sample")
    p_sample.add_argument("--per_block", type=int, default=20,
                          help="rows per (file, language) block")
    p_sample.add_argument("--seed", type=int, default=0)
    p_score = sub.add_parser("score", help="agreement of annotator files with the key")
    p_score.add_argument("filled", nargs="+", help="annotated copies of validation_blind.csv")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    if args.cmd == "sample":
        make_sample(annotation_dir, args.per_block, args.seed)
    else:
        score(annotation_dir, args.filled)


if __name__ == "__main__":
    main()
