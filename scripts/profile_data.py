"""Profile the competition data and write the results to eda/.

Run from the repository root:  python -m scripts.profile_data

Writes:
  eda/profile.md               human-readable summary
  eda/column_summary.csv       one row per column
  eda/target_rate_by_level.csv target rate per level of each low-cardinality column
  eda/univariate_auc.csv       single-feature ROC AUC, as a crude signal ranking

Outputs are aggregate statistics, not competition data, so they are safe to
commit. The raw CSVs stay gitignored.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.data import coerce_target

EDA_DIR = config.ROOT / "eda"
MAX_LEVELS = 20  # a column with more distinct values is not broken down by level


def _fmt(x: float, places: int = 4) -> str:
    return "—" if pd.isna(x) else f"{x:.{places}f}"


def column_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in train.columns:
        s = train[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "kind": "numeric" if numeric else "categorical",
            "n_unique": int(s.nunique(dropna=True)),
            "missing_train": int(s.isna().sum()),
            "missing_train_pct": round(100 * s.isna().mean(), 4),
            "missing_test": int(test[col].isna().sum()) if col in test.columns else np.nan,
            "in_test": col in test.columns,
        }
        if numeric:
            desc = s.describe()
            row.update(
                min=desc.get("min"),
                q1=desc.get("25%"),
                median=desc.get("50%"),
                q3=desc.get("75%"),
                max=desc.get("max"),
                mean=desc.get("mean"),
                std=desc.get("std"),
                test_mean=test[col].mean() if col in test.columns else np.nan,
            )
        else:
            counts = s.value_counts(dropna=True)
            row["top_level"] = counts.index[0] if len(counts) else None
            row["top_level_share"] = round(counts.iloc[0] / len(s), 4) if len(counts) else np.nan
            if col in test.columns:
                only_test = set(test[col].dropna().unique()) - set(s.dropna().unique())
                row["levels_only_in_test"] = len(only_test)
        rows.append(row)
    return pd.DataFrame(rows)


def target_rate_by_level(train: pd.DataFrame) -> pd.DataFrame:
    y = train[config.TARGET_COL]
    rows = []
    for col in train.columns:
        if col in (config.ID_COL, config.TARGET_COL):
            continue
        if train[col].nunique(dropna=True) > MAX_LEVELS:
            continue
        grp = train.groupby(col, dropna=False, observed=True)[config.TARGET_COL].agg(["count", "mean"])
        for level, r in grp.iterrows():
            rows.append(
                {
                    "column": col,
                    "level": level,
                    "n": int(r["count"]),
                    "share": round(r["count"] / len(train), 4),
                    "target_rate": round(float(r["mean"]), 6),
                    "lift_vs_base": round(float(r["mean"]) / y.mean(), 4),
                }
            )
    return pd.DataFrame(rows)


def univariate_auc(train: pd.DataFrame) -> pd.DataFrame:
    """Single-feature ROC AUC. Categoricals are scored by their target rate.

    This uses the target on the full training set, so it is a descriptive
    ranking only — never feed these encodings into a model without folding
    them inside cross-validation.
    """
    y = train[config.TARGET_COL]
    rows = []
    for col in train.columns:
        if col in (config.ID_COL, config.TARGET_COL):
            continue
        s = train[col]
        if pd.api.types.is_numeric_dtype(s):
            score = s.fillna(s.median())
        else:
            score = s.map(train.groupby(col, observed=True)[config.TARGET_COL].mean())
            score = score.fillna(y.mean())
        auc = roc_auc_score(y, score)
        rows.append(
            {
                "column": col,
                "auc": round(float(auc), 5),
                "abs_auc_minus_half": round(abs(float(auc) - 0.5), 5),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_auc_minus_half", ascending=False)


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    # The shipped target is "Yes"/"No"; everything below needs it as 0/1.
    train[config.TARGET_COL] = coerce_target(train[config.TARGET_COL])

    summary = column_summary(train, test)
    rates = target_rate_by_level(train)
    aucs = univariate_auc(train)

    summary.to_csv(EDA_DIR / "column_summary.csv", index=False)
    rates.to_csv(EDA_DIR / "target_rate_by_level.csv", index=False)
    aucs.to_csv(EDA_DIR / "univariate_auc.csv", index=False)

    y = train[config.TARGET_COL]
    dup_feature_cols = [c for c in train.columns if c not in (config.ID_COL, config.TARGET_COL)]
    n_dup = int(train.duplicated(subset=dup_feature_cols).sum())

    buf = io.StringIO()
    w = buf.write
    w("# S6E11 data profile\n\n")
    w("Generated by `python -m scripts.profile_data`. Regenerate rather than edit.\n\n")
    w("## Shape\n\n")
    w(f"- train: {train.shape[0]:,} rows x {train.shape[1]} columns\n")
    w(f"- test: {test.shape[0]:,} rows x {test.shape[1]} columns\n")
    w(f"- test / train row ratio: {len(test) / len(train):.3f}\n")
    w(f"- duplicate feature rows in train: {n_dup:,} ({100 * n_dup / len(train):.3f}%)\n\n")
    w("## Target\n\n")
    w(f"- positives: {int(y.sum()):,} of {len(y):,}\n")
    w(f"- positive rate: {y.mean():.5f} (1 in {1 / y.mean():.1f})\n")
    w(f"- implied majority-class accuracy: {max(y.mean(), 1 - y.mean()):.5f} "
      "— accuracy is useless here, which is why the metric is ROC AUC\n\n")
    w("## Columns\n\n")
    w("| column | kind | unique | missing % | notes |\n|---|---|---|---|---|\n")
    for _, r in summary.iterrows():
        if r["column"] in (config.ID_COL, config.TARGET_COL):
            continue
        if r["kind"] == "numeric":
            note = f"range {_fmt(r.get('min'), 2)} – {_fmt(r.get('max'), 2)}, median {_fmt(r.get('median'), 2)}"
        else:
            note = f"top level {r.get('top_level')!r} ({100 * (r.get('top_level_share') or 0):.1f}%)"
            if r.get("levels_only_in_test"):
                note += f", {int(r['levels_only_in_test'])} level(s) only in test"
        w(f"| {r['column']} | {r['kind']} | {r['n_unique']} | {r['missing_train_pct']:.3f} | {note} |\n")
    w("\n## Univariate signal\n\n")
    w("Single-feature ROC AUC. Descriptive only — categorical encodings here use "
      "the full-train target rate and would leak if used as model features.\n\n")
    w("| column | AUC | distance from 0.5 |\n|---|---|---|\n")
    for _, r in aucs.iterrows():
        w(f"| {r['column']} | {r['auc']:.5f} | {r['abs_auc_minus_half']:.5f} |\n")
    w("\n## Strongest level effects\n\n")
    if not rates.empty:
        top = rates.reindex(rates["lift_vs_base"].sub(1).abs().sort_values(ascending=False).index).head(15)
        w("| column | level | n | target rate | lift vs base |\n|---|---|---|---|---|\n")
        for _, r in top.iterrows():
            w(f"| {r['column']} | {r['level']} | {r['n']:,} | {r['target_rate']:.5f} | {r['lift_vs_base']:.3f}x |\n")
    w("\nFull breakdown in `target_rate_by_level.csv`.\n")

    (EDA_DIR / "profile.md").write_text(buf.getvalue())
    print(f"wrote {EDA_DIR.name}/profile.md, column_summary.csv, "
          "target_rate_by_level.csv, univariate_auc.csv")


if __name__ == "__main__":
    main()
