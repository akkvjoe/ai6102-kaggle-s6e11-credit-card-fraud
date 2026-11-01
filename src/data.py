"""Loading and light preparation.

The competition schema is gated behind accepting the rules on Kaggle, so this
module deliberately does not hard-code feature names. Columns are typed by
inspection: anything non-numeric becomes a pandas ``category``, which is what
HistGradientBoostingClassifier consumes natively.
"""
from __future__ import annotations

import pandas as pd

from . import config


TRUTHY = {"yes", "y", "true", "t", "1"}
FALSY = {"no", "n", "false", "f", "0"}


def coerce_target(s: pd.Series) -> pd.Series:
    """Return the target as 0/1 ints.

    S6E11 is expected to ship ``is_fraud`` as 0/1, but the sibling competition
    S6E9 shipped its target as the strings "Yes"/"No", so both forms are
    accepted. The submission expects a probability either way. Anything not
    recognisably binary raises rather than silently becoming NaN.
    """
    if pd.api.types.is_numeric_dtype(s):
        values = set(pd.unique(s.dropna()))
        if not values <= {0, 1}:
            raise ValueError(f"numeric target is not binary; found {sorted(values)[:10]}")
        return s.astype(int)

    norm = s.astype(str).str.strip().str.lower()
    mapped = norm.map(lambda v: 1 if v in TRUTHY else (0 if v in FALSY else None))
    if mapped.isna().any():
        bad = sorted(set(norm[mapped.isna()]))[:10]
        raise ValueError(f"could not read target labels as binary: {bad}")
    return mapped.astype(int)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train.csv and test.csv, with a clear error if they are absent."""
    missing = [p for p in (config.TRAIN_CSV, config.TEST_CSV) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing {names} in {config.DATA_DIR}. "
            "Download them from the competition Data tab after accepting the rules."
        )
    return pd.read_csv(config.TRAIN_CSV), pd.read_csv(config.TEST_CSV)


def align_categories(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast object columns to ``category`` using categories drawn from both frames.

    Using the union keeps the encoding identical across train and test. Only the
    feature columns are touched, so no target information crosses over.
    """
    train, test = train.copy(), test.copy()
    obj_cols = [c for c in test.columns if c != config.ID_COL and train[c].dtype == "object"]
    for col in obj_cols:
        cats = pd.Index(pd.concat([train[col], test[col]]).dropna().unique()).sort_values()
        dtype = pd.CategoricalDtype(categories=cats)
        train[col] = train[col].astype(dtype)
        test[col] = test[col].astype(dtype)
    return train, test


def split_xy(train: pd.DataFrame, test: pd.DataFrame):
    """Return X, y, X_test and the test ids, dropping id and target from features."""
    if config.TARGET_COL not in train.columns:
        raise KeyError(
            f"Expected target column {config.TARGET_COL!r}; found {list(train.columns)}"
        )
    train, test = align_categories(train, test)

    drop = [c for c in (config.ID_COL, config.TARGET_COL) if c in train.columns]
    X = train.drop(columns=drop)
    y = coerce_target(train[config.TARGET_COL])

    X_test = test[X.columns]
    test_ids = test[config.ID_COL]
    return X, y, X_test, test_ids
