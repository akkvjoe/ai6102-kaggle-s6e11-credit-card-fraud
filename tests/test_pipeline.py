"""End-to-end smoke test on synthetic data. Run: python -m tests.test_pipeline"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from scripts.make_synthetic import make
from src import config
from src.cv import run_cv
from src.data import split_xy
from src.submit import write_submission


def main() -> None:
    train = make(3000, 0, with_target=True)
    test = make(1200, 3000, with_target=False)

    X, y, X_test, test_ids = split_xy(train, test)
    assert config.TARGET_COL not in X.columns, "target leaked into features"
    assert config.ID_COL not in X.columns, "id leaked into features"
    assert list(X.columns) == list(X_test.columns), "train/test feature mismatch"

    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    assert cat_cols, "expected some categorical columns"
    for col in cat_cols:
        assert list(X[col].cat.categories) == list(X_test[col].cat.categories), (
            f"category mismatch in {col}"
        )

    result = run_cv(X, y, X_test, n_folds=3, verbose=False)
    assert len(result.fold_scores) == 3
    assert all(0.5 < s <= 1.0 for s in result.fold_scores), result.fold_scores
    assert result.test_pred.shape == (len(test),)
    assert ((result.test_pred >= 0) & (result.test_pred <= 1)).all()

    with tempfile.TemporaryDirectory() as tmp:
        original = config.SUBMISSION_DIR
        config.SUBMISSION_DIR = Path(tmp)
        try:
            path = write_submission(test_ids, result.test_pred, tag="smoke")
            sub = pd.read_csv(path)
        finally:
            config.SUBMISSION_DIR = original

    assert list(sub.columns) == [config.ID_COL, config.TARGET_COL], sub.columns
    assert len(sub) == len(test)
    assert sub[config.ID_COL].tolist() == test_ids.tolist()

    print(f"OK  fold AUCs {[round(s, 4) for s in result.fold_scores]}")
    print(f"OK  submission {sub.shape} columns {list(sub.columns)}")


if __name__ == "__main__":
    main()
