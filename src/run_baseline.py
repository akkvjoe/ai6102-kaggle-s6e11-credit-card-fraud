"""Entry point: python -m src.run_baseline"""
from __future__ import annotations

import argparse

from . import config
from .cv import run_cv
from .data import load_raw, split_xy
from .submit import write_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="S6E11 gradient boosting baseline")
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()

    train, test = load_raw()
    print(f"train {train.shape}  test {test.shape}")

    X, y, X_test, test_ids = split_xy(train, test)
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    print(f"{X.shape[1]} features ({len(cat_cols)} categorical), positive rate {y.mean():.4f}")

    result = run_cv(X, y, X_test, n_folds=args.folds, seed=args.seed)
    path = write_submission(test_ids, result.test_pred, tag=args.tag)
    print(f"wrote {path.relative_to(config.ROOT)}  (mean fold AUC {result.oof_auc:.5f})")


if __name__ == "__main__":
    main()
