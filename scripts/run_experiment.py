"""Run one registered model through cross-validation and record the result.

  python -m scripts.run_experiment --model linear_main
  python -m scripts.run_experiment --model lgbm --folds 5 --submit

Appends a row to results/experiments.csv so every model is comparable on the
same folds. --subsample trains on a stratified sample and records the row count,
so a compute-limited run is never silently compared against a full-data one.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.cv import run_cv
from src.data import load_raw, split_xy
from src.models import REGISTRY
from src.submit import write_submission

RESULTS = config.ROOT / "results" / "experiments.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--subsample", type=int, default=0, help="train rows; 0 = all")
    ap.add_argument("--submit", action="store_true", help="also write a submission file")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    train, test = load_raw()
    X, y, X_test, test_ids = split_xy(train, test)

    if args.subsample and args.subsample < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=args.subsample, stratify=y, random_state=args.seed
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    print(f"{args.model}: {len(X):,} rows, {args.folds} folds")
    t0 = time.time()
    result = run_cv(X, y, X_test, model_name=args.model, n_folds=args.folds, seed=args.seed)
    elapsed = time.time() - t0

    sub_path = ""
    if args.submit:
        sub_path = write_submission(test_ids, result.test_pred, tag=args.model).name

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "n_rows": len(X),
        "folds": args.folds,
        "seed": args.seed,
        "mean_fold_auc": round(result.oof_auc, 6),
        "std_fold_auc": round(float(pd.Series(result.fold_scores).std()), 6),
        "fold_aucs": ";".join(f"{s:.6f}" for s in result.fold_scores),
        "fit_seconds": round(elapsed, 1),
        "submission": sub_path,
        "note": args.note,
    }
    pd.DataFrame([row]).to_csv(
        RESULTS, mode="a", header=not RESULTS.exists(), index=False
    )
    print(f"  {elapsed:.1f}s -> recorded in results/experiments.csv")


if __name__ == "__main__":
    main()
