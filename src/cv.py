"""Stratified cross-validation producing out-of-fold and test predictions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .models import build_model


@dataclass
class CVResult:
    oof: np.ndarray
    test_pred: np.ndarray
    fold_scores: list[float] = field(default_factory=list)

    @property
    def oof_auc(self) -> float:
        return float(np.mean(self.fold_scores)) if self.fold_scores else float("nan")


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    model_name: str = "hgb",
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
    verbose: bool = True,
) -> CVResult:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    result = CVResult(oof=oof, test_pred=test_pred)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = build_model(model_name, X)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = va_pred
        score = roc_auc_score(y.iloc[va_idx], va_pred)
        result.fold_scores.append(float(score))

        test_pred += model.predict_proba(X_test)[:, 1] / n_folds

        if verbose:
            print(f"  fold {fold}/{n_folds}  ROC AUC {score:.5f}")

    if verbose:
        overall = roc_auc_score(y, oof)
        print(f"  mean fold AUC {result.oof_auc:.5f}   pooled OOF AUC {overall:.5f}")

    return result
