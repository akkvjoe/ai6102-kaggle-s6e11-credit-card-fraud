"""Model registry.

Every experiment goes through the same CV harness, so models must share one
interface: a name maps to an estimator that accepts the raw feature frame
(numeric columns as-is, categoricals as pandas ``category``).

The registry is deliberately ordered as an argument:
  linear_main   - control; structurally cannot represent interactions
  linear_inter  - the same model plus all pairwise interactions
  tree_shallow  - legible; should recover the subsidy gate as a split
  nb_binned     - Naive Bayes; its independence assumption is what the data breaks
  hgb           - scikit-learn gradient boosting, the working baseline
  lgbm          - LightGBM, the performance model
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from . import config


def _split_cols(X):
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def _linear(interactions: bool):
    """Logistic regression on scaled numerics + one-hot categoricals."""

    def build(X):
        num, cat = _split_cols(X)
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
            ]
        )
        steps = [("pre", pre)]
        if interactions:
            # All pairwise products. The point is to give the linear model the
            # capacity it lacks by default, not to hand-pick the winning cross.
            steps.append(
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
            )
        steps.append(
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1))
        )
        return Pipeline(steps)

    return build


def _tree(max_depth: int):
    def build(X):
        num, cat = _split_cols(X)
        pre = ColumnTransformer(
            [
                ("num", "passthrough", num),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
            ]
        )
        return Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_leaf=200,
                        random_state=config.SEED,
                    ),
                ),
            ]
        )

    return build


def _naive_bayes(n_bins: int = 10):
    """CategoricalNB over quantile-binned numerics + ordinal-coded categoricals.

    Discretising everything lets one NB variant cover the whole frame, which is
    cleaner than mixing Gaussian and Categorical likelihoods across columns.
    """

    def build(X):
        num, cat = _split_cols(X)
        pre = ColumnTransformer(
            [
                (
                    "num",
                    KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                                     subsample=200_000, random_state=config.SEED),
                    num,
                ),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
            ]
        )
        return Pipeline([("pre", pre), ("clf", CategoricalNB(alpha=1.0, min_categories=n_bins))])

    return build


def _hgb(**overrides):
    def build(X):
        params = dict(
            learning_rate=0.06,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            categorical_features="from_dtype",
            random_state=config.SEED,
        )
        params.update(overrides)
        return HistGradientBoostingClassifier(**params)

    return build


def _lgbm(**overrides):
    def build(X):
        from lightgbm import LGBMClassifier

        params = dict(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=60,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=config.SEED,
            n_jobs=-1,
            verbose=-1,
        )
        params.update(overrides)
        return LGBMClassifier(**params)

    return build


REGISTRY = {
    "linear_main": _linear(interactions=False),
    "linear_inter": _linear(interactions=True),
    "tree_d4": _tree(max_depth=4),
    "tree_d8": _tree(max_depth=8),
    "nb_binned": _naive_bayes(),
    "hgb": _hgb(),
    "lgbm": _lgbm(),
}


def build_model(name: str, X):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](X)
