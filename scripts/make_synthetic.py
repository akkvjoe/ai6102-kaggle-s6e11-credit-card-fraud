"""Generate stand-in train/test CSVs so the pipeline can be exercised offline.

Run from the repository root: ``python -m scripts.make_synthetic``

This is NOT competition data, and scores measured on it mean nothing. It exists
only so the code path can be verified before the real files are downloaded.
Column names other than id and the target are invented; the pipeline infers its
features, so they do not need to match the real schema.

One property is deliberately imitated rather than invented: the **class
imbalance**. A fraud target with a positive rate near 0.5% exercises the paths
that a balanced stand-in would leave untested — stratified folds with very few
positives per fold, and ROC AUC computed against a tiny positive class. The
real rate is unknown until the data is downloaded; ``--positive-rate`` overrides
the guess.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

RNG = np.random.default_rng(0)

DEFAULT_POSITIVE_RATE = 0.005


def make(n: int, start_id: int, with_target: bool, positive_rate: float) -> pd.DataFrame:
    amount = RNG.lognormal(mean=3.4, sigma=1.1, size=n)
    hour = RNG.integers(0, 24, size=n)
    age = RNG.integers(18, 85, size=n)
    distance_km = RNG.gamma(shape=1.6, scale=22.0, size=n)
    category = RNG.choice(
        ["grocery", "travel", "online", "fuel", "restaurant", "electronics"],
        size=n,
        p=[0.26, 0.09, 0.28, 0.14, 0.17, 0.06],
    )
    channel = RNG.choice(["chip", "swipe", "online", "contactless"], size=n, p=[0.38, 0.12, 0.3, 0.2])
    card_type = RNG.choice(["standard", "gold", "platinum"], size=n, p=[0.7, 0.22, 0.08])

    df = pd.DataFrame(
        {
            config.ID_COL: np.arange(start_id, start_id + n),
            "amount": amount.round(2),
            "hour_of_day": hour,
            "cardholder_age": age,
            "distance_from_home_km": distance_km.round(2),
            "merchant_category": category,
            "entry_channel": channel,
            "card_type": card_type,
        }
    )
    # Scatter some missing values — the real data may well have them.
    df.loc[RNG.random(n) < 0.03, "distance_from_home_km"] = np.nan

    if with_target:
        # Signal, then an intercept solved for numerically so the realised
        # positive rate lands on positive_rate rather than wherever the
        # coefficients happen to put it.
        signal = (
            0.55 * (np.log(amount) - 3.4)
            + 0.9 * np.isin(channel, ["online", "swipe"])
            + 0.012 * np.nan_to_num(distance_km, nan=float(np.nanmean(distance_km)))
            + 0.7 * np.isin(category, ["online", "electronics"])
            + 0.4 * ((hour < 5) | (hour >= 23))
            + RNG.normal(0, 0.5, size=n)
        )
        lo, hi = -30.0, 30.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(signal + mid)))).mean() < positive_rate:
                lo = mid
            else:
                hi = mid
        p = 1 / (1 + np.exp(-(signal + (lo + hi) / 2)))
        df[config.TARGET_COL] = RNG.binomial(1, p).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=8000)
    parser.add_argument("--positive-rate", type=float, default=DEFAULT_POSITIVE_RATE)
    parser.add_argument("--out", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train = make(args.n_train, 0, True, args.positive_rate)
    test = make(args.n_test, args.n_train, False, args.positive_rate)

    train.to_csv(args.out / "train.csv", index=False)
    test.to_csv(args.out / "test.csv", index=False)
    print(f"synthetic train {train.shape} test {test.shape} -> {args.out}")
    rate = train[config.TARGET_COL].mean()
    print(f"positive rate {rate:.4f}  ({int(train[config.TARGET_COL].sum())} positives)")
    print("NOT competition data — any score measured on this is meaningless.")


if __name__ == "__main__":
    main()
