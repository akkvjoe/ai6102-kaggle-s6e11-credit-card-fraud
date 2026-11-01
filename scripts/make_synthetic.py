"""Generate stand-in train/test CSVs so the pipeline can be exercised offline.

Run from the repository root: ``python -m scripts.make_synthetic``

This is NOT competition data. It exists only so the code path can be verified
before the real files are downloaded. Column names other than id and the target
are invented; the pipeline infers its features, so they do not need to match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

RNG = np.random.default_rng(0)


def make(n: int, start_id: int, with_target: bool) -> pd.DataFrame:
    income = RNG.lognormal(mean=10.8, sigma=0.45, size=n)
    age = RNG.integers(21, 75, size=n)
    commute = RNG.gamma(shape=2.0, scale=9.0, size=n)
    charging = RNG.choice(["home", "work", "public", "none"], size=n, p=[0.45, 0.2, 0.2, 0.15])
    region = RNG.choice(["north", "south", "east", "west", "central"], size=n)
    owns_home = RNG.choice(["yes", "no"], size=n, p=[0.6, 0.4])

    df = pd.DataFrame(
        {
            config.ID_COL: np.arange(start_id, start_id + n),
            "annual_income": income.round(2),
            "age": age,
            "commute_km_per_day": commute.round(2),
            "charging_access": charging,
            "region": region,
            "owns_home": owns_home,
        }
    )
    # Scatter some missing values — the real data may well have them.
    df.loc[RNG.random(n) < 0.03, "commute_km_per_day"] = np.nan

    if with_target:
        logit = (
            -4.2
            + 0.45 * (np.log(income) - 10.8)
            - 0.02 * (age - 45)
            + 0.015 * commute
            + np.where(charging == "home", 0.9, 0.0)
            + np.where(owns_home == "yes", 0.35, 0.0)
            + RNG.normal(0, 0.4, size=n)
        )
        p = 1 / (1 + np.exp(-logit))
        # Yes/No strings, matching how the competition ships the target.
        df[config.TARGET_COL] = np.where(RNG.binomial(1, p) == 1, "Yes", "No")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train = make(args.n_train, 0, with_target=True)
    test = make(args.n_test, args.n_train, with_target=False)

    train.to_csv(args.out / "train.csv", index=False)
    test.to_csv(args.out / "test.csv", index=False)
    print(f"synthetic train {train.shape} test {test.shape} -> {args.out}")
    print(f"positive rate {(train[config.TARGET_COL] == 'Yes').mean():.4f}")


if __name__ == "__main__":
    main()
