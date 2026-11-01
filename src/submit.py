"""Write a submission file in the format the competition requires."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def write_submission(test_ids: pd.Series, preds: np.ndarray, tag: str = "baseline") -> Path:
    if len(test_ids) != len(preds):
        raise ValueError(f"{len(test_ids)} ids but {len(preds)} predictions")
    if not np.all((preds >= 0) & (preds <= 1)):
        raise ValueError("predictions must be probabilities in [0, 1]")

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = config.SUBMISSION_DIR / f"{tag}_{stamp}.csv"

    pd.DataFrame({config.ID_COL: test_ids.values, config.TARGET_COL: preds}).to_csv(
        path, index=False
    )
    return path
