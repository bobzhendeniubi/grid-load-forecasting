"""Rolling-origin backtesting comparing three forecasters with error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, models
from .features import feature_columns


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (actual != 0)
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def rolling_backtest(feat: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward eval: seasonal-naive vs Ridge vs XGBoost on each fold."""
    cols = feature_columns(feat)
    n = len(feat)
    fold = config.TEST_DAYS * 24
    results = []

    for k in range(config.N_BACKTEST_FOLDS, 0, -1):
        test_end = n - (k - 1) * fold
        test_start = test_end - fold
        if test_start <= fold:
            continue
        train, test = feat.iloc[:test_start], feat.iloc[test_start:test_end]
        actual = test[config.TARGET].to_numpy()

        xgb = models.train_xgb(train[cols], train[config.TARGET])
        lin = models.train_linear(train[cols], train[config.TARGET])
        base = models.seasonal_naive(feat).take(test.index.to_numpy())

        results.append({
            "fold": config.N_BACKTEST_FOLDS - k + 1,
            "baseline_mape": mape(actual, base),
            "linear_mape": mape(actual, lin.predict(test[cols])),
            "xgb_mape": mape(actual, xgb.predict(test[cols])),
            "xgb_mae": mae(actual, xgb.predict(test[cols])),
        })

    out = pd.DataFrame(results)
    out["improvement_pct"] = (
        (out["baseline_mape"] - out["xgb_mape"]) / out["baseline_mape"] * 100
    )
    return out
