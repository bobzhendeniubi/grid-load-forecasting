"""Forecasting models: seasonal-naive + linear baselines and XGBoost.

XGBoost is trained both as a point forecaster and as P10/P90 quantile models
to produce the uncertainty band grid operators rely on for reserve planning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config


def seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Same hour, one week ago — a strong, honest baseline for hourly load."""
    return df[config.TARGET].shift(168).to_numpy()


def train_linear(x: pd.DataFrame, y: pd.Series):
    """Standardized Ridge regression — a transparent ML baseline."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x, y)
    return model


def train_xgb(x: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(x, y)
    return model


def train_quantiles(x: pd.DataFrame, y: pd.Series) -> dict:
    """Train P10/P90 XGBoost quantile models for a day-ahead uncertainty band."""
    bands = {}
    for q in (0.1, 0.9):
        params = {**config.XGB_PARAMS, "objective": "reg:quantileerror",
                  "quantile_alpha": q}
        m = XGBRegressor(**params)
        m.fit(x, y)
        bands[q] = m
    return bands
