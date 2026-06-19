"""Feature engineering for hourly load forecasting.

Calendar (cyclical), leak-safe lags/rolling stats, and weather features
(temperature, heating/cooling degrees, temp lags). Lag/rolling features are
shifted by the forecast horizon so the model never sees future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_HOLIDAYS = {"01-01", "07-04", "12-25", "11-11"}
COMFORT_C = 18.0  # balance point for heating/cooling degrees


def _is_holiday(ts: pd.Series) -> np.ndarray:
    return ts.dt.strftime("%m-%d").isin(_HOLIDAYS).to_numpy().astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame (rows with complete lags only)."""
    out = df.sort_values("timestamp").copy()
    ts = out["timestamp"]

    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["is_holiday"] = _is_holiday(ts)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    out["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)

    # Weather features (temperature is known/forecast at prediction time).
    if "temp_c" in out.columns:
        out["temp_c"] = out["temp_c"].astype(float)
        out["cooling_deg"] = np.clip(out["temp_c"] - COMFORT_C, 0, None)
        out["heating_deg"] = np.clip(COMFORT_C - out["temp_c"], 0, None)
        out["temp_lag_24"] = out["temp_c"].shift(24)

    h = config.HORIZON_HOURS
    for lag in (h, h + 1, h + 24, h + 48, h + 168):
        out[f"lag_{lag}"] = out[config.TARGET].shift(lag)

    lagged = out[config.TARGET].shift(h)
    out["roll_mean_24"] = lagged.rolling(24).mean()
    out["roll_std_24"] = lagged.rolling(24).std()
    out["roll_mean_168"] = lagged.rolling(168).mean()

    return out.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"timestamp", config.TARGET}]
