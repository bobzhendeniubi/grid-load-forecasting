"""Temperature data — the dominant driver of electricity demand.

Real source: Open-Meteo's free archive API (no key required). Synthetic
fallback generates a realistic hourly temperature series with daily + annual
cycles so the pipeline runs offline. Temperature feeds heating/cooling-degree
features, which is what lets the ML model beat a naive seasonal baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Default coordinates: Los Angeles basin (SCE service territory).
LATITUDE = float(__import__("os").environ.get("WEATHER_LAT", "34.05"))
LONGITUDE = float(__import__("os").environ.get("WEATHER_LON", "-118.24"))
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def synthetic_temperature(index: pd.DatetimeIndex) -> np.ndarray:
    """Realistic hourly temperature (°C): annual + daily cycles + noise."""
    rng = np.random.default_rng(config.RANDOM_SEED + 1)
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    annual = 12 * -np.cos((doy - 15) / 365 * 2 * np.pi)   # hot summer, cool winter
    daily = 6 * -np.cos((hour - 15) / 24 * 2 * np.pi)     # afternoon peak
    return 18 + annual + daily + rng.normal(0, 2.0, len(index))


def fetch_temperature(index: pd.DatetimeIndex) -> np.ndarray:
    """Fetch hourly temperature for the index range from Open-Meteo, else synth."""
    try:
        import requests

        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": index.min().date().isoformat(),
            "end_date": index.max().date().isoformat(),
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
        resp.raise_for_status()
        h = resp.json()["hourly"]
        s = pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]))
        return s.reindex(index).interpolate().to_numpy()
    except Exception as exc:  # noqa: BLE001
        print(f"[weather] Open-Meteo unavailable ({exc}); using synthetic temps.")
        return synthetic_temperature(index)
