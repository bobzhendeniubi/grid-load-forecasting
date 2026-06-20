"""Load hourly electricity-demand + temperature for a grid balancing authority.

Real path: EIA v2 hourly demand (CAISO) joined to Open-Meteo temperature.
Offline path: a temperature-driven synthetic series whose load responds
non-linearly to heating/cooling demand — so weather-aware models earn a real,
non-trivial lift over a seasonal-naive baseline (as they do on live grid data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, weather


def _temp_response(temp: np.ndarray) -> np.ndarray:
    """Non-linear weather load: cooling above 20°C, heating below 12°C."""
    cooling = np.clip(temp - 20.0, 0, None) ** 1.6 * 140.0
    heating = np.clip(12.0 - temp, 0, None) ** 1.5 * 110.0
    return cooling + heating


def _fetch_eia() -> pd.DataFrame:
    """Fetch hourly demand, paginating past the EIA 5,000-row-per-request cap."""
    import requests

    page_size = 5000
    target_rows = config.HISTORY_DAYS * 24
    collected: list[dict] = []
    offset = 0
    while len(collected) < target_rows:
        params = {
            "api_key": config.EIA_API_KEY,
            "frequency": "hourly",
            "data[]": "value",
            "facets[respondent][]": config.EIA_RESPONDENT,
            "facets[type][]": "D",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": offset,
            "length": page_size,
        }
        resp = requests.get(config.EIA_BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()["response"]["data"]
        if not rows:
            break
        collected.extend(rows)
        offset += page_size

    if not collected:
        raise ValueError("EIA returned no rows; check respondent code / API key.")
    df = pd.DataFrame(collected).rename(columns={"period": "timestamp", "value": config.TARGET})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")
    return df[["timestamp", config.TARGET]].dropna().sort_values("timestamp")


def _synthetic() -> pd.DataFrame:
    """Temperature-driven synthetic load with weekly/daily human patterns."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    periods = config.HISTORY_DAYS * 24
    idx = pd.date_range(end=pd.Timestamp("2026-01-01"), periods=periods, freq="h")
    temp = weather.synthetic_temperature(idx)

    hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    base = 18_000.0
    daily = 3_500 * np.sin((hour - 4) / 24 * 2 * np.pi)
    evening = 2_500 * np.exp(-((hour - 19) ** 2) / 8)
    weekly = np.where(dow >= 5, -2_000, 0)
    load = base + daily + evening + weekly + _temp_response(temp)
    load += rng.normal(0, 450, periods)

    for pos in rng.choice(periods, size=10, replace=False):  # outages / spikes
        load[pos] += rng.choice([-1, 1]) * rng.uniform(5_000, 9_000)

    return pd.DataFrame({"timestamp": idx, config.TARGET: load.round(1),
                         "temp_c": temp.round(2)})


def load_demand(use_cache: bool = True) -> pd.DataFrame:
    """Return hourly frame: ``timestamp``, ``load_mw``, ``temp_c``.

    Resolution: cached CSV -> EIA + Open-Meteo (if key) -> synthetic fallback.
    """
    config.ensure_dirs()
    if use_cache and config.RAW_LOAD_CSV.exists():
        return pd.read_csv(config.RAW_LOAD_CSV, parse_dates=["timestamp"]) \
            .sort_values("timestamp").reset_index(drop=True)

    if config.EIA_API_KEY:
        try:
            df = _fetch_eia()
            df["temp_c"] = weather.fetch_temperature(pd.DatetimeIndex(df["timestamp"]))
        except Exception as exc:  # noqa: BLE001
            print(f"[data] EIA fetch failed ({exc}); using synthetic series.")
            df = _synthetic()
    else:
        print("[data] No EIA_API_KEY set; using synthetic series. "
              "Set EIA_API_KEY for real CAISO data.")
        df = _synthetic()

    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(config.RAW_LOAD_CSV, index=False)
    return df
