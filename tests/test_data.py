import pandas as pd

from src import config


def test_schema(raw):
    assert {"timestamp", config.TARGET, "temp_c"} <= set(raw.columns)
    assert len(raw) > 24 * 60


def test_hourly_and_sorted(raw):
    ts = pd.DatetimeIndex(raw["timestamp"])
    assert ts.is_monotonic_increasing
    assert (ts.to_series().diff().dropna() == pd.Timedelta(hours=1)).mean() > 0.99


def test_load_positive(raw):
    assert (raw[config.TARGET] > 0).all()
