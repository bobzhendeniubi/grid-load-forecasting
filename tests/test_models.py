from src import config
from src.backtest import mape, rolling_backtest
from src.anomaly import flag_anomalies
import numpy as np


def test_mape_zero_when_perfect():
    a = np.array([100.0, 200.0, 300.0])
    assert mape(a, a) == 0.0


def test_anomaly_flags_outliers():
    actual = np.full(200, 1000.0)
    pred = np.full(200, 1000.0)
    actual[50] = 9000.0  # injected spike
    out = flag_anomalies(actual, pred)
    assert bool(out.loc[50, "is_anomaly"]) is True
    assert out["is_anomaly"].sum() < 5


def test_xgb_beats_baseline(feat):
    bt = rolling_backtest(feat)
    assert (bt["xgb_mape"] < bt["baseline_mape"]).mean() >= 0.5
