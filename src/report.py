"""Generate publication-quality figures for the README and dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

FIG_DIR = config.ROOT / "reports" / "figures"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forecast(feat: pd.DataFrame, pred, lo=None, hi=None, hours: int = 24 * 10):
    """Actual vs forecast (last N hours) with optional P10–P90 band."""
    v = feat.tail(hours)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(v["timestamp"], v[config.TARGET], label="Actual", lw=1.4)
    ax.plot(v["timestamp"], pred[-hours:], label="XGBoost forecast", lw=1.4)
    if lo is not None and hi is not None:
        ax.fill_between(v["timestamp"], lo[-hours:], hi[-hours:], alpha=0.18,
                        label="P10–P90 band")
    ax.set_title("Day-ahead load forecast vs. actual")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right")
    return _save(fig, "forecast_vs_actual.png")


def plot_backtest(bt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bt["fold"], bt["baseline_mape"], "o-", label="Seasonal-naive")
    ax.plot(bt["fold"], bt["linear_mape"], "s-", label="Ridge")
    ax.plot(bt["fold"], bt["xgb_mape"], "^-", label="XGBoost")
    ax.set_xlabel("Backtest fold")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Rolling-origin backtest: MAPE by model")
    ax.legend()
    return _save(fig, "backtest_mape.png")


def plot_anomalies(anom: pd.DataFrame, hours: int = 24 * 30):
    v = anom.tail(hours)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(v["timestamp"], v["actual"], lw=1.0, label="Actual load")
    flagged = v[v["is_anomaly"]]
    ax.scatter(flagged["timestamp"], flagged["actual"], color="crimson", s=28,
               zorder=5, label="Anomaly")
    ax.set_title("Residual-based demand anomalies")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    return _save(fig, "anomalies.png")
