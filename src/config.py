"""Central configuration for the grid load-forecasting pipeline.

All tunable constants live here so the rest of the code stays declarative.
Secrets (e.g. the EIA API key) are read from the environment, never hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RAW_LOAD_CSV = DATA_DIR / "load_hourly.csv"
MODEL_PATH = MODEL_DIR / "xgb_load_forecaster.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --- Data source -----------------------------------------------------------
# EIA balancing-authority code. CISO = California ISO (the SCE grid region).
EIA_RESPONDENT = os.environ.get("EIA_RESPONDENT", "CISO")
EIA_API_KEY = os.environ.get("EIA_API_KEY")  # optional; synthetic fallback if unset
EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
HISTORY_DAYS = int(os.environ.get("HISTORY_DAYS", "730"))  # ~2 years

# --- Modeling --------------------------------------------------------------
TARGET = "load_mw"
HORIZON_HOURS = 24            # day-ahead forecast
TEST_DAYS = 30               # hold-out window for the final backtest
N_BACKTEST_FOLDS = 6         # rolling-origin folds
RANDOM_SEED = 42

XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Residual-based anomaly threshold, in robust standard deviations (MAD-scaled).
ANOMALY_SIGMA = 3.5


def ensure_dirs() -> None:
    """Create the data/ and models/ directories if they do not yet exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
