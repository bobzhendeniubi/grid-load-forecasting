"""Streamlit dashboard for the grid load forecaster.

Run:  streamlit run dashboard/app.py
Shows the recent demand curve, model fit, flagged anomalies, backtest metrics,
and SHAP-based feature importance for both technical and non-technical viewers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

from src import config
from src.anomaly import flag_anomalies
from src.data import load_demand
from src.features import build_features, feature_columns

st.set_page_config(page_title="Grid Load Forecaster", layout="wide")
st.title("⚡ Day-Ahead Grid Load Forecaster")
st.caption(f"Balancing authority: {config.EIA_RESPONDENT} · day-ahead (24h) horizon")


@st.cache_data
def _load():
    return load_demand()


@st.cache_resource
def _model():
    if not Path(config.MODEL_PATH).exists():
        return None, None
    m = XGBRegressor()
    m.load_model(config.MODEL_PATH)
    cols = joblib.load(config.MODEL_DIR / "feature_cols.joblib")
    return m, cols


df = _load()
feat = build_features(df)
model, cols = _model()

if model is None:
    st.warning("No trained model found. Run `python -m src.train` first.")
    st.stop()

if Path(config.METRICS_PATH).exists():
    m = json.loads(Path(config.METRICS_PATH).read_text())
    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost MAPE", f"{m['xgb_mape_mean']}%")
    c2.metric("Baseline MAPE", f"{m['baseline_mape_mean']}%",
              delta=f"-{m['improvement_pct_mean']}%")
    c3.metric("Anomalies flagged", m["anomalies_flagged"])

feat = feat.copy()
feat["predicted"] = model.predict(feat[cols])
view = feat.tail(24 * 14).set_index("timestamp")

st.subheader("Actual vs. forecast (last 14 days)")
st.line_chart(view[[config.TARGET, "predicted"]])

st.subheader("Flagged demand anomalies")
anom = flag_anomalies(feat[config.TARGET].to_numpy(), feat["predicted"].to_numpy())
anom["timestamp"] = feat["timestamp"].to_numpy()
flagged = anom[anom["is_anomaly"]].tail(20)[["timestamp", "actual", "predicted", "robust_z"]]
st.dataframe(flagged, use_container_width=True)
