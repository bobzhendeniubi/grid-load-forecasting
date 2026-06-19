"""FastAPI service exposing the trained load forecaster.

Run:  uvicorn api.main:app --reload
Endpoints:
  GET /health        -> liveness + whether a model is loaded
  GET /metrics       -> last training metrics (MAPE, lift, anomalies)
  POST /forecast     -> day-ahead forecast from a recent load window
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBRegressor

from src import config
from src.features import build_features, feature_columns

app = FastAPI(title="Grid Load Forecaster", version="1.0.0")


class LoadPoint(BaseModel):
    timestamp: str
    load_mw: float


class ForecastRequest(BaseModel):
    history: list[LoadPoint]  # recent hourly load, >= 200 points recommended


def _load_model() -> XGBRegressor | None:
    if not Path(config.MODEL_PATH).exists():
        return None
    model = XGBRegressor()
    model.load_model(config.MODEL_PATH)
    return model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": Path(config.MODEL_PATH).exists()}


@app.get("/metrics")
def metrics() -> dict:
    if not Path(config.METRICS_PATH).exists():
        raise HTTPException(404, "No metrics yet; run `python -m src.train` first.")
    return json.loads(Path(config.METRICS_PATH).read_text())


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict:
    model = _load_model()
    if model is None:
        raise HTTPException(503, "Model not trained; run `python -m src.train`.")
    df = pd.DataFrame([p.model_dump() for p in req.history])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feat = build_features(df)
    if feat.empty:
        raise HTTPException(422, "Not enough history to build features (need ~200h).")
    cols = joblib.load(config.MODEL_DIR / "feature_cols.joblib")
    row = feat.iloc[[-1]][cols]
    pred = float(model.predict(row)[0])
    next_ts = feat["timestamp"].iloc[-1] + pd.Timedelta(hours=config.HORIZON_HOURS)
    return {"forecast_timestamp": next_ts.isoformat(), "forecast_mw": round(pred, 1)}
