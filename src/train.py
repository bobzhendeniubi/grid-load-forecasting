"""End-to-end training entrypoint.

Run:  python -m src.train
Loads data + weather, builds features, backtests three models, fits the final
XGBoost point + P10/P90 quantile models, flags anomalies, renders report
figures, and persists models + metrics.
"""

from __future__ import annotations

import json

import joblib

from . import anomaly, backtest, config, models, report
from .data import load_demand
from .features import build_features, feature_columns


def main() -> None:
    config.ensure_dirs()
    print("[train] loading demand + weather ...")
    feat = build_features(load_demand())
    cols = feature_columns(feat)
    print(f"[train] {len(feat):,} rows, {len(cols)} features")

    print("[train] rolling backtest (seasonal-naive vs Ridge vs XGBoost) ...")
    bt = backtest.rolling_backtest(feat)
    print(bt.round(2).to_string(index=False))

    print("[train] fitting final point + quantile models ...")
    model = models.train_xgb(feat[cols], feat[config.TARGET])
    bands = models.train_quantiles(feat[cols], feat[config.TARGET])
    model.save_model(config.MODEL_PATH)
    joblib.dump(cols, config.MODEL_DIR / "feature_cols.joblib")

    fitted = model.predict(feat[cols])
    lo, hi = bands[0.1].predict(feat[cols]), bands[0.9].predict(feat[cols])
    anomalies = anomaly.flag_anomalies(feat[config.TARGET].to_numpy(), fitted)
    anomalies["timestamp"] = feat["timestamp"].to_numpy()
    n_anom = int(anomalies["is_anomaly"].sum())

    print("[train] rendering report figures ...")
    report.plot_forecast(feat, fitted, lo, hi)
    report.plot_backtest(bt)
    report.plot_anomalies(anomalies)

    coverage = float(((feat[config.TARGET] >= lo) & (feat[config.TARGET] <= hi)).mean())
    metrics = {
        "rows": int(len(feat)),
        "features": int(len(cols)),
        "baseline_mape_mean": round(float(bt["baseline_mape"].mean()), 2),
        "linear_mape_mean": round(float(bt["linear_mape"].mean()), 2),
        "xgb_mape_mean": round(float(bt["xgb_mape"].mean()), 2),
        "xgb_mae_mean": round(float(bt["xgb_mae"].mean()), 1),
        "improvement_pct_mean": round(float(bt["improvement_pct"].mean()), 2),
        "p10_p90_coverage": round(coverage, 3),
        "anomalies_flagged": n_anom,
    }
    with open(config.METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[train] done. XGBoost {metrics['xgb_mape_mean']}% MAPE vs "
          f"baseline {metrics['baseline_mape_mean']}% "
          f"({metrics['improvement_pct_mean']}% better), "
          f"P10–P90 coverage {metrics['p10_p90_coverage']:.0%}, "
          f"{n_anom} anomalies. Figures in reports/figures/.")


if __name__ == "__main__":
    main()
