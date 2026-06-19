"""Residual-based anomaly detection on the load series.

Flags hours where actual demand departs from the model forecast by more than
``ANOMALY_SIGMA`` robust standard deviations (MAD-scaled), the kind of signal a
grid operator uses to catch sensor faults, outages, or unmodeled demand spikes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def flag_anomalies(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Return per-hour residuals, robust z-scores, and a boolean anomaly flag."""
    residual = actual - predicted
    median = np.median(residual)
    mad = np.median(np.abs(residual - median)) or 1e-9
    robust_z = 0.6745 * (residual - median) / mad  # 0.6745 = MAD->sigma scaling
    return pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
            "robust_z": robust_z,
            "is_anomaly": np.abs(robust_z) > config.ANOMALY_SIGMA,
        }
    )
