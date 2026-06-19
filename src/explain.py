"""SHAP-based model interpretability for the load forecaster."""

from __future__ import annotations

import pandas as pd


def global_importance(model, x: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """Return mean absolute SHAP value per feature (global importance ranking).

    Sampled to ``max_rows`` for speed; the ranking is stable well below the
    full dataset size.
    """
    import shap

    sample = x.sample(min(len(x), max_rows), random_state=0)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    importance = pd.DataFrame(
        {"feature": sample.columns, "mean_abs_shap": abs(values).mean(axis=0)}
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
