"""
predict_pipeline.py
---------------------
The end-to-end inference pipeline: takes raw employee data (a dict for a
single employee, or a dataframe for a batch), applies the exact same
preprocessing and feature engineering used during training, and returns a
resignation-risk prediction. Phase 2 adds an explainability layer
(get_top_factors) so the application can show *why* a prediction was made.
"""

import pandas as pd

from src.data_preprocessing import preprocess
from src.feature_engineering import engineer_features
from src.utils import risk_tier, get_top_factors


def _prepare_features(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = engineer_features(df_raw)
    df, _ = preprocess(df, encoders=artifacts["encoders"], reference_columns=None)
    df = df.reindex(columns=artifacts["feature_columns"], fill_value=0)
    return df


def predict_single(employee: dict, artifacts: dict, explain: bool = True) -> dict:
    """Run the full pipeline on a single employee record and (optionally)
    explain the prediction with the top contributing factors."""
    df_raw = pd.DataFrame([employee])
    df = _prepare_features(df_raw, artifacts)

    X_scaled = artifacts["scaler"].transform(df)
    probability = float(artifacts["model"].predict_proba(X_scaled)[0][1])
    prediction = int(probability >= 0.5)

    result = {
        "probability": round(probability, 4),
        "tier": risk_tier(probability),
        "prediction": "Yes (likely to resign)" if prediction == 1 else "No (likely to stay)",
    }

    if explain:
        result["top_factors"] = get_top_factors(
            artifacts["model"], X_scaled[0], artifacts["feature_columns"], top_n=5
        )

    return result


def predict_batch(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Run the full pipeline on a batch of employee records (a dataframe)."""
    df = _prepare_features(df_raw, artifacts)

    X_scaled = artifacts["scaler"].transform(df)
    probabilities = artifacts["model"].predict_proba(X_scaled)[:, 1]

    result = df_raw.copy()
    result["ResignationProbability"] = probabilities.round(4)
    result["RiskTier"] = [risk_tier(p) for p in probabilities]
    return result
