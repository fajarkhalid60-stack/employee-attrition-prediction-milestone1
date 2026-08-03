"""
predict_pipeline.py
---------------------
The end-to-end inference pipeline: takes a single raw employee record (as a
dict) or a small dataframe, applies the exact same preprocessing and feature
engineering used during training, and returns a resignation-risk prediction.

This is the module the application file (app.py) calls — it never touches
the raw model directly, so preprocessing logic only lives in one place.
"""

import pandas as pd

from src.data_preprocessing import preprocess
from src.feature_engineering import engineer_features
from src.utils import risk_tier


def predict_single(employee: dict, artifacts: dict) -> dict:
    """
    Run the full pipeline on a single employee record.

    Parameters
    ----------
    employee : dict
        Raw employee feature values, e.g. {"Age": 29, "MonthlyIncome": 4200, ...}
    artifacts : dict
        The loaded model bundle from model_loader.load_artifacts()

    Returns
    -------
    dict with keys: probability, tier, prediction
    """
    df = pd.DataFrame([employee])

    df = engineer_features(df)
    df, _ = preprocess(df, encoders=artifacts["encoders"], reference_columns=None)

    # Align columns exactly to the training-time feature order, filling any
    # missing one-hot columns with 0 (e.g. a category not present in this row).
    df = df.reindex(columns=artifacts["feature_columns"], fill_value=0)

    X_scaled = artifacts["scaler"].transform(df)
    probability = float(artifacts["model"].predict_proba(X_scaled)[0][1])
    prediction = int(probability >= 0.5)

    return {
        "probability": round(probability, 4),
        "tier": risk_tier(probability),
        "prediction": "Yes (likely to resign)" if prediction == 1 else "No (likely to stay)",
    }


def predict_batch(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Run the full pipeline on a batch of employee records (a dataframe)."""
    df = engineer_features(df_raw)
    df, _ = preprocess(df, encoders=artifacts["encoders"], reference_columns=None)
    df = df.reindex(columns=artifacts["feature_columns"], fill_value=0)

    X_scaled = artifacts["scaler"].transform(df)
    probabilities = artifacts["model"].predict_proba(X_scaled)[:, 1]

    result = df_raw.copy()
    result["ResignationProbability"] = probabilities.round(4)
    result["RiskTier"] = [risk_tier(p) for p in probabilities]
    return result
