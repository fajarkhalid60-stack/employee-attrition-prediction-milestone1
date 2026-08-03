"""
model_loader.py
-----------------
Loads the trained model and its associated preprocessing artifacts
(scaler, label encoders, and the exact training-time column order) that
were serialized with Joblib during training. This avoids retraining the
model every time the application starts.
"""

import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts(model_dir: str = MODEL_DIR) -> dict:
    """
    Load the serialized model bundle. Returns a dict with:
      - model: the trained scikit-learn classifier
      - scaler: the fitted StandardScaler
      - encoders: dict of fitted LabelEncoders for binary columns
      - feature_columns: exact column order expected by the model
      - metrics: evaluation metrics recorded at training time
    """
    bundle_path = os.path.join(model_dir, "attrition_model_bundle.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Model bundle not found at {bundle_path}. Run `python train.py` first "
            "to train and save the model."
        )
    return joblib.load(bundle_path)
