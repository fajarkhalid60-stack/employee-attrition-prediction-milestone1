"""
model_loader.py
-----------------
Loads the trained model and its associated preprocessing artifacts
(scaler, label encoders, and the exact training-time column order) that
were serialized with Joblib during training, avoiding retraining on
every application run.
"""

import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts(model_dir: str = MODEL_DIR) -> dict:
    """Load the serialized model bundle (model, scaler, encoders, feature
    columns, and evaluation metrics)."""
    bundle_path = os.path.join(model_dir, "attrition_model_bundle.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Model bundle not found at {bundle_path}. Run `python train.py` first "
            "to train and save the model."
        )
    return joblib.load(bundle_path)
