"""
data_preprocessing.py
----------------------
Handles loading, cleaning, and encoding of the HR Attrition dataset.
Mirrors the preprocessing steps performed in Milestone 1 (Dataset Selection
& Preprocessing), refactored into reusable functions instead of notebook cells.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

CONSTANT_COLUMNS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
BINARY_COLUMNS = ["Attrition", "Gender", "OverTime"]
MULTICLASS_COLUMNS = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]

NUMERIC_COLUMNS = [
    "Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw HR Attrition CSV from disk."""
    return pd.read_csv(path)


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier/constant columns that carry no predictive signal."""
    cols_present = [c for c in CONSTANT_COLUMNS if c in df.columns]
    return df.drop(columns=cols_present)


def check_data_quality(df: pd.DataFrame) -> dict:
    """Report missing values and duplicate rows (as verified in Milestone 1)."""
    return {
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "shape": df.shape,
    }


def encode_binary_columns(df: pd.DataFrame, encoders: dict = None):
    """Label-encode binary categorical columns (Attrition, Gender, OverTime)."""
    df = df.copy()
    encoders = encoders or {}
    for col in BINARY_COLUMNS:
        if col not in df.columns:
            continue
        if col in encoders:
            le = encoders[col]
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders


def encode_multiclass_columns(df: pd.DataFrame, reference_columns: list = None) -> pd.DataFrame:
    """One-hot encode multi-class categorical columns, aligned to reference_columns if given."""
    cols_present = [c for c in MULTICLASS_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)
    if reference_columns is not None:
        df = df.reindex(columns=reference_columns, fill_value=0)
    return df


def preprocess(df: pd.DataFrame, encoders: dict = None, reference_columns: list = None):
    """Full preprocessing pipeline used identically at train and inference time."""
    df = drop_constant_columns(df)
    df, encoders = encode_binary_columns(df, encoders)
    df = encode_multiclass_columns(df, reference_columns)
    return df, encoders
