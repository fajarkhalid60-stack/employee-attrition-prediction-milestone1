"""
feature_engineering.py
------------------------
Creates the engineered features developed in Milestone 2, refactored into a
single reusable function so the exact same transformations are applied at
both training time and inference time.
"""

import numpy as np
import pandas as pd

SATISFACTION_COLUMNS = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe. Safe to call on a single-row
    inference dataframe or a full training dataframe."""
    df = df.copy()

    safe_total_years = df["TotalWorkingYears"].replace(0, 0.5)
    safe_years_at_company = df["YearsAtCompany"].replace(0, 0.5)

    df["TenureRatio"] = df["YearsAtCompany"] / safe_total_years
    df["PromotionGap"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]

    available_sat_cols = [c for c in SATISFACTION_COLUMNS if c in df.columns]
    df["AvgSatisfactionScore"] = df[available_sat_cols].mean(axis=1)

    if "YearsWithCurrManager" in df.columns:
        df["ManagerStabilityRatio"] = df["YearsWithCurrManager"] / safe_years_at_company

    df["IsFrequentJobHopper"] = (df["NumCompaniesWorked"] >= 4).astype(int)

    # Log-transform skewed income, then drop the raw column — keeping both
    # would make the two features fight each other in the model (multicollinearity),
    # which produces confusing, contradictory "top factor" explanations at inference time.
    df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])
    df = df.drop(columns=["MonthlyIncome"])

    return df
