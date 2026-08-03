"""
feature_engineering.py
------------------------
Creates the engineered features developed in Milestone 2 (Feature Engineering
& EDA), refactored into a single reusable function so the exact same
transformations are applied at both training time and inference time.
"""

import numpy as np
import pandas as pd

SATISFACTION_COLUMNS = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe. Safe to call on a single-row
    inference dataframe or a full training dataframe."""
    df = df.copy()

    # Avoid division by zero for employees with 0 total working years
    safe_total_years = df["TotalWorkingYears"].replace(0, 0.5)
    safe_years_at_company = df["YearsAtCompany"].replace(0, 0.5)

    # 1. Tenure ratio: share of career spent at this company
    df["TenureRatio"] = df["YearsAtCompany"] / safe_total_years

    # 2. Promotion gap: years at company vs. years since last promotion
    df["PromotionGap"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]

    # 3. Average satisfaction score across satisfaction-related columns
    available_sat_cols = [c for c in SATISFACTION_COLUMNS if c in df.columns]
    df["AvgSatisfactionScore"] = df[available_sat_cols].mean(axis=1)

    # 4. Manager stability ratio (only if the column exists — optional field)
    if "YearsWithCurrManager" in df.columns:
        df["ManagerStabilityRatio"] = df["YearsWithCurrManager"] / safe_years_at_company

    # 5. Frequent job-hopper flag
    df["IsFrequentJobHopper"] = (df["NumCompaniesWorked"] >= 4).astype(int)

    # 6. Log-transform skewed income to reduce right-skew for the model
    df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])

    return df
