"""
utils.py
---------
Small helper functions shared across the preprocessing, training, and
prediction scripts.
"""

RISK_THRESHOLDS = {"low": 0.35, "high": 0.60}


def risk_tier(probability: float) -> str:
    """Convert a raw resignation probability into a Low/Medium/High tier."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if probability < RISK_THRESHOLDS["high"]:
        return "Medium"
    return "High"


def get_top_factors(model, scaled_row, feature_columns, top_n=5):
    """
    Explain a single prediction (Phase 2) by decomposing it into each
    feature's contribution to the logistic regression's log-odds score
    (coefficient * scaled feature value). Returns the top_n features with
    the largest absolute contribution, tagged as increasing or decreasing risk.
    """
    contributions = model.coef_[0] * scaled_row
    order = list(contributions.argsort()[::-1])
    ranked = sorted(order, key=lambda i: -abs(contributions[i]))[:top_n]
    factors = []
    for i in ranked:
        factors.append({
            "feature": feature_columns[i],
            "contribution": round(float(contributions[i]), 4),
            "direction": "increases risk" if contributions[i] > 0 else "decreases risk",
        })
    return factors


def validate_employee_input(data: dict) -> list:
    """Basic validation for a single employee record submitted through the app."""
    errors = []
    required_fields = [
        "Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears",
        "YearsAtCompany", "YearsSinceLastPromotion", "NumCompaniesWorked",
        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
        "StockOptionLevel", "JobLevel", "OverTime",
    ]
    for field in required_fields:
        if field not in data or data[field] in (None, ""):
            errors.append(f"{field} is required.")

    if "Age" in data and data["Age"] not in (None, "") and not (18 <= float(data["Age"]) <= 65):
        errors.append("Age must be between 18 and 65.")

    if "MonthlyIncome" in data and data["MonthlyIncome"] not in (None, "") and float(data["MonthlyIncome"]) <= 0:
        errors.append("Monthly Income must be greater than 0.")

    return errors


REQUIRED_CSV_COLUMNS = [
    "Age", "Gender", "BusinessTravel", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def validate_csv_columns(columns) -> list:
    """Check that an uploaded CSV has all required columns. Returns a list
    of missing column names (empty list = valid)."""
    return [c for c in REQUIRED_CSV_COLUMNS if c not in columns]
