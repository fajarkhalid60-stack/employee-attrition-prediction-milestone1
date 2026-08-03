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


def validate_employee_input(data: dict) -> list:
    """
    Basic validation for a single employee record submitted through the app.
    Returns a list of error messages (empty list = valid).
    """
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
