"""
train.py
---------
Trains the employee attrition classifier end-to-end and serializes the
resulting model bundle with Joblib.

Usage
-----
    python train.py
    python train.py --real-data data/HR_Attrition.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.combine import SMOTETomek

from src.data_preprocessing import preprocess, check_data_quality
from src.feature_engineering import engineer_features

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DEPARTMENTS = ["Sales", "Research & Development", "Human Resources"]
JOB_ROLES = ["Sales Executive", "Research Scientist", "Laboratory Technician",
             "Manufacturing Director", "Healthcare Representative", "Manager",
             "Sales Representative", "Research Director", "Human Resources"]
EDUCATION_FIELDS = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"]
MARITAL_STATUS = ["Single", "Married", "Divorced"]
BUSINESS_TRAVEL = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]


def generate_synthetic_dataset(n=1470):
    age = np.random.randint(18, 61, n)
    total_working_years = np.clip((age - 18) - np.random.randint(0, 6, n), 0, None)
    years_at_company = np.clip(total_working_years - np.random.randint(0, 8, n), 0, None)
    years_in_current_role = np.clip(years_at_company - np.random.randint(0, 4, n), 0, None)
    years_since_promotion = np.clip(np.random.poisson(2, n), 0, years_at_company)
    years_with_manager = np.clip(years_at_company - np.random.randint(0, 5, n), 0, None)
    job_level = np.random.randint(1, 6, n)
    monthly_income = (job_level * 2200 + total_working_years * 150 + np.random.normal(0, 900, n)).clip(1000, 20000)
    overtime = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    job_satisfaction = np.random.randint(1, 5, n)
    environment_satisfaction = np.random.randint(1, 5, n)
    work_life_balance = np.random.randint(1, 5, n)
    relationship_satisfaction = np.random.randint(1, 5, n)
    job_involvement = np.random.randint(1, 5, n)
    performance_rating = np.random.choice([3, 4], n, p=[0.85, 0.15])
    stock_option_level = np.random.randint(0, 4, n)
    percent_salary_hike = np.random.randint(11, 26, n)
    training_times = np.random.randint(0, 7, n)
    distance_from_home = np.random.randint(1, 30, n)
    num_companies_worked = np.random.randint(0, 9, n)
    education = np.random.randint(1, 6, n)

    df = pd.DataFrame({
        "Age": age, "Gender": np.random.choice(["Male", "Female"], n),
        "BusinessTravel": np.random.choice(BUSINESS_TRAVEL, n, p=[0.7, 0.19, 0.11]),
        "Department": np.random.choice(DEPARTMENTS, n, p=[0.31, 0.65, 0.04]),
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": np.random.choice(EDUCATION_FIELDS, n),
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": np.random.choice(JOB_ROLES, n),
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": np.random.choice(MARITAL_STATUS, n),
        "MonthlyIncome": monthly_income.round(0),
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager,
    })

    logit = (
        -1.6
        + 1.15 * (df["OverTime"] == "Yes").astype(int)
        - 0.55 * (df["JobSatisfaction"] - 2.5)
        - 0.45 * (df["WorkLifeBalance"] - 2.5)
        - 0.35 * (df["EnvironmentSatisfaction"] - 2.5)
        - 0.00012 * (df["MonthlyIncome"] - 6000)
        - 0.05 * df["YearsAtCompany"]
        + 0.10 * df["NumCompaniesWorked"]
        + 0.02 * df["DistanceFromHome"]
        - 0.20 * df["StockOptionLevel"]
        - 0.06 * df["JobLevel"]
        + 0.06 * df["YearsSinceLastPromotion"]
        - 0.015 * (df["Age"] - 35)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["Attrition"] = np.random.binomial(1, prob)
    df["Attrition"] = df["Attrition"].map({1: "Yes", 0: "No"})
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=str, default=None)
    parser.add_argument("--n", type=int, default=1470)
    args = parser.parse_args()

    if args.real_data and os.path.exists(args.real_data):
        print(f"Loading real dataset from {args.real_data} ...")
        df = pd.read_csv(args.real_data)
    else:
        print(f"No real dataset provided — generating {args.n} rows of synthetic data ...")
        df = generate_synthetic_dataset(args.n)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/HR_Attrition_synthetic.csv", index=False)
        print("Synthetic dataset saved to data/HR_Attrition_synthetic.csv")

    print(f"Data quality check: {check_data_quality(df)}")

    df = engineer_features(df)
    df_encoded, encoders = preprocess(df, encoders=None, reference_columns=None)

    target_col = "Attrition"
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    feature_columns = X.columns.tolist()

    print(f"Final feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_scaled, y_train)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n=== Evaluation on held-out test set ===")
    for k, v in metrics.items():
        print(f"{k.capitalize():10s}: {v}")
    print(f"Confusion Matrix: {cm}")

    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": model, "scaler": scaler, "encoders": encoders,
        "feature_columns": feature_columns, "metrics": metrics,
        "confusion_matrix": cm,
        "trained_on": "real_data" if (args.real_data and os.path.exists(args.real_data)) else "synthetic_data",
    }
    bundle_path = "models/attrition_model_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"\nModel bundle saved to {bundle_path}")


if __name__ == "__main__":
    main()
