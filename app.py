"""
app.py
-------
Employee Attrition Risk Predictor — Phase 2: full application.

Loads the pre-trained model once (no retraining at runtime), and provides
two ways to interact with it:
  1. Single Employee Prediction — a form with instant results and an
     explanation of the top factors driving that prediction.
  2. Batch Prediction (CSV Upload) — score an entire team at once, with
     validation, a summary chart, and a downloadable results file.

Run with:
    streamlit run app.py
"""

import io
import pandas as pd
import streamlit as st

from src.model_loader import load_artifacts
from src.predict_pipeline import predict_single, predict_batch
from src.utils import validate_employee_input, validate_csv_columns, REQUIRED_CSV_COLUMNS

st.set_page_config(page_title="Employee Attrition Risk Predictor", page_icon="📊", layout="wide")


@st.cache_resource
def get_artifacts():
    """Load the trained model bundle once and cache it across reruns, so the
    model is never retrained while the app is running."""
    return load_artifacts()


st.title("📊 Employee Attrition Risk Predictor")
st.write(
    "Predict an employee's resignation risk from workplace behavioral data, "
    "using a Logistic Regression model trained on HR analytics data."
)

try:
    artifacts = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Model Info")
    st.metric("F1-Score", artifacts["metrics"]["f1_score"])
    st.metric("Accuracy", artifacts["metrics"]["accuracy"])
    st.metric("Recall", artifacts["metrics"]["recall"])
    st.caption(f"Trained on: {artifacts['trained_on']}")
    st.divider()
    st.caption(
        "Recall is prioritized over precision for this use case — missing "
        "an at-risk employee (a false negative) is more costly than a false alarm."
    )

tab1, tab2 = st.tabs(["🔍 Single Employee Prediction", "📁 Batch Prediction (CSV Upload)"])

# ---------------------------------------------------------------------------
# TAB 1 — Single employee prediction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Check an employee's resignation risk")

    with st.form("employee_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
            distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=60, value=10)
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=45, value=8)
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])

        with col2:
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=4)
            years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
            num_companies_worked = st.number_input("Companies Worked Previously", min_value=0, max_value=15, value=2)
            job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4], index=2)
            environment_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4], index=2)
            work_life_balance = st.selectbox("Work-Life Balance (1-4)", [1, 2, 3, 4], index=2)

        with col3:
            stock_option_level = st.selectbox("Stock Option Level (0-3)", [0, 1, 2, 3], index=0)
            overtime = st.selectbox("Works Overtime", ["Yes", "No"], index=1)
            job_role = st.text_input("Job Role", value="Sales Executive")
            education_field = st.selectbox(
                "Education Field",
                ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
            )
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

        submitted = st.form_submit_button("Predict Resignation Risk", type="primary")

    if submitted:
        employee = {
            "Age": age, "Gender": "Male", "BusinessTravel": business_travel,
            "Department": department, "DistanceFromHome": distance_from_home,
            "Education": 3, "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction, "JobInvolvement": 3,
            "JobLevel": job_level, "JobRole": job_role, "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
            "NumCompaniesWorked": num_companies_worked, "OverTime": overtime,
            "PercentSalaryHike": 15, "PerformanceRating": 3,
            "RelationshipSatisfaction": 3, "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": 2,
            "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": max(years_at_company - 1, 0),
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": max(years_at_company - 1, 0),
        }

        errors = validate_employee_input(employee)
        if errors:
            for err in errors:
                st.error(err)
        else:
            result = predict_single(employee, artifacts, explain=True)

            st.divider()
            r1, r2 = st.columns([1, 2])

            with r1:
                st.metric("Resignation Probability", f"{result['probability'] * 100:.1f}%")
                tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[result["tier"]]
                st.markdown(f"### Risk Tier: :{tier_color}[{result['tier']}]")
                st.write(f"**Prediction:** {result['prediction']}")

            with r2:
                st.write("**Top factors driving this prediction:**")
                factor_df = pd.DataFrame(result["top_factors"])
                factor_df["abs_contribution"] = factor_df["contribution"].abs()
                factor_df = factor_df.sort_values("abs_contribution")
                st.bar_chart(
                    factor_df.set_index("feature")["contribution"],
                    horizontal=True,
                )
                for f in result["top_factors"]:
                    arrow = "🔺" if f["contribution"] > 0 else "🔻"
                    st.caption(f"{arrow} **{f['feature']}** {f['direction']}")

# ---------------------------------------------------------------------------
# TAB 2 — Batch prediction via CSV upload
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Score an entire team at once")
    st.caption(f"Required columns: {', '.join(REQUIRED_CSV_COLUMNS)}")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            df_input = None

        if df_input is not None:
            missing_cols = validate_csv_columns(df_input.columns)
            if missing_cols:
                st.error(f"The CSV is missing required column(s): {', '.join(missing_cols)}")
            elif df_input.empty:
                st.error("The uploaded CSV has no data rows.")
            else:
                with st.spinner("Scoring employees..."):
                    results_df = predict_batch(df_input, artifacts)

                st.success(f"Scored {len(results_df)} employees successfully.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Employees", len(results_df))
                c2.metric("Low Risk", int((results_df["RiskTier"] == "Low").sum()))
                c3.metric("Medium Risk", int((results_df["RiskTier"] == "Medium").sum()))
                c4.metric("High Risk", int((results_df["RiskTier"] == "High").sum()))

                st.bar_chart(results_df["RiskTier"].value_counts())

                st.dataframe(
                    results_df.sort_values("ResignationProbability", ascending=False),
                    use_container_width=True,
                )

                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇ Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="attrition_risk_results.csv",
                    mime="text/csv",
                )
