import streamlit as st

from src.model_loader import load_artifacts
from src.predict_pipeline import predict_single
from src.utils import validate_employee_input

st.set_page_config(page_title="Employee Attrition Risk — Deployment Demo", page_icon="📊")

st.title("Employee Attrition Risk Predictor")

@st.cache_resource
def get_artifacts():
    """Load the trained model bundle once and cache it across reruns,
    so the model is never retrained while the app is running."""
    return load_artifacts()


try:
    artifacts = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.success(
    f"Model loaded successfully · trained on {artifacts['trained_on']} · "
    f"F1-score: {artifacts['metrics']['f1_score']}"
)

st.divider()
st.subheader("Check an employee's resignation risk")

with st.form("employee_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
        distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=60, value=10)
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=45, value=8)
        years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=4)
        years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

    with col2:
        num_companies_worked = st.number_input("Companies Worked Previously", min_value=0, max_value=15, value=2)
        job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4], index=2)
        environment_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4], index=2)
        work_life_balance = st.selectbox("Work-Life Balance (1-4)", [1, 2, 3, 4], index=2)
        stock_option_level = st.selectbox("Stock Option Level (0-3)", [0, 1, 2, 3], index=0)
        overtime = st.selectbox("Works Overtime", ["Yes", "No"], index=1)

    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.text_input("Job Role", value="Sales Executive")
    education_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
    )
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    submitted = st.form_submit_button("Predict Resignation Risk")

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
        result = predict_single(employee, artifacts)

        st.divider()
        tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[result["tier"]]
        st.metric("Resignation Probability", f"{result['probability'] * 100:.1f}%")
        st.markdown(f"**Risk Tier:** :{tier_color}[{result['tier']}]")
        st.write(f"**Prediction:** {result['prediction']}")

st.divider()
with st.expander("Model details"):
    st.json(artifacts["metrics"])
    st.caption(f"Trained on: {artifacts['trained_on']}")
