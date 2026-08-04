# Employee Attrition Risk — ML Deployment (Phase 2: Application Development)

**Project:** Predicting Employee Burnout and Resignation Risk Using Machine
Learning on Workplace Behavioral Data
**Phase:** ML Project Deployment — Phase 2 (Application Development & Deployment)
**Author:** Fajar Khalid

---

## 1. Overview

This is the final implementation milestone: the trained model from Milestones
1-3, restructured into a modular backend in Phase 1, is now wired into a
**complete, functional web application** — a Streamlit app that loads the
saved model, accepts real user input, runs it through the full prediction
pipeline, and displays results, for both a single employee and a bulk CSV
upload.

## 2. Project Structure

```
employee-attrition-deployment/
├── data/
│   ├── HR_Attrition_synthetic.csv   # Auto-generated training dataset
│   └── sample_employees.csv         # Sample file for testing batch upload
├── models/
│   └── attrition_model_bundle.joblib # Serialized model + scaler + encoders + feature order
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Cleaning, label/one-hot encoding
│   ├── feature_engineering.py        # Engineered features from Milestone 2
│   ├── model_loader.py               # Loads the serialized model bundle
│   ├── predict_pipeline.py           # End-to-end inference + explainability
│   └── utils.py                      # Risk-tier logic, validation, top-factor explanation
├── train.py                          # Trains and serializes the model
├── app.py                            # Full Streamlit application (Phase 2)
├── requirements.txt
└── README.md
```

## 3. Application Features

The app has two tabs:

**🔍 Single Employee Prediction**
- A form covering all model inputs (demographics, satisfaction scores, tenure, compensation, etc.)
- Instant resignation probability, risk tier (Low/Medium/High), and prediction
- **Explainability panel** — shows the top 5 factors driving that specific prediction, each tagged as increasing or decreasing risk, visualized as a bar chart

**📁 Batch Prediction (CSV Upload)**
- Upload a CSV of an entire team at once
- Validates that all required columns are present before processing
- Summary metrics (total employees, Low/Medium/High counts) and a risk-tier bar chart
- Full results table, sortable by risk
- One-click **CSV download** of the scored results

Both tabs use the exact same `predict_pipeline.py` functions, so single and
batch predictions are always guaranteed to be consistent with each other.

## 4. Installation & Usage

```bash
pip install -r requirements.txt

# Train the model (only needs to be run once, or whenever the data changes)
python train.py

# Run the application
streamlit run app.py
```

To test batch prediction, use the bundled `data/sample_employees.csv` in the
"Batch Prediction" tab.

To train on the real IBM HR Analytics dataset instead of the bundled
synthetic data:
```bash
python train.py --real-data path/to/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## 5. Technology Stack

- **Language:** Python 3
- **Data handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression, StandardScaler, LabelEncoder), Imbalanced-learn (SMOTE-Tomek)
- **Model persistence:** Joblib
- **Application framework:** Streamlit
- **Version control:** Git & GitHub

## 6. Model Performance

| Metric | Value |
|---|---|
| Accuracy | 0.7653 |
| Precision | 0.4000 |
| Recall | 0.8163 |
| F1-Score | 0.5369 |

Recall is prioritized over precision for this use case: missing an at-risk
employee (a false negative) defeats the purpose of the tool, since HR would
fail to intervene in time — a false alarm is a much lower-cost mistake.

## 7. Issues Faced & How They Were Resolved

- **Contradictory explanations from correlated features:** Early testing of
  the explainability feature (Section 8, "Issues" from Phase 1's report also
  applies here) surfaced a new problem specific to Phase 2: `MonthlyIncome`
  and its log-transformed version `LogMonthlyIncome` were both fed into the
  model as separate features. Because they're almost perfectly correlated,
  the model split their combined effect across both — a classic
  multicollinearity symptom — which produced a nonsensical explanation
  (e.g., "MonthlyIncome increases risk" and "LogMonthlyIncome decreases
  risk" appearing as the top two factors for the same prediction). This was
  fixed by dropping the raw `MonthlyIncome` column immediately after
  computing `LogMonthlyIncome` in `feature_engineering.py`, keeping only one
  version of the income signal. This also **improved** the model's F1-score
  from 0.5032 to 0.5369, since the model no longer had to "split" its
  confidence across two competing, redundant features.
- **Single-prediction vs. batch-prediction consistency:** Initially, single
  and batch prediction had separate, slightly different code paths. This was
  refactored so both `predict_single()` and `predict_batch()` call the same
  internal `_prepare_features()` helper in `predict_pipeline.py` — guaranteeing
  a single employee scored individually and the same employee scored as part
  of a CSV batch always produce an identical result.
- **CSV upload validation:** An uploaded CSV with missing or misnamed columns
  would previously fail deep inside the pipeline with a confusing pandas
  error. This was fixed by adding `validate_csv_columns()` in `utils.py`,
  which checks for all required columns up front and shows the user a clear,
  specific error message (listing exactly which columns are missing) before
  any processing happens.
- **Avoiding retraining on every interaction:** Streamlit reruns the entire
  script on every user interaction (e.g., switching tabs, submitting the
  form). `@st.cache_resource` around the model-loading function ensures the
  Joblib bundle is loaded from disk exactly once per session, regardless of
  how many predictions are made afterward.

## 8. Verification

Before submission, the following were tested directly:
- `predict_single()` on a clearly high-risk profile (99.58% probability,
  correctly flagged High) and a clearly low-risk profile (1.02% probability,
  correctly flagged Low), confirming both the prediction and the
  explanation are sensible and non-contradictory after the multicollinearity fix.
- `predict_batch()` against `data/sample_employees.csv` (15 employees),
  producing a realistic spread across all three risk tiers (7 Low, 4 Medium,
  4 High).
- The full Streamlit application was started with `streamlit run app.py`
  and confirmed to load successfully (HTTP 200, no errors) before submission.

## 9. About the Data

The development environment does not have direct access to Kaggle, so
`train.py` generates a synthetic dataset that mirrors the exact column
schema and documented behavioral patterns of the IBM HR Analytics Employee
Attrition dataset used in Milestones 1-3. `train.py` fully supports swapping
in the real dataset via `--real-data`, with no other code changes required.

## 10. Possible Future Enhancements

- User authentication for multi-HR-user deployments
- Persistent storage of past predictions (database instead of stateless CSV upload)
- A model-selection toggle (Random Forest / SVM were also validated in Milestone 3)
- Deployment to a public hosting platform (e.g., Streamlit Community Cloud)
