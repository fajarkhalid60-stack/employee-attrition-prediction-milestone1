# Employee Attrition Risk — ML Deployment (Milestone 4, Phase 1)

**Project:** Predicting Employee Burnout and Resignation Risk Using Machine
Learning on Workplace Behavioral Data
**Phase:** ML Project Deployment — Phase 1 (Project Structure & Backend Setup)
**Author:** Fajar Khalid

---

## 1. Objective of This Phase

This phase transforms the trained machine learning model from Milestones 1-3
(notebook-based dataset preprocessing, feature engineering, and model
training/evaluation) into a clean, modular, deployment-ready project
structure with a basic working interface — not the final polished
application, but proof that the backend is organized and functioning
correctly.

## 2. Project Structure

```
employee-attrition-deployment/
├── data/
│   └── HR_Attrition_synthetic.csv   # Auto-generated demo dataset (see Section 6)
├── models/
│   └── attrition_model_bundle.joblib # Serialized model + scaler + encoders + feature order
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Cleaning, label/one-hot encoding
│   ├── feature_engineering.py        # Engineered features from Milestone 2
│   ├── model_loader.py               # Loads the serialized model bundle
│   ├── predict_pipeline.py           # End-to-end inference (single + batch)
│   └── utils.py                      # Risk-tier logic, input validation
├── train.py                          # Trains and serializes the model
├── app.py                            # Basic Streamlit demo interface
├── requirements.txt
└── README.md
```

Each script has a single responsibility, so the same `data_preprocessing.py`
and `feature_engineering.py` logic is reused identically during training
(`train.py`) and inference (`predict_pipeline.py`) — this guarantees the
model never sees differently-shaped data at inference time than it did
during training.

## 3. How the Pieces Fit Together

1. **`train.py`** loads/generates data → calls `feature_engineering.py` →
   calls `data_preprocessing.py` → splits, scales, balances (SMOTE-Tomek) →
   trains a Logistic Regression model (the best-performing model identified
   in Milestone 3) → serializes everything needed for inference into one
   Joblib bundle (`models/attrition_model_bundle.joblib`).
2. **`app.py`** loads that bundle once via `model_loader.py` (cached, so the
   model is never retrained while the app runs), collects a single
   employee's data through a form, and calls `predict_pipeline.py` to return
   a resignation-risk score.

## 4. Installation & Usage

```bash
pip install -r requirements.txt

# Train the model (only needs to be run once, or whenever the data changes)
python train.py

# Run the basic demo app
streamlit run app.py
```

To train on the real IBM HR Analytics dataset instead of the bundled
synthetic data:
```bash
python train.py --real-data path/to/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## 5. Model Serialization

The model, fitted `StandardScaler`, fitted `LabelEncoder`s, and the exact
training-time feature column order are all saved together in a single
Joblib bundle (`models/attrition_model_bundle.joblib`). Bundling them
together (rather than saving the model alone) was a deliberate choice — it
guarantees inference-time preprocessing can never drift out of sync with
what the model was actually trained on.

## 6. About the Data

The development environment does not have direct access to Kaggle, so
`train.py` generates a synthetic dataset that mirrors the exact column
schema and documented behavioral patterns of the IBM HR Analytics Employee
Attrition dataset used in Milestones 1-3 (see the logistic relationships
defined in `generate_synthetic_dataset()` — e.g., overtime and low
satisfaction increase resignation probability; tenure and income reduce it).
`train.py` fully supports swapping in the real dataset via `--real-data`,
with no other code changes required.

## 7. Technology Stack

- **Language:** Python 3
- **Data handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression, StandardScaler, LabelEncoder), Imbalanced-learn (SMOTE-Tomek)
- **Model persistence:** Joblib
- **Application framework:** Streamlit (basic demo interface for this phase)
- **Version control:** Git & GitHub

## 8. Issues Faced & How They Were Resolved

- **Inference-time column mismatch:** One-hot encoding a single employee
  record at inference time produces different (and fewer) dummy columns than
  the full training set — for example, if that one employee's Department
  happens to be the category that was dropped during `drop_first=True`
  encoding, or if a category present in training simply doesn't appear in a
  single test row. This was resolved by saving the exact training-time
  `feature_columns` list in the model bundle and calling
  `DataFrame.reindex(columns=feature_columns, fill_value=0)` before scaling —
  this guarantees the inference feature vector always has the identical
  shape and column order the model was trained on, regardless of which
  categories appear in a given input row.
- **Avoiding retraining on every run:** Initially, loading the model inside
  `app.py` directly would have retrained it on every Streamlit rerun
  (Streamlit reruns the whole script on each interaction). This was solved
  with Streamlit's `@st.cache_resource` decorator around the model-loading
  function, so the Joblib bundle is loaded from disk exactly once per app
  session instead of on every form submission.
- **Division-by-zero in feature engineering:** The `TenureRatio` feature
  (YearsAtCompany / TotalWorkingYears) breaks for employees with 0 total
  working years. This was fixed by replacing 0 with a small constant (0.5)
  only for the denominator, preserving the ratio's meaning without crashing.
- **Keeping train/inference preprocessing identical:** Early on, it was easy
  to accidentally write slightly different encoding logic for training vs.
  a single inference row. This was resolved by refactoring all preprocessing
  into shared functions in `src/data_preprocessing.py` that both `train.py`
  and `predict_pipeline.py` import and call — there is now only one
  implementation of the preprocessing logic, not two that could drift apart.

## 9. Verification

The full pipeline was tested end-to-end before submission:
- `train.py` ran successfully, producing a model with F1-score ≈ 0.50 on
  held-out synthetic test data (consistent with the Logistic Regression
  results from Milestone 3).
- `predict_pipeline.py` was tested directly with a clearly high-risk profile
  (young, low income, overtime, low satisfaction) and a clearly low-risk
  profile (senior, high income, high satisfaction) — the model correctly
  scored the high-risk profile at 99.3% and the low-risk profile at 0.7%
  probability of resignation.
- `app.py` was started with `streamlit run app.py` and confirmed to load and
  serve the interface successfully (HTTP 200) without errors.

## 10. Next Steps (Future Phases)

- Build out the full web application interface (beyond this basic demo form)
- Add batch CSV upload to the Streamlit/web interface
- Containerize the application (Docker) for deployment
- Deploy to a hosting platform (e.g., Streamlit Community Cloud, Render, or similar)
