# 📉 Customer Churn Prediction Dashboard

An end-to-end ML project that predicts customer churn risk and explains
*why* each customer is at risk, using SHAP. Includes an interactive
Streamlit dashboard for business-facing exploration.

## Problem

Subscription and telecom businesses lose significant revenue to customer
churn. This project builds a model to flag at-risk customers early and
surfaces the *drivers* of that risk, so a retention team can act on it
(not just a black-box score).

## Dataset

Uses the schema of the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(Kaggle / IBM sample data), ~7,000 customers with demographic, account,
and service-usage features.

> This repo ships with `generate_sample_data.py`, which creates a synthetic
> dataset with the identical schema so you can run the whole pipeline
> immediately. **Before showcasing this on a resume/portfolio, swap in the
> real Kaggle CSV** — see Setup step 2.

## Approach

- **Preprocessing:** cleaned missing values, label-encoded categoricals
- **Model:** XGBoost classifier (`n_estimators=200, max_depth=4`)
- **Evaluation:** ROC-AUC, precision/recall, confusion matrix
- **Explainability:** SHAP TreeExplainer for global feature importance
  and per-customer waterfall explanations
- **Dashboard:** Streamlit app with segment filters, risk ranking table,
  revenue-at-risk KPI, and individual customer explanations

## Results

| Metric | Score |
|---|---|
| ROC-AUC | ~0.77 (synthetic data) / ~0.82–0.85 typical on real data |
| Precision (Churn) | ~0.68 |
| Recall (Churn) | ~0.68 |

Top churn drivers (from SHAP): contract type (month-to-month), tenure,
internet service type, and payment method.

## Setup

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Get data (choose one)
python generate_sample_data.py                          # synthetic, works immediately
# OR download the real CSV from Kaggle and save as:
# data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Train the model (saves model.pkl, shap_values.pkl, etc.)
python train.py

# 4. Launch the dashboard
streamlit run app.py
```

Then open `http://localhost:8501`.

## Project structure

```
churn-dashboard/
├── data/                         # raw CSV goes here
├── generate_sample_data.py       # synthetic data generator (optional)
├── train.py                      # data cleaning, training, SHAP export
├── app.py                        # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Deployment

Push to GitHub, then deploy free at [share.streamlit.io](https://share.streamlit.io)
(Streamlit Community Cloud) or [Hugging Face Spaces](https://huggingface.co/spaces) —
point it at `app.py`, it installs from `requirements.txt` automatically.

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · Plotly
