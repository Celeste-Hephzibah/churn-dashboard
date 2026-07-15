# Customer Churn Prediction Dashboard

An end-to-end ML project that predicts customer churn risk and explains *why* each customer is at risk, using SHAP. Includes an interactive Streamlit dashboard for business-facing exploration.

## Live Demo

**[churn-dashboard-mzj5srqxmwydvhdforff7r.streamlit.app](https://churn-dashboard-mzj5srqxmwydvhdforff7r.streamlit.app/)**

<!-- Add a screenshot here — ideally the risk ranking table + a SHAP waterfall explanation.
Example:
![Dashboard screenshot](assets/demo-screenshot.png)
-->

Revenue-at-risk KPI is displayed in ₹ (INR).

## Problem

Subscription and telecom businesses lose significant revenue to customer churn. This project builds a model to flag at-risk customers early and surfaces the drivers of that risk, so a retention team can act on it — not just a black-box score.

## Dataset

Uses the schema of the Telco Customer Churn dataset (Kaggle / IBM sample data), ~7,000 customers with demographic, account, and service-usage features.

This repo ships with `generate_sample_data.py`, which creates a synthetic dataset with the identical schema so you can run the whole pipeline immediately. Before showcasing this on a resume/portfolio, swap in the real Kaggle CSV — see Setup step 2.

## Approach

- **Preprocessing:** cleaned missing values, label-encoded categoricals
- **Model:** XGBoost classifier (`n_estimators=200`, `max_depth=4`)
- **Evaluation:** ROC-AUC, precision/recall, confusion matrix
- **Explainability:** SHAP TreeExplainer for global feature importance and per-customer waterfall explanations
- **Dashboard:** Streamlit app with segment filters, risk ranking table, revenue-at-risk KPI, and individual customer explanations

## Results

| Metric | Synthetic Data | Typical on Real Data |
|---|---|---|
| ROC-AUC | ~0.77 | ~0.82–0.85 |
| Precision (Churn) | ~0.68 | — |
| Recall (Churn) | ~0.68 | — |

Top churn drivers (from SHAP): contract type (month-to-month), tenure, internet service type, and payment method.

## Setup

```bash
# 1. Create environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
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

Then open http://localhost:8501.

## Project Structure

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

Already deployed on Streamlit Community Cloud (see Live Demo above). To deploy your own fork: push to GitHub, then deploy free at [share.streamlit.io](https://share.streamlit.io) or Hugging Face Spaces — point it at `app.py`, it installs from `requirements.txt` automatically.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · Plotly
