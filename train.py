"""
Trains an XGBoost churn classifier on the Telco dataset and saves
everything the Streamlit dashboard needs: the model, encoders,
test set, and precomputed SHAP values.

Usage:
    python train.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# --- Clean ---
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# --- Encode categoricals ---
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

preds_proba = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

print("\n--- Model Performance ---")
print("ROC-AUC:", round(roc_auc_score(y_test, preds_proba), 4))
print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

# --- Save artifacts for the dashboard ---
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "columns.pkl")
X_test.reset_index(drop=True).to_csv("X_test.csv", index=False)
y_test.reset_index(drop=True).to_csv("y_test.csv", index=False)

print("\nComputing SHAP values (may take a moment)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
joblib.dump(shap_values, "shap_values.pkl")
joblib.dump(explainer.expected_value, "expected_value.pkl")

print("\nDone. Artifacts saved: model.pkl, encoders.pkl, columns.pkl, "
      "X_test.csv, y_test.csv, shap_values.pkl, expected_value.pkl")
print("Now run: streamlit run app.py")
