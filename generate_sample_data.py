"""
Generates a synthetic dataset with the SAME schema as the real
Kaggle 'Telco Customer Churn' dataset, so the pipeline runs end-to-end
without needing an external download.

To use REAL data instead (recommended before putting this on your resume):
1. Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Save it as data/WA_Fn-UseC_-Telco-Customer-Churn.csv
3. Skip this script entirely.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

genders = np.random.choice(["Male", "Female"], N)
senior = np.random.choice([0, 1], N, p=[0.85, 0.15])
partner = np.random.choice(["Yes", "No"], N)
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
tenure = np.random.randint(0, 73, N)
phone = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.where(
    phone == "No", "No phone service", np.random.choice(["Yes", "No"], N)
)
internet = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.2])

def dependent_service(internet_col):
    return np.where(
        internet_col == "No", "No internet service", np.random.choice(["Yes", "No"], N)
    )

online_security = dependent_service(internet)
online_backup = dependent_service(internet)
device_protection = dependent_service(internet)
tech_support = dependent_service(internet)
streaming_tv = dependent_service(internet)
streaming_movies = dependent_service(internet)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.2]
)
paperless = np.random.choice(["Yes", "No"], N, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N,
)

# Monthly charges influenced by internet type and add-ons
base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 45, 20))
addon_count = (
    (online_security == "Yes").astype(int)
    + (online_backup == "Yes").astype(int)
    + (device_protection == "Yes").astype(int)
    + (tech_support == "Yes").astype(int)
    + (streaming_tv == "Yes").astype(int)
    + (streaming_movies == "Yes").astype(int)
)
monthly_charges = base + addon_count * 5 + np.random.normal(0, 5, N)
monthly_charges = np.clip(monthly_charges, 18, 120).round(2)
total_charges = (monthly_charges * tenure + np.random.normal(0, 20, N)).clip(min=0).round(2)

# Churn probability driven by realistic risk factors
risk = (
    (contract == "Month-to-month") * 0.35
    + (internet == "Fiber optic") * 0.15
    + (tenure < 12) * 0.25
    + (payment_method == "Electronic check") * 0.15
    + (paperless == "Yes") * 0.05
    + (senior == 1) * 0.1
    - (tenure > 48) * 0.2
    - (contract == "Two year") * 0.3
    + np.random.normal(0, 0.15, N)
)
churn_prob = 1 / (1 + np.exp(-4 * (risk - 0.3)))
churn = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"{i:04d}-CUST{np.random.randint(1000,9999)}" for i in range(N)],
    "gender": genders,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

df.to_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", index=False)
print(f"Generated {len(df)} rows -> data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Churn rate: {(df['Churn']=='Yes').mean():.1%}")
