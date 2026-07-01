"""
Streamlit dashboard for customer churn risk.
Run with: streamlit run app.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide", page_icon="📉")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    columns = joblib.load("columns.pkl")
    shap_values = joblib.load("shap_values.pkl")
    expected_value = joblib.load("expected_value.pkl")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")
    return model, encoders, columns, shap_values, expected_value, X_test, y_test


model, encoders, columns, shap_values, expected_value, X_test, y_test = load_artifacts()

X_test["churn_prob"] = model.predict_proba(X_test[columns])[:, 1]
X_test["actual_churn"] = y_test.values

def decode(col, series):
    """Turn label-encoded ints back into readable category names."""
    inv_map = dict(enumerate(encoders[col].classes_))
    return series.map(inv_map)

readable = X_test.copy()
for col in encoders:
    if col in readable.columns:
        readable[col] = decode(col, readable[col])

# ---------- Header ----------
st.title("📉 Customer Churn Risk Dashboard")
st.caption("XGBoost model with SHAP-based explainability")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
contract_options = list(encoders["Contract"].classes_)
contract_filter = st.sidebar.multiselect(
    "Contract Type", contract_options, default=contract_options
)
tenure_range = st.sidebar.slider("Tenure (months)", 0, int(X_test["tenure"].max()), (0, int(X_test["tenure"].max())))
internet_options = list(encoders["InternetService"].classes_)
internet_filter = st.sidebar.multiselect(
    "Internet Service", internet_options, default=internet_options
)

mask = (
    readable["Contract"].isin(contract_filter)
    & readable["tenure"].between(*tenure_range)
    & readable["InternetService"].isin(internet_filter)
)
filtered = readable[mask]
filtered_idx = filtered.index  # positions into X_test / shap_values

# ---------- KPI row ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers Shown", f"{len(filtered):,}")
col2.metric("Avg Churn Risk", f"{filtered['churn_prob'].mean():.1%}" if len(filtered) else "—")
high_risk = (filtered["churn_prob"] > 0.5).sum()
col3.metric("High-Risk Customers", f"{high_risk:,}")
revenue_at_risk = (filtered["churn_prob"] * filtered["MonthlyCharges"]).sum()
col4.metric("Monthly Revenue at Risk", f"₹{revenue_at_risk:,.0f}")

st.divider()

# ---------- Charts ----------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Churn Risk by Contract Type")
    fig = px.box(filtered, x="Contract", y="churn_prob", color="Contract", points=False)
    fig.update_layout(showlegend=False, yaxis_title="Churn Probability")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Churn Risk vs. Tenure")
    fig2 = px.scatter(
        filtered, x="tenure", y="churn_prob", color="Contract",
        opacity=0.5, labels={"tenure": "Tenure (months)", "churn_prob": "Churn Probability"}
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------- Risk table ----------
st.subheader("🚨 Customers Ranked by Churn Risk")
display_cols = ["tenure", "Contract", "InternetService", "MonthlyCharges", "churn_prob"]
top_risk = filtered.sort_values("churn_prob", ascending=False)[display_cols].head(25)
st.dataframe(
    top_risk.style.format({"churn_prob": "{:.1%}", "MonthlyCharges": "₹{:.2f}"})
    .background_gradient(subset=["churn_prob"], cmap="Reds"),
    use_container_width=True,
)

st.divider()

# ---------- Individual explanation ----------
st.subheader("🔍 Explain an Individual Customer's Risk")

if len(filtered) == 0:
    st.info("No customers match the current filters.")
else:
    options = filtered.sort_values("churn_prob", ascending=False).index.tolist()
    selected = st.selectbox(
        "Select a customer (sorted by risk, highest first)",
        options,
        format_func=lambda i: f"Row {i} — {readable.loc[i,'Contract']}, "
                               f"{readable.loc[i,'tenure']}mo tenure, "
                               f"risk {readable.loc[i,'churn_prob']:.1%}",
    )

    row = readable.loc[selected]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Churn Probability", f"{row['churn_prob']:.1%}")
        st.write(f"**Contract:** {row['Contract']}")
        st.write(f"**Tenure:** {row['tenure']} months")
        st.write(f"**Internet:** {row['InternetService']}")
        st.write(f"**Monthly Charges:** ₹{row['MonthlyCharges']:.2f}")
        st.write(f"**Payment Method:** {row['PaymentMethod']}")

    with right:
        st.write("**Why this prediction — SHAP waterfall**")
        exp = shap.Explanation(
            values=shap_values[selected],
            base_values=expected_value,
            data=X_test.loc[selected, columns].values,
            feature_names=columns,
        )
        fig3, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(exp, max_display=10, show=False)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

st.divider()

# ---------- Global feature importance ----------
st.subheader("🌍 Global Feature Importance")
st.caption("What drives churn predictions across the whole customer base")
fig4, ax4 = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test[columns], show=False, plot_size=None)
st.pyplot(fig4, use_container_width=True)
plt.close(fig4)

st.caption("Built with XGBoost + SHAP + Streamlit")