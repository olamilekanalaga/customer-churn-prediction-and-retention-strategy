from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "business_summary.json"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
)


@st.cache_resource
def load_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "High Risk", "#dc2626"
    if probability >= 0.4:
        return "Medium Risk", "#d97706"
    return "Low Risk", "#059669"


def retention_actions(customer: dict, probability: float) -> list[str]:
    actions = []
    if customer["Contract"] == "Month-to-month":
        actions.append("Offer an annual contract incentive or loyalty discount.")
    if customer["PaymentMethod"] == "Electronic check":
        actions.append("Promote automatic payment options with a small billing credit.")
    if customer["tenure"] <= 12:
        actions.append("Trigger onboarding support and a 30-day satisfaction check-in.")
    if customer["TechSupport"] == "No":
        actions.append("Bundle technical support into a retention offer.")
    if probability >= 0.65:
        actions.append("Route to a priority retention queue within 48 hours.")
    if not actions:
        actions.append("Maintain current relationship and monitor for billing or usage changes.")
    return actions


bundle = load_model_bundle()
df = load_data()
metrics = load_json(METRICS_PATH)
summary = load_json(SUMMARY_PATH)
model = bundle["model"]
model_name = bundle["model_name"]

st.title("Customer Churn Prediction & Retention Strategy")
st.caption("Business-focused churn risk scoring for proactive customer retention.")

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Customers Analyzed", f"{summary['customers']:,}")
kpi_2.metric("Observed Churn Rate", f"{summary['overall_churn_rate']:.1%}")
kpi_3.markdown(
    f"""
    <div style="padding-top: 0.15rem;">
        <div style="font-size: 0.9rem; color: #334155;">Best Model</div>
        <div style="font-size: 1.35rem; font-weight: 700; color: #111827; line-height: 1.2;">{model_name}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
kpi_4.metric("ROC-AUC", f"{bundle['metrics']['roc_auc']:.3f}")

tab_predict, tab_insights, tab_model = st.tabs(
    ["Risk Scoring", "Business Insights", "Model Performance"]
)

with tab_predict:
    left, right = st.columns([0.38, 0.62])

    with left:
        st.subheader("Customer Profile")
        gender = st.selectbox("Gender", sorted(df["gender"].unique()))
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", sorted(df["Partner"].unique()))
        dependents = st.selectbox("Dependents", sorted(df["Dependents"].unique()))
        tenure = st.slider("Tenure", 0, 72, 8)
        phone = st.selectbox("Phone Service", sorted(df["PhoneService"].unique()))
        multiple_lines = st.selectbox("Multiple Lines", sorted(df["MultipleLines"].unique()))
        internet = st.selectbox("Internet Service", sorted(df["InternetService"].unique()))
        online_security = st.selectbox("Online Security", sorted(df["OnlineSecurity"].unique()))
        online_backup = st.selectbox("Online Backup", sorted(df["OnlineBackup"].unique()))
        device_protection = st.selectbox("Device Protection", sorted(df["DeviceProtection"].unique()))
        tech_support = st.selectbox("Tech Support", sorted(df["TechSupport"].unique()))
        streaming_tv = st.selectbox("Streaming TV", sorted(df["StreamingTV"].unique()))
        streaming_movies = st.selectbox("Streaming Movies", sorted(df["StreamingMovies"].unique()))
        contract = st.selectbox("Contract", sorted(df["Contract"].unique()))
        paperless = st.selectbox("Paperless Billing", sorted(df["PaperlessBilling"].unique()))
        payment = st.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()))
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=79.85)
        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(max(0, tenure * monthly)),
        )

    customer = {
        "gender": gender,
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
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    customer_df = pd.DataFrame([customer])
    probability = float(model.predict_proba(customer_df)[0, 1])
    label, color = risk_label(probability)

    with right:
        st.subheader("Predicted Churn Risk")
        st.markdown(
            f"""
            <div style="border-left: 8px solid {color}; padding: 1rem 1.25rem; background: #f8fafc;">
                <div style="font-size: 1rem; color: #475569;">Risk Segment</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {color};">{label}</div>
                <div style="font-size: 1.3rem; color: #0f172a;">Churn probability: {probability:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(probability)
        st.subheader("Recommended Retention Actions")
        for action in retention_actions(customer, probability):
            st.write(f"- {action}")

with tab_insights:
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(PROJECT_ROOT / "reports" / "figures" / "churn_by_contract.png"))
        st.image(str(PROJECT_ROOT / "reports" / "figures" / "tenure_by_churn.png"))
    with col_b:
        st.subheader("Commercial Readout")
        st.write(
            "Churn is concentrated among month-to-month customers, customers using electronic checks, "
            "and customers with short tenure. These segments should receive earlier engagement and "
            "more targeted retention offers."
        )
        st.metric(
            "Observed Monthly Revenue at Risk",
            f"${summary['observed_monthly_revenue_at_risk']:,.0f}",
        )
        st.write("High-risk segment churn rates:")
        for segment, value in summary["high_risk_segments"].items():
            st.write(f"- {segment.replace('_', ' ').title()}: {value:.1%}")

with tab_model:
    st.subheader("Model Comparison")
    metrics_df = pd.DataFrame(metrics).T.sort_values("roc_auc", ascending=False)
    st.dataframe(metrics_df, width="stretch")
    col_c, col_d = st.columns(2)
    with col_c:
        st.image(str(PROJECT_ROOT / "reports" / "figures" / "confusion_matrix.png"))
    with col_d:
        st.image(str(PROJECT_ROOT / "reports" / "figures" / "roc_curve.png"))
    st.image(str(PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"))
