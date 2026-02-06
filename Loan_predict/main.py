import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.write("Predict whether a loan should be approved based on applicant details.")

st.divider()

# -------------------------------
# Load Model & Features Safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "loan_approval_model.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)

# -------------------------------
# Employment Type Mapping
# -------------------------------
employment_map = {
    "Salaried": 0,
    "Self-Employed": 1,
    "Unemployed": 2
}

# -------------------------------
# User Input Section
# -------------------------------
user_input = {}

for feature in features:
    if feature == "Employment_Type":
        selected_emp = st.selectbox(
            "Select Employment Type",
            options=list(employment_map.keys())
        )
        user_input[feature] = employment_map[selected_emp]

    elif feature == "Age":
        user_input[feature] = st.number_input(
            "Enter Age",
            min_value=18,
            max_value=100,
            value=30
        )

    elif feature == "Credit_Score":
        user_input[feature] = st.number_input(
            "Enter Credit Score",
            min_value=300,
            max_value=900,
            value=700
        )

    else:
        user_input[feature] = st.number_input(
            f"Enter {feature.replace('_', ' ')}",
            min_value=0.0,
            value=0.0
        )

input_df = pd.DataFrame([user_input])

# -------------------------------
# Prediction
# -------------------------------
probability = model.predict_proba(input_df)[0][1]

# Decision Rules
if probability >= 0.7:
    decision = "✅ Approved"
elif probability >= 0.4:
    decision = "🟡 Manual Review Required"
else:
    decision = "❌ Rejected"

# -------------------------------
# Results Display
# -------------------------------
st.divider()
st.subheader("📊 Prediction Result")

st.metric("Approval Probability", f"{probability:.2%}")
st.subheader(f"Decision: {decision}")

# -------------------------------
# Model Explanation
# -------------------------------
st.divider()
st.subheader("🔍 Model Explanation")

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0],
    "Odds Ratio": np.exp(model.coef_[0])
}).sort_values("Odds Ratio", ascending=False)

st.dataframe(coef_df, use_container_width=True)

st.caption("Odds Ratio > 1 increases approval likelihood, < 1 decreases it.")







