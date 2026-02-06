import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("loan_approval_model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Predict whether a loan should be approved based on applicant details.")

st.divider()

# ---- User Inputs ----
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(
        label=f"Enter {feature}",
        value=0.0
    )

input_df = pd.DataFrame([user_input])

# ---- Prediction ----
probability = model.predict_proba(input_df)[0][1]

# ---- Decision Logic ----
if probability >= 0.7:
    decision = "✅ Approved"
elif probability >= 0.4:
    decision = "🟡 Manual Review Required"
else:
    decision = "❌ Rejected"

# ---- Display Results ----
st.divider()
st.subheader("📊 Prediction Result")

st.metric("Approval Probability", f"{probability:.2%}")
st.subheader(f"Decision: {decision}")

# ---- Explanation ----
st.divider()
st.subheader("🔍 Model Explanation")

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0],
    "Odds Ratio": np.exp(model.coef_[0])
}).sort_values("Odds Ratio", ascending=False)

st.dataframe(coef_df)

st.caption("Odds Ratio > 1 increases approval likelihood, < 1 decreases it.")
