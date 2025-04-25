import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('credit_risk_model.pkl')

st.title("Credit Risk Predictor")

st.write("Fill in the details to check if a person is likely to be **High Risk** or **Low Risk**.")

age = st.slider("Age", 18, 75)
job = st.selectbox("Job", [0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", value=1000)
duration = st.slider("Duration (months)", 4, 72)

# Format input for model
input_data = {
    'Age': age,
    'Job': job,
    'Credit amount': credit_amount,
    'Duration': duration
}

def preprocess_input(data):
    df = pd.DataFrame([data])
    df.replace('NA', 'no_info', inplace=True)
    df = pd.get_dummies(df)
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    return df[model.feature_names_in_]

# Prediction
if st.button("Predict"):
    final_input = preprocess_input(input_data)
    pred = model.predict(final_input)[0]
    proba = model.predict_proba(final_input)[0][1]

    if pred == 1:
        st.error(f"High Credit Risk (Confidence: {proba:.2f})")
    else:
        st.success(f"Low Credit Risk (Confidence: {1 - proba:.2f})")
