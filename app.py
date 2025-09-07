import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Set the path to the working directory in Kaggle
working_dir = '/kaggle/working/'

# --- 1. Load the trained model and preprocessor components ---
try:
    # Load the trained Keras model as a joblib file
    model_path = os.path.join(working_dir, 'credit_score_nn_model.joblib')
    model = joblib.load(model_path)
    # Load the preprocessor and label encoder
    preprocessor_path = os.path.join(working_dir, 'preprocessor.joblib')
    preprocessor = joblib.load(preprocessor_path)
    label_encoder_path = os.path.join(working_dir, 'label_encoder.joblib')
    label_encoder = joblib.load(label_encoder_path)
except FileNotFoundError:
    st.error("Error: Model or preprocessing files not found. Please run the training script first to create them.")
    st.stop()

# --- 2. Define the app title and form ---
st.title("Credit Score Prediction App")
st.write("Enter the customer's information below to predict the credit score (Poor, Standard, or Good).")

with st.form("prediction_form"):
    st.header("Customer Data")
    
    # Input fields
    annual_income = st.number_input("Annual Income", min_value=0.0, format="%.2f")
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, format="%.2f")
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, step=1)
    num_credit_card = st.number_input("Number of Credit Cards", min_value=0, step=1)
    interest_rate = st.number_input("Interest Rate", min_value=0.0, format="%.2f")
    num_of_loan = st.number_input("Number of Loans", min_value=0, step=1)
    delay_from_due_date = st.number_input("Delay from Due Date (in days)", min_value=0, step=1)
    num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, step=1)
    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, format="%.2f")
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, step=1)
    outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, format="%.2f")
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, format="%.2f")
    total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, format="%.2f")
    amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, format="%.2f")
    monthly_balance = st.number_input("Monthly Balance", min_value=0.0, format="%.2f")
    credit_history_age_str = st.text_input("Credit History Age (e.g., 15 Years and 3 Months)")
    
    # Categorical features
    occupation = st.selectbox("Occupation", ["Scientist", "Mechanic", "Architect", "Engineer"])
    credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
    payment_of_min_amount = st.selectbox("Payment of Min Amount", ["No", "Yes"])
    payment_behaviour = st.selectbox("Payment Behaviour", ["High_spent_Small_value_payments", "Low_spent_Small_value_payments", "High_spent_Medium_value_payments"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Credit Score")

    if submitted:
        # --- 3. Process the input and make a prediction ---
        # Convert credit history age to months
        try:
            parts = credit_history_age_str.replace('and', '').replace('Months', '').replace('Years', '').split()
            years = int(parts[0])
            months = int(parts[1])
            credit_history_age_months = years * 12 + months
        except (ValueError, IndexError):
            st.warning("Please enter the credit history age in the correct format.")
            st.stop()
            
        # Create a DataFrame from the user's input
        input_data = pd.DataFrame([{
            'Annual_Income': annual_income,
            'Monthly_Inhand_Salary': monthly_inhand_salary,
            'Num_Bank_Accounts': num_bank_accounts,
            'Num_Credit_Card': num_credit_card,
            'Interest_Rate': interest_rate,
            'Num_of_Loan': num_of_loan,
            'Delay_from_due_date': delay_from_due_date,
            'Num_of_Delayed_Payment': num_of_delayed_payment,
            'Changed_Credit_Limit': changed_credit_limit,
            'Num_Credit_Inquiries': num_credit_inquiries,
            'Outstanding_Debt': outstanding_debt,
            'Credit_Utilization_Ratio': credit_utilization_ratio,
            'Total_EMI_per_month': total_emi_per_month,
            'Amount_invested_monthly': amount_invested_monthly,
            'Monthly_Balance': monthly_balance,
            'Occupation': occupation,
            'Credit_Mix': credit_mix,
            'Payment_of_Min_Amount': payment_of_min_amount,
            'Payment_Behaviour': payment_behaviour,
            'Credit_History_Age_Months': credit_history_age_months
        }])
        
        # Preprocess the input data using the saved preprocessor
        processed_input = preprocessor.transform(input_data)
        
        # Make a prediction
        prediction_prob = model.predict(processed_input)
        prediction = np.argmax(prediction_prob, axis=1)
        
        # Convert the numerical prediction back to a label
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # --- 4. Display the result ---
        if predicted_label == 'Good':
            st.success(f"Prediction Result: {predicted_label}")
        elif predicted_label == 'Standard':
            st.info(f"Prediction Result: {predicted_label}")
        else:
            st.warning(f"Prediction Result: {predicted_label}")
