import streamlit as st
import pandas as pd
import joblib


# CALLBACK FUNCTIONS

def phone_service_callback():
    if st.session_state["phone_service"] == "No":
        st.session_state["multiple_lines"] = "No phone service"


def no_internet_service_callback():
    if st.session_state["internet_service"] == "No":
        for key in [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies"
        ]:
            st.session_state[key] = "No internet service"

# LOAD MODEL

model = joblib.load("model/telco_churn_model.pkl")

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details to estimate their probability of churn.")

# CUSTOMER INFORMATION

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

# PHONE SERVICE

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"],
    key="phone_service",
    on_change=phone_service_callback
)

no_phone = phone_service == "No"

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"],
    key="multiple_lines",
    disabled=no_phone
)

# INTERNET SERVICE

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
    key="internet_service",
    on_change=no_internet_service_callback
)

no_internet = internet_service == "No"


# INTERNET-DEPENDENT SERVICES

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"],
    key="online_security",
    disabled=no_internet
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"],
    key="online_backup",
    disabled=no_internet
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"],
    key="device_protection",
    disabled=no_internet
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"],
    key="tech_support",
    disabled=no_internet
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"],
    key="streaming_tv",
    disabled=no_internet
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"],
    key="streaming_movies",
    disabled=no_internet
)

# CONTRACT & BILLING

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


# CREATE INPUT DATAFRAME

input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges]
})

# PREDICTION

if st.button("Predict Churn"):

    probability = model.predict_proba(input_data)[0, 1]

    st.subheader("Prediction")

    st.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )

    # Adjustable Threshold

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.2711404287709699,
        step=0.05
    )

    prediction = probability >= threshold

    if prediction:
        st.error("⚠️ Customer is predicted to churn")
    else:
        st.success("✅ Customer is predicted to stay")