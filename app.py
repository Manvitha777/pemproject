import streamlit as st
import pandas as pd
import joblib

# =====================================================
# LOAD MODEL & SCALER & COLUMNS
# =====================================================
model = joblib.load("sepsis_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Sepsis Prediction", layout="centered")
st.title("🩺 Sepsis Prediction System")
st.write("Enter patient details to predict sepsis risk")

# =====================================================
# INPUT FIELDS
# =====================================================
st.subheader("Patient Information")
age = st.number_input("Age", 0, 120, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
insurance = st.selectbox("Insurance", ["Medicare", "Private", "Self-pay"])
weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
height = st.number_input("Height (cm)", 100.0, 220.0, 165.0)
bmi = weight / ((height/100) ** 2)

st.subheader("Vital Signs")
hr = st.number_input("Heart Rate", 40, 180, 80)
sbp = st.number_input("Systolic BP", 60, 200, 110)
dbp = st.number_input("Diastolic BP", 40, 130, 70)
temp = st.number_input("Temperature (C)", 34.0, 42.0, 37.0)
spo2 = st.number_input("SpO2", 70, 100, 98)
resp = st.number_input("Respiratory Rate", 10, 40, 18)

st.subheader("Lab Values")
wbc = st.number_input("WBC", 1.0, 30.0, 7.0)
glucose = st.number_input("Glucose", 50.0, 300.0, 100.0)
sodium = st.number_input("Sodium", 120.0, 160.0, 140.0)
potassium = st.number_input("Potassium", 2.0, 6.0, 4.0)
hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.0)

st.subheader("Comorbidities")
diabetes = st.selectbox("Diabetes", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
copd = st.selectbox("COPD", [0, 1])
chronic_kidney_disease = st.selectbox("Kidney Disease", [0, 1])

# =====================================================
# PREDICTION & RULES
# =====================================================
if st.button("Predict Sepsis"):

    input_dict = {
        'age': age,
        'weight_kg': weight,
        'height_cm': height,
        'bmi': bmi,
        'hr_mean': hr,
        'sbp_mean': sbp,
        'dbp_mean': dbp,
        'temp_celsius_mean': temp,
        'spo2_mean': spo2,
        'respiratory_rate_mean': resp,
        'wbc': wbc,
        'glucose': glucose,
        'sodium': sodium,
        'potassium': potassium,
        'hemoglobin': hemoglobin,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'copd': copd,
        'chronic_kidney_disease': chronic_kidney_disease,
        'insurance': insurance,
        'gender': gender
    }

    input_df = pd.DataFrame([input_dict])

    # ENCODE categorical
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # SCALE numeric
    input_scaled = scaler.transform(input_df)

    # MODEL PREDICTION
    prediction = model.predict(input_scaled)[0]

    # RULE-BASED OVERRIDE (critical vitals)
    critical_vitals = 0
    if hr > 100: critical_vitals += 1
    if sbp < 90: critical_vitals += 1
    if dbp < 60: critical_vitals += 1
    if temp > 38: critical_vitals += 1
    if spo2 < 92: critical_vitals += 1
    if resp > 20: critical_vitals += 1
    if wbc > 12: critical_vitals += 1

    # If 3 or more critical vitals abnormal, override to high risk
    high_risk = prediction == 1 or critical_vitals >= 3

    # CONDITION ANALYSIS
    conditions = []
    if hr > 100: conditions.append("High Heart Rate")
    if sbp < 90: conditions.append("Low Systolic BP")
    if dbp < 60: conditions.append("Low Diastolic BP")
    if temp > 38: conditions.append("High Temperature")
    if spo2 < 92: conditions.append("Low Oxygen Level")
    if resp > 20: conditions.append("High Respiratory Rate")
    if wbc > 12: conditions.append("High Infection Level")

    # =====================================================
    # OUTPUT
    # =====================================================
    if high_risk:
        st.error("⚠️ High Risk of Sepsis Detected")
        st.subheader("Condition Analysis")
        if conditions:
            for c in conditions:
                st.write("•", c)
        else:
            st.write("Multiple clinical indicators present")

        st.subheader("Precautions")
        st.write("""
        • Consult doctor immediately  
        • Monitor vitals regularly  
        • Maintain hydration  
        • Avoid infections  
        • Hospital admission if needed  
        • Oxygen support if required  
        """)

        st.subheader("Treatment Guidance")
        st.write("""
        • Antibiotics (as prescribed by doctor)  
        • IV fluids  
        • Oxygen support  
        • ICU monitoring  
        • Organ support if required  
        """)
    else:
        st.success("✅ Person is at LOW RISK of Sepsis")
        st.subheader("Health Maintenance Guidance")
        st.write("""
        • Maintain hygiene  
        • Drink enough water  
        • Eat healthy food  
        • Regular medical checkups  
        • Treat infections early  
        • Maintain immunity  
        • Monitor temperature and blood pressure  
        """)

    st.warning("This system provides predictions for informational purposes only. It is not a medical diagnosis. Please consult a doctor.")