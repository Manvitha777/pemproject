# ================== APP.PY ==================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# ==========================================================
# 🌈 PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Sepsis Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 🌗 DARK MODE
# ==========================================================
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
import plotly.io as pio
pio.templates.default = "plotly_dark" if dark_mode else "plotly_white"

# ==========================================================
# 🎨 CSS STYLING
# ==========================================================
st.markdown("""
<style>
[data-testid="stSidebar"] {background: linear-gradient(to bottom, #006064, #004D40);}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {color: #FFFFFF !important; font-size: 16px; font-weight:600;}
[data-testid="stSidebar"] h2 {color:#E0F7FA !important; font-size:24px; font-weight:800;}
[data-testid="stSidebar"] .stSelectbox div, [data-testid="stSidebar"] input {color:#004D40; font-weight:600;}
[data-testid="stSidebar"] ul li {color:#004D40 !important;}
[data-testid="stSidebar"] button {color:#FFFFFF !important; background-color:#FF8C00 !important; font-size:18px; font-weight:800; border-radius:10px; border:none;}
[data-testid="stSidebar"] button:hover {background-color:#E65100 !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 🎯 LOAD DATA
# ==========================================================
df = pd.read_csv("sepsis.csv")

# Fill missing numeric values with median
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ==========================================================
# 🟦 TOP NAVIGATION
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "overview"

col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("🏠 Overview"): st.session_state.page = "overview"
if col2.button("📊 Dataset"): st.session_state.page = "dataset"
if col3.button("🔍 EDA"): st.session_state.page = "eda"
if col4.button("📈 Metrics"): st.session_state.page = "metrics"
if col5.button("🩺 Prediction"): st.session_state.page = "prediction"

st.markdown("---")

# ==========================================================
# 🏠 OVERVIEW PAGE
# ==========================================================
if st.session_state.page == "overview":
    st.markdown("## 🏠 Sepsis Prediction System Dashboard")
    st.write("""
    This app predicts **Sepsis risk** using **clinical rules**.
    Users can input patient details, vitals, labs, and comorbidities to see risk.
    Features:
    - Rule-based prediction with probability gauge
    - Treatment / Precaution guidance
    - Dataset and EDA visualization
    - Model Metrics (Confusion Matrix, ROC)
    """)

# ==========================================================
# 📊 DATASET PAGE
# ==========================================================
elif st.session_state.page == "dataset":
    st.markdown("## 📊 Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)
    st.write("Dataset shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

# ==========================================================
# 🔍 EDA PAGE
# ==========================================================
elif st.session_state.page == "eda":
    st.markdown("## 🔍 Exploratory Data Analysis")
    fig1 = px.histogram(df, x="sepsis_label", color="sepsis_label",
                        title="Sepsis Label Distribution", barmode="group",
                        color_discrete_sequence=["#4A90E2","#FF6F00"])
    st.plotly_chart(fig1, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "sepsis_label" in numeric_cols: numeric_cols.remove("sepsis_label")
    correlations = df[numeric_cols + ["sepsis_label"]].corr()["sepsis_label"].abs().sort_values(ascending=False)
    top_features = correlations[1:11]  # top 10
    fig2 = px.pie(values=top_features.values, names=top_features.index, title="Top 10 Features Correlated with Sepsis")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# 📈 METRICS PAGE
# ==========================================================
elif st.session_state.page == "metrics":
    st.markdown("## 📈 Model Metrics (Rule-based Approximation)")
    st.write("Since we use a rule-based predictor, this simulates predictions on the dataset.")

    # Rule-based predictions for dataset
    X = df.copy()
    pred_list = []
    for i, row in X.iterrows():
        risk = 0
        if row.get("hr_mean",0) > 90: risk +=1
        if row.get("sbp_mean",0) < 90: risk +=1
        if row.get("temp_celsius_mean",0) > 38: risk +=1
        if row.get("wbc",0) < 4 or row.get("wbc",0) > 11: risk +=1
        if row.get("glucose",0) > 180: risk +=1
        if row.get("hemoglobin",0) < 12: risk +=1
        if row.get("diabetes",0) or row.get("hypertension",0) or row.get("liver_disease",0) or row.get("cancer_active",0): risk +=1
        pred_list.append(1 if risk>=3 else 0)

    y_true = df["sepsis_label"]
    y_pred = np.array(pred_list)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.text("Classification Report:")
    st.text(classification_report(y_true, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig_roc.update_layout(title=f"ROC Curve (AUC={auc_score:.4f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

# ==========================================================
# 🩺 PREDICTION PAGE
# ==========================================================
elif st.session_state.page == "prediction":
    st.markdown("## 🩺 Sepsis Prediction")

    # -------------------- Sidebar Inputs --------------------
    with st.sidebar.expander("🧍 Patient Details", expanded=True):
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
        height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=165.0)
        gender = st.selectbox("Gender", ["Male","Female"])
        insurance = st.selectbox("Insurance", ["Medicare","Private","Self-pay"])

    with st.sidebar.expander("🫀 Vitals", expanded=True):
        hr_mean = st.number_input("Heart Rate", min_value=0, max_value=300, value=80)
        sbp_mean = st.number_input("Systolic BP", min_value=0, max_value=300, value=120)
        dbp_mean = st.number_input("Diastolic BP", min_value=0, max_value=200, value=80)
        temp_celsius_mean = st.number_input("Temperature (C)", min_value=25.0, max_value=45.0, value=36.5)
        spo2_mean = st.number_input("SpO2 (%)", min_value=0, max_value=100, value=98)
        respiratory_rate_mean = st.number_input("Respiratory Rate", min_value=0, max_value=100, value=18)

    with st.sidebar.expander("🧪 Lab Values", expanded=True):
        wbc = st.number_input("WBC", min_value=0.0, value=7.0)
        glucose = st.number_input("Glucose", min_value=0.0, value=100.0)
        hemoglobin = st.number_input("Hemoglobin", min_value=0.0, value=13.5)

    with st.sidebar.expander("⚕️ Disease", expanded=True):
        hypertension = st.selectbox("Hypertension", [0,1], index=0)
        diabetes = st.selectbox("Diabetes", [0,1], index=0)
        liver_disease = st.selectbox("Liver Disease", [0,1], index=0)
        cancer_active = st.selectbox("Active Cancer", [0,1], index=0)

    predict_clicked = st.sidebar.button("🧠 Predict Sepsis")

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
        st.session_state.prob = None

    if predict_clicked:
        # -------------------- Rule-based Sepsis --------------------
        sepsis_risk = 0
        if hr_mean > 90: sepsis_risk += 1
        if sbp_mean < 90: sepsis_risk += 1
        if temp_celsius_mean > 38: sepsis_risk += 1
        if wbc < 4 or wbc > 11: sepsis_risk += 1
        if glucose > 180: sepsis_risk += 1
        if hemoglobin < 12: sepsis_risk += 1
        if diabetes or hypertension or liver_disease or cancer_active: sepsis_risk += 1

        pred = 1 if sepsis_risk >= 3 else 0
        prob = min(sepsis_risk / 7, 1.0)

        st.session_state.prediction = pred
        st.session_state.prob = prob

    # -------------------- Show Prediction --------------------
    if st.session_state.prediction is None:
        st.info("👈 Enter inputs in sidebar and click **Predict Sepsis**")
    else:
        pred = st.session_state.prediction
        prob = st.session_state.prob
        st.success(f"🩺 Predicted Sepsis: **{'Yes' if pred==1 else 'No'}** (Probability: {prob*100:.2f}%)")

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={"text":"Sepsis Probability (%)"},
            gauge={"axis":{"range":[0,100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # Guidance buttons
        st.markdown("### Guidance")
        if pred==1:
            col1, col2, col3 = st.columns(3)
            if col1.button("🧾 Present Condition"):
                st.warning("Patient is at high risk of sepsis. Immediate medical attention recommended.")
            if col2.button("💊 Treatment"):
                st.success("Treatment may include antibiotics, IV fluids, oxygen support, blood pressure support, and ICU care.")
            if col3.button("⚠️ Precautions"):
                st.info("Precautions: Continuous monitoring, maintain hydration, infection control, follow doctor's advice.")
        else:
            col1, col2, col3 = st.columns(3)
            if col1.button("🧾 Health Condition"):
                st.info("Patient is stable. No sepsis detected. Normal monitoring recommended.")
            if col2.button("⚠️ Prevention"):
                st.success("Prevention: Maintain hygiene, regular checkups, control BP and sugar, balanced diet, exercise.")
            if col3.button("💊 Preventive Care"):
                st.info("Preventive care: Proper nutrition, hydration, infection control, regular medical checkups.")

# ==========================================================
# 🟪 FOOTER
# ==========================================================
st.markdown("---")
st.markdown("<p style='text-align:center;font-weight:600;'>Sepsis Prediction System | Streamlit & Clinical Rules 🩺</p>", unsafe_allow_html=True)