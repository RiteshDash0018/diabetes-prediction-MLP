import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS DESIGN
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #e8eaf6 50%, #fce4ec 100%);
    }

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #475569;
        margin-bottom: 30px;
    }

    /* Information box */
    .info-box {
        background: rgba(255, 255, 255, 0.75);
        padding: 18px;
        border-radius: 15px;
        border-left: 6px solid #2563eb;
        margin-bottom: 25px;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    }

    /* Section heading */
    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #172554;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Input boxes */
    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    /* Prediction button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 55px;
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        box-shadow: 0px 6px 15px rgba(37, 99, 235, 0.3);
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0px 8px 20px rgba(124, 58, 237, 0.35);
    }

    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .probability {
        font-size: 22px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #64748b;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

model = tf.keras.models.load_model("diabetes_mlp.keras")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Multilayer Perceptron based Diabetes Risk Prediction</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# INFORMATION BOX
# --------------------------------------------------

st.markdown("""
<div class="info-box">
<b>📋 About this application</b><br>
Enter the patient's health information below.
The trained Machine Learning model will predict whether the patient
is likely to be diabetic or non-diabetic.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "🤰 Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "🩸 Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "💓 Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "📏 Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

with col2:

    insulin = st.number_input(
        "💉 Insulin",
        min_value=0.0,
        max_value=900.0,
        value=125.0,
        step=1.0
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=70.0,
        value=30.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🔍 Predict Diabetes")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    # ----------------------------------------------
    # Replace invalid zero values with training
    # medians
    # ----------------------------------------------

    if glucose == 0:
        glucose = 117.0

    if blood_pressure == 0:
        blood_pressure = 72.0

    if skin_thickness == 0:
        skin_thickness = 29.0

    if insulin == 0:
        insulin = 125.0

    if bmi == 0:
        bmi = 32.3


    # ----------------------------------------------
    # Feature Engineering
    # ----------------------------------------------

    glucose_bmi = glucose * bmi
    glucose_age = glucose * age
    bmi_age = bmi * age

    insulin_log = np.log1p(insulin)

    pedigree_log = np.log1p(diabetes_pedigree)


    # ----------------------------------------------
    # Create input in EXACT training order
    # ----------------------------------------------

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
        glucose_bmi,
        glucose_age,
        bmi_age,
        insulin_log,
        pedigree_log
    ]])


    # ----------------------------------------------
    # Scale input
    # ----------------------------------------------

    input_scaled = scaler.transform(input_data)


    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    probability = model.predict(input_scaled, verbose=0)[0][0]

    prediction = 1 if probability >= 0.5 else 0

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    if prediction == 1:

        confidence = probability * 100

        st.error("⚠️ Diabetic")
        st.write(f"### Prediction Probability: {confidence:.2f}%")
        st.warning("The model predicts a higher likelihood of diabetes.")

    else:

        confidence = (1 - probability) * 100

        st.success("✅ Non-Diabetic")
        st.write(f"### Prediction Probability: {confidence:.2f}%")
        st.info("The model predicts a lower likelihood of diabetes.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    🎓 College Minor Project | Predicting Diabetes with Multilayer Perceptrons
    <br>
    <small>For educational purposes only — not a medical diagnosis.</small>
</div>
""", unsafe_allow_html=True)
