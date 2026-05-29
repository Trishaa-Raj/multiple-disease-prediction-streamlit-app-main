import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Custom CSS for better UI
st.markdown("""
    <style>
    .result-box { padding: 1rem; border-radius: 10px; font-size: 1.2rem; font-weight: bold; text-align: center; margin-top: 1rem; }
    .positive  { background-color: #ffe0e0; color: #b00020; border: 2px solid #b00020; }
    .negative  { background-color: #e0f7e9; color: #1b5e20; border: 2px solid #1b5e20; }
    .hint      { font-size: 0.75rem; color: #888; }
    </style>
""", unsafe_allow_html=True)

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
@st.cache_resource
def load_models():
    diabetes_model     = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
    parkinsons_model   = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
    return diabetes_model, heart_disease_model, parkinsons_model

diabetes_model, heart_disease_model, parkinsons_model = load_models()

# Sidebar navigation
with st.sidebar:
    selected = option_menu('MedPredict AI',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)
    st.markdown("---")
    st.markdown("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a doctor.")

# ─────────────────────────────────────────────
# HELPER: show result with correct color
# ─────────────────────────────────────────────
def show_result(is_positive, positive_msg, negative_msg, confidence=None):
    css_class = "positive" if is_positive else "negative"
    msg = positive_msg if is_positive else negative_msg
    conf_text = f"<br><small>Confidence: {confidence:.1f}%</small>" if confidence else ""
    st.markdown(f'<div class="result-box {css_class}">{msg}{conf_text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER: validate all inputs are filled & numeric
# ─────────────────────────────────────────────
def validate_inputs(inputs):
    for val in inputs:
        if val.strip() == "":
            return False, "⚠️ Please fill in all fields before predicting."
        try:
            float(val)
        except ValueError:
            return False, f"⚠️ '{val}' is not a valid number. Please enter numeric values only."
    return True, ""

# ─────────────────────────────────────────────
# 1. DIABETES PREDICTION
# ─────────────────────────────────────────────
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')
    st.markdown("Enter the patient's medical details below. Values should come from a blood test report.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', placeholder='e.g. 2')
        st.markdown('<p class="hint">Normal: 0–17</p>', unsafe_allow_html=True)
    with col2:
        Glucose = st.text_input('Glucose Level (mg/dL)', placeholder='e.g. 110')
        st.markdown('<p class="hint">Normal: 70–140</p>', unsafe_allow_html=True)
    with col3:
        BloodPressure = st.text_input('Blood Pressure (mm Hg)', placeholder='e.g. 80')
        st.markdown('<p class="hint">Normal: 60–90</p>', unsafe_allow_html=True)
    with col1:
        SkinThickness = st.text_input('Skin Thickness (mm)', placeholder='e.g. 20')
        st.markdown('<p class="hint">Normal: 10–50</p>', unsafe_allow_html=True)
    with col2:
        Insulin = st.text_input('Insulin Level (μU/mL)', placeholder='e.g. 80')
        st.markdown('<p class="hint">Normal: 0–846</p>', unsafe_allow_html=True)
    with col3:
        BMI = st.text_input('BMI Value', placeholder='e.g. 25.5')
        st.markdown('<p class="hint">Normal: 18.5–40</p>', unsafe_allow_html=True)
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', placeholder='e.g. 0.5')
        st.markdown('<p class="hint">Range: 0.078–2.42</p>', unsafe_allow_html=True)
    with col2:
        Age = st.text_input('Age of the Person', placeholder='e.g. 35')
        st.markdown('<p class="hint">Range: 1–120</p>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        predict_btn = st.button('🔍 Predict')
    with col_btn2:
        if st.button('🔄 Reset'):
            st.rerun()

    if predict_btn:
        inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        valid, error_msg = validate_inputs(inputs)
        if not valid:
            st.warning(error_msg)
        else:
            user_input = [float(x) for x in inputs]
            prediction = diabetes_model.predict([user_input])
            # Confidence score
            try:
                proba = diabetes_model.predict_proba([user_input])[0]
                confidence = proba[prediction[0]] * 100
            except:
                confidence = None
            show_result(
                prediction[0] == 1,
                "🔴 The person is likely DIABETIC",
                "🟢 The person is likely NOT Diabetic",
                confidence
            )

# ─────────────────────────────────────────────
# 2. HEART DISEASE PREDICTION
# ─────────────────────────────────────────────
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')
    st.markdown("Enter the patient's cardiac details. Values from ECG report and blood test.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', placeholder='e.g. 55')
    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)', placeholder='0 or 1')
    with col3:
        cp = st.text_input('Chest Pain Type (0–3)', placeholder='0/1/2/3')
        st.markdown('<p class="hint">0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic</p>', unsafe_allow_html=True)
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', placeholder='e.g. 130')
        st.markdown('<p class="hint">Normal: 90–180 mm Hg</p>', unsafe_allow_html=True)
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dl)', placeholder='e.g. 200')
        st.markdown('<p class="hint">Normal: 100–564</p>', unsafe_allow_html=True)
    with col3:
        fbs = st.text_input('Fasting Blood Sugar >120 mg/dl (1=Yes, 0=No)', placeholder='0 or 1')
    with col1:
        restecg = st.text_input('Resting ECG Results (0–2)', placeholder='0/1/2')
        st.markdown('<p class="hint">0=normal, 1=ST-T abnormal, 2=LV hypertrophy</p>', unsafe_allow_html=True)
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved', placeholder='e.g. 150')
        st.markdown('<p class="hint">Normal: 71–202</p>', unsafe_allow_html=True)
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)', placeholder='0 or 1')
    with col1:
        oldpeak = st.text_input('ST Depression (Oldpeak)', placeholder='e.g. 1.5')
        st.markdown('<p class="hint">Range: 0–6.2</p>', unsafe_allow_html=True)
    with col2:
        slope = st.text_input('Slope of Peak ST Segment (0–2)', placeholder='0/1/2')
    with col3:
        ca = st.text_input('Major Vessels Colored (0–3)', placeholder='0/1/2/3')
    with col1:
        thal = st.text_input('Thal (0=normal, 1=fixed, 2=reversible)', placeholder='0/1/2')

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        predict_btn = st.button('🔍 Predict')
    with col_btn2:
        if st.button('🔄 Reset'):
            st.rerun()

    if predict_btn:
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        valid, error_msg = validate_inputs(inputs)
        if not valid:
            st.warning(error_msg)
        else:
            user_input = [float(x) for x in inputs]
            prediction = heart_disease_model.predict([user_input])
            try:
                proba = heart_disease_model.predict_proba([user_input])[0]
                confidence = proba[prediction[0]] * 100
            except:
                confidence = None
            show_result(
                prediction[0] == 1,
                "🔴 The person is likely to have HEART DISEASE",
                "🟢 The person likely does NOT have Heart Disease",
                confidence
            )

# ─────────────────────────────────────────────
# 3. PARKINSON'S PREDICTION
# ─────────────────────────────────────────────
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")
    st.markdown("Enter voice measurement values. These are obtained from a medical voice recording test.")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', placeholder='e.g. 119.99')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', placeholder='e.g. 157.30')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', placeholder='e.g. 74.99')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)', placeholder='e.g. 0.00784')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', placeholder='e.g. 0.00007')
    with col1:
        RAP = st.text_input('MDVP:RAP', placeholder='e.g. 0.00370')
    with col2:
        PPQ = st.text_input('MDVP:PPQ', placeholder='e.g. 0.00554')
    with col3:
        DDP = st.text_input('Jitter:DDP', placeholder='e.g. 0.01109')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer', placeholder='e.g. 0.04374')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', placeholder='e.g. 0.42600')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3', placeholder='e.g. 0.02182')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5', placeholder='e.g. 0.03130')
    with col3:
        APQ = st.text_input('MDVP:APQ', placeholder='e.g. 0.02971')
    with col4:
        DDA = st.text_input('Shimmer:DDA', placeholder='e.g. 0.06545')
    with col5:
        NHR = st.text_input('NHR', placeholder='e.g. 0.02211')
    with col1:
        HNR = st.text_input('HNR', placeholder='e.g. 21.033')
    with col2:
        RPDE = st.text_input('RPDE', placeholder='e.g. 0.41478')
    with col3:
        DFA = st.text_input('DFA', placeholder='e.g. 0.81528')
    with col4:
        spread1 = st.text_input('spread1', placeholder='e.g. -4.813')
    with col5:
        spread2 = st.text_input('spread2', placeholder='e.g. 0.26648')
    with col1:
        D2 = st.text_input('D2', placeholder='e.g. 2.30144')
    with col2:
        PPE = st.text_input('PPE', placeholder='e.g. 0.28465')

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        predict_btn = st.button("🔍 Predict")
    with col_btn2:
        if st.button('🔄 Reset'):
            st.rerun()

    if predict_btn:
        inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                  Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR,
                  HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        valid, error_msg = validate_inputs(inputs)
        if not valid:
            st.warning(error_msg)
        else:
            user_input = [float(x) for x in inputs]
            prediction = parkinsons_model.predict([user_input])
            try:
                proba = parkinsons_model.predict_proba([user_input])[0]
                confidence = proba[prediction[0]] * 100
            except:
                confidence = None
            show_result(
                prediction[0] == 1,
                "🔴 The person likely HAS Parkinson's Disease",
                "🟢 The person likely does NOT have Parkinson's Disease",
                confidence
            )