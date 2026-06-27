import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

## Load the trained Model
model=tf.keras.models.load_model('model.h5')

## Load the encoders and scalers
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## Streamlit App

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="centered"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
 
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
 
/* Main card */
.main-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
 
/* Section headers */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}
 
/* Title */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    margin-bottom: 0rem;
}
 
/* Result cards */
.result-safe {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #34d399;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-danger {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #f87171;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.5rem 0 0.25rem;
}
.result-sub {
    color: rgba(255,255,255,0.7);
    font-size: 0.9rem;
}
.result-prob {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}
 
/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
}
 
/* Override Streamlit widget label colors */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
 
/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}
 
/* Number input */
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}
 
/* Slider */
.stSlider > div > div > div > div {
    background: #7c3aed !important;
}
 
/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}
 
/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 

st.markdown("""
<div class="main-card">
    <div class="hero-title">🔮 Churn Predictor</div>
    <p class="hero-sub">Fill in the customer details below to predict the likelihood of churn using our trained ANN model.</p>
</div>
""", unsafe_allow_html=True)
 
 

st.markdown('<div class="main-card">', unsafe_allow_html=True)
 

st.markdown('<div class="section-label">👤 Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
with col2:
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
 
col3, col4 = st.columns(2)
with col3:
    age = st.slider('Age', 18, 92, 35)
with col4:
    tenure = st.slider('Tenure (Years)', 0, 10, 3)
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
 

st.markdown('<div class="section-label">💰 Financial Details</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
with col6:
    balance = st.number_input('Balance ($)', min_value=0.0, value=0.0, format="%.2f")
 
estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0, format="%.2f")
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
 

st.markdown('<div class="section-label">🏦 Account Details</div>', unsafe_allow_html=True)
col7, col8, col9 = st.columns(3)
with col7:
    num_of_products = st.slider('No. of Products', 1, 4, 1)
with col8:
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col9:
    is_active_member = st.selectbox('Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
 
st.markdown('</div>', unsafe_allow_html=True)
 

predict_btn = st.button("🔮  Predict Churn")
 
if predict_btn:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
 
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
 
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
 
    input_data_scaled = scaler.transform(input_data)
 
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]
 
    
    st.markdown("<br>", unsafe_allow_html=True)
 
    if prediction_prob > 0.5:
        st.markdown(f"""
        <div class="result-danger">
            <div style="font-size:2.5rem">⚠️</div>
            <div class="result-prob">{prediction_prob:.0%}</div>
            <div class="result-title">High Churn Risk</div>
            <div class="result-sub">This customer is likely to leave. Consider retention strategies.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div style="font-size:2.5rem">✅</div>
            <div class="result-prob">{prediction_prob:.0%}</div>
            <div class="result-title">Low Churn Risk</div>
            <div class="result-sub">This customer is likely to stay. Keep up the good service!</div>
        </div>
        """, unsafe_allow_html=True)