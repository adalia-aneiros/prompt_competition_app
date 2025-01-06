
# Import libraries
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os
import io
from google.cloud import aiplatform
import google.generativeai as genai

# Load predictive model (neural network)
model_diabetes = load_model('diabetes_predictor.keras') 


api_key = st.secrets["gemini"]["api_key"]

# Configurar la API de Gemini
genai.configure(api_key=api_key)

# Usar el modelo
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to make predictions based on the input of the user and using the predictive model
def make_prediction(input_data):
    input_data = np.array([input_data])
    prediction = model_diabetes.predict([input_data])
    predicted_class = (prediction > 0.5).astype(int)
    return predicted_class

# Define prompts based on prediction result and takes into the account the input data of the user
def get_recommendations(prediction, input_data): 

    gender_binary, age, hypertension, heart_disease, smoking_history_binary, bmi, HbA1c_level, blood_glucose_level = input_data

    if prediction == 1:
        prompt = (
            "You are an AI providing health advice. Provide the disclaimer about this information and remind the user to consult a healthcare provider for more personalized advice. "
            "The patient is at high risk of diabetes, based on their health data: "
            f"BMI: {bmi}, Hemoglobine Level: {HbA1c_level}, Blood Glucose Level: {blood_glucose_level}, Hypertension: {hypertension}, heart_disease: {heart_disease}, smoking_history: {smoking_history_binary}. "
            "Take this data into account and offer them detailed and practical recommendations for managing their condition, "
            "including diet, exercise tips."
        )
    else:
        prompt = (
            "You are an AI providing health advice. Provide the disclaimer about this information and remind the user to consult a healthcare provider for more personalized advice. "
            "The patient is not at high risk of diabetes, based on their health data: "
            f"BMI: {bmi}, Hemoglobine Level: {HbA1c_level}, Blood Glucose Level: {blood_glucose_level}, Hypertension: {hypertension}, heart_disease: {heart_disease}, smoking_history: {smoking_history_binary}. "
            "Take this data into account and offer them general health maintenance recommendations, "
            "including healthy habits for diet, exercise, and mental well-being."
        )
    response = model.generate_content(prompt)
    return response.text 

# Design and questions of the app
st.title('🩺 Diabetes prediction Application')
st.subheader('Please fill out the following questionnaire to obtain information about your health!')

with st.sidebar.expander("ℹ️ About this App"):
    st.write("""
    - **Purpose**: This app predicts your diabetes risk and provides actionable recommendations.
    - **How It Works**: The prediction is based on health metrics such as BMI, blood pressure, and blood glucose level.
    - **Disclaimer**: The results are not a substitute for professional medical advice.
    """)

gender_binary = st.radio('**Which is your gender?:**', ('Male', 'Female'))
age = st.slider('**Which is your age?:**', min_value=18, max_value=90, step=1)
hypertension = st.radio('**Do you have high blood pressure?:**', ('No', 'Yes'))
heart_disease = st.radio('**Do you have heart disease?:**', ('No', 'Yes'))
smoking_history_binary = st.radio('**How is your smoking history?:**', ('Never', 'Former', 'Current', 'Not current', 'Ever'))
bmi = st.slider('**Body mass index:**', min_value=10.0, max_value=50.0, step=0.1)
HbA1c_level = st.slider('**How is your homoglobin level?:**', min_value=3.0, max_value=10.0, step=0.1)
blood_glucose_level = st.slider('**How is your blood glucose level?:**', min_value=50, max_value=200, step=1)


# Change type of responses
def yes_no_to_binary(value):
    return 1 if value == 'Yes' else 0

if gender_binary == 'Female':
    gender_binary = 1
else:
    gender_binary = 0
 
if smoking_history_binary == 'Never':
    smoking_history_binary = 1
elif smoking_history_binary == 'Former':
    smoking_history_binary = 2
elif smoking_history_binary == 'Current':
    smoking_history_binary = 3
elif smoking_history_binary == 'Ever':
    smoking_history_binary = 4
elif smoking_history_binary == 'Not current':
    smoking_history_binary = 5


# Button to trigger prediction and show results and recommendations
if st.button('Predict'):   
    inputs = [
        gender_binary,
        age,
        yes_no_to_binary(hypertension),
        yes_no_to_binary(heart_disease),
        smoking_history_binary,
        bmi,
        HbA1c_level,
        blood_glucose_level
    ]
    
    prediction = make_prediction(inputs)
    if prediction == 1:
        st.warning('You are at high risk for diabetes.')

    else:
        st.success('Your health status is normal.')


    with st.sidebar.expander("**💡 Recommendations**"):
        recommendations = get_recommendations(prediction, inputs)
        st.write(recommendations)

    results_text = f"Prediction: {prediction}\n\nRecommendations:\n{recommendations}"
    st.sidebar.download_button(
    label="📥 Download Results",
    data=results_text,
    file_name="diabetes_results.txt",
    mime="text/plain",
)
