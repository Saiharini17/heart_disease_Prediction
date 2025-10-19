import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Load the trained model
# -------------------------------
model = joblib.load('heart_disease_model.pkl')

# -------------------------------
# App Title
# -------------------------------
st.title("❤️ Heart Disease Prediction App")

# -------------------------------
# Create Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏥 Input Data", "ℹ️ Attribute Info", "📊 Prediction", "📈 Visualization"])

# -------------------------------
# Tab 1: Input Data
# -------------------------------
with tab1:
    st.subheader("Enter Patient Details:")
    age = st.number_input("Age", min_value=1, max_value=120, value=52)
    sex = st.selectbox("Sex", options=[0,1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=125)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=212)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", options=[0,1])
    restecg = st.selectbox("Resting ECG Results (0-2)", options=[0,1,2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=168)
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", options=[0,1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment (0-2)", options=[0,1,2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", options=[0,1,2,3])
    thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", options=[1,2,3])

# -------------------------------
# Tab 2: Attribute Info
# -------------------------------
with tab2:
    st.subheader("Explanation of Each Attribute:")
    st.markdown("""
- **Age:** Age of the patient in years  
- **Sex:** 0 = Female, 1 = Male  
- **Chest Pain Type (cp):** 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic  
- **Resting Blood Pressure (trestbps):** in mm Hg  
- **Cholesterol (chol):** Serum cholesterol in mg/dl  
- **Fasting Blood Sugar > 120 mg/dl (fbs):** 0 = No, 1 = Yes  
- **Resting ECG Results (restecg):** 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy  
- **Maximum Heart Rate Achieved (thalach):** Highest heart rate during exercise  
- **Exercise Induced Angina (exang):** 0 = No, 1 = Yes  
- **ST Depression (oldpeak):** ST depression induced by exercise  
- **Slope of ST Segment (slope):** 0 = Upsloping, 1 = Flat, 2 = Downsloping  
- **Number of Major Vessels (ca):** 0-3 colored by fluoroscopy  
- **Thalassemia (thal):** 1 = Normal, 2 = Fixed defect, 3 = Reversible defect
""")

# -------------------------------
# Tab 3: Prediction
# -------------------------------
with tab3:
    st.subheader("Prediction Result")
    if st.button("Predict Heart Disease"):
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                               exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"💔 The person is likely to have Heart Disease. (Risk: {probability*100:.2f}%)")
        else:
            st.success(f"💚 The person is unlikely to have Heart Disease. (Risk: {probability*100:.2f}%)")

# -------------------------------
# Tab 4: Visualization
# -------------------------------
with tab4:
    st.subheader("Prediction Probability & Feature Importance")
    
    # Probability chart
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                           exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prob = model.predict_proba(input_data)[0][1]
    
    st.markdown("**Heart Disease Risk Probability:**")
    fig, ax = plt.subplots()
    ax.bar(["No Disease", "Disease"], [1-prob, prob], color=["green","red"])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    for i, v in enumerate([1-prob, prob]):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
    st.pyplot(fig)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.markdown("**Feature Importance:**")
        fi = model.feature_importances_
        feature_names = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                         "exang","oldpeak","slope","ca","thal"]
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.barh(feature_names, fi)
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)
    else:
        st.info("Feature importance is not available for this model.")
