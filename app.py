# ============================================
# app.py — Cloud Deployment Version
# This version is for Streamlit Community Cloud (Linux)
# 1. Caches the model with @st.cache_resource for speed.
# 2. Removes all macOS-specific environment hacks.
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 🧩 Environment Fixes
# ============================================
# Force CPU (good practice for non-GPU cloud instances)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                      # Suppress TF logs

# ============================================
# 🧠 Core Libraries
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Disable Streamlit thread conflict (still good practice)
try:
    st.runtime.scriptrunner.add_script_run_ctx = lambda *a, **k: None
except AttributeError:
    pass # In case of newer streamlit versions

# ============================================
# 🎨 Streamlit Config
# ============================================
st.set_page_config(page_title="Depression Risk Predictor", layout="centered")

# ============================================
# 📦 Load Model & Preprocessing Artifacts
# ============================================
MODEL_PATH = "best_model.keras"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

@st.cache_resource
def load_artifacts():
    """Load and cache all artifacts for performance."""
    model = load_model(MODEL_PATH, compile=False)
    model.make_predict_function()  # pre-compile TF graph
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model or preprocessors: {e}")
    st.error("Please ensure 'best_model.keras', 'scaler.pkl', and 'feature_names.pkl' are in the GitHub repository.")
    model_loaded = False

# ============================================
# 🧾 Helper Functions
# ============================================
NUMERIC_COLS = [
    "Age", "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction", "Work/Study Hours", "Financial Stress"
]

def preprocess_input(df, feature_names):
    """Encode categorical and align columns."""
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_encoded

def scale_numeric_columns(df_encoded):
    """Scale numeric columns using trained scaler."""
    scaled_df = df_encoded.copy()
    num_cols = [col for col in df_encoded.columns if col in NUMERIC_COLS]
    if num_cols:
        scaled_df[num_cols] = scaler.transform(df_encoded[num_cols])
    return scaled_df

# ============================================
# 🧠 UI
# ============================================
st.title("🧠 Depression Risk Prediction App")
st.markdown("Predict depression likelihood using Deep Learning 🔍")

mode = st.radio(
    "Choose mode:",
    ["🔹 Manual Input (Single Prediction)", "📂 Upload File (Batch Prediction)"]
)

# ============================================
# 🔹 Manual Input Prediction
# ============================================
if mode == "🔹 Manual Input (Single Prediction)":
    with st.form("manual_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 10, 80, 25)
            city = st.text_input("City", "Hyderabad")
            degree = st.text_input("Degree", "B.Tech")
        with col2:
            work_type = st.selectbox("Occupation Type", ["Student", "Working Professional"])
            profession = st.text_input("Profession", "Engineer")
            sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5–7 hours", "7–9 hours", "More than 9 hours"])
            dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

        academic_pressure = st.slider("Academic Pressure (1–10)", 1, 10, 6)
        work_pressure = st.slider("Work Pressure (1–10)", 1, 10, 6)
        cgpa = st.slider("CGPA (1–10)", 1.0, 10.0, 7.5)
        study_satisfaction = st.slider("Study/Job Satisfaction (1–10)", 1, 10, 7)
        job_satisfaction = st.slider("Overall Life Satisfaction (1–10)", 1, 10, 7)
        work_hours = st.slider("Work/Study Hours per Day", 1, 16, 8)
        financial_stress = st.slider("Financial Stress (1–10)", 1, 10, 5)
        suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])
        family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Depression Risk")

    if submitted and model_loaded:
        try:
            st.info("🧮 Processing input... please wait.")
            progress = st.progress(0)
            time.sleep(0.2)

            input_data = pd.DataFrame({
                "Gender": [gender],
                "Age": [age],
                "City": [city],
                "Working Professional or Student": [work_type],
                "Profession": [profession],
                "Academic Pressure": [academic_pressure],
                "Work Pressure": [work_pressure],
                "CGPA": [cgpa],
                "Study Satisfaction": [study_satisfaction],
                "Job Satisfaction": [job_satisfaction],
                "Sleep Duration": [sleep_duration],
                "Dietary Habits": [dietary_habits],
                "Degree": [degree],
                "Have you ever had suicidal thoughts ?": [suicidal_thoughts],
                "Work/Study Hours": [work_hours],
                "Financial Stress": [financial_stress],
                "Family History of Mental Illness": [family_history]
            })

            input_encoded = preprocess_input(input_data, feature_names)
            progress.progress(40)
            st.info(f"🧾 Model expects {model.input_shape[-1]} features, input now has {input_encoded.shape[1]}")

            scaled_input = scale_numeric_columns(input_encoded)
            progress.progress(70)

            with tf.device("/CPU:0"):
                pred_prob = float(model.predict(scaled_input.values, verbose=0)[0][0])

            pred_class = int(pred_prob >= 0.5)
            progress.progress(100)
            time.sleep(0.3)

            st.subheader("📊 Prediction Result")
            if pred_class == 1:
                st.error(f"⚠️ High Risk of Depression ({pred_prob:.2%} probability)")
            else:
                st.success(f"✅ Low Risk of Depression ({pred_prob:.2%} probability)")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ============================================
# 📂 Batch Prediction (Optimized for Big Files)
# ============================================
elif mode == "📂 Upload File (Batch Prediction)":
    st.info("Upload your CSV file for batch prediction (must match training columns).")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

    if uploaded_file is not None and model_loaded:
        try:
            test_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully with {len(test_df)} rows")
            st.info("🔄 Encoding and scaling data...")

            encoded_df = preprocess_input(test_df, feature_names)
            scaled_df = scale_numeric_columns(encoded_df)

            st.info(f"🧾 Model expects {model.input_shape[-1]} features, input now has {scaled_df.shape[1]}")
            progress = st.progress(0)

            preds = []
            batch_size = 5000
            for i in range(0, len(scaled_df), batch_size):
                batch = scaled_df.iloc[i:i + batch_size].values
                with tf.device("/CPU:0"):
                    batch_pred = model.predict(batch, verbose=0)
                preds.extend(batch_pred.flatten())
                progress.progress(min(int((i + batch_size) / len(scaled_df) * 100), 100))

            preds = np.array(preds)
            pred_classes = (preds >= 0.5).astype(int)

            st.success("✅ Batch prediction completed successfully!")

            submission = pd.DataFrame({
                "id": range(1, len(pred_classes) + 1),
                "Depression": pred_classes
            })

            st.dataframe(submission.head())

            csv_data = submission.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Submission CSV",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error during batch prediction: {e}")

# ============================================
# 🧾 Footer
# ============================================
st.markdown("---")
st.caption("Developed by **Harika Paila** | Powered by Deep Learning 🧠")