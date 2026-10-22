from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "studentscore_ridge_model.joblib"
DATA_PATH = ROOT / "data" / "StudentPerformanceFactors.csv"

st.set_page_config(page_title="StudentScore AI", page_icon="🎓", layout="centered")

st.title("🎓 StudentScore AI")
st.subheader("Exam Score Prediction")
st.write("Enter student information to estimate an exam score.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Expected: models/studentscore_ridge_model.joblib")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

model = load_model()
df = load_data()

numeric_defaults = {
    "Hours_Studied": (0, 40, 10),
    "Attendance": (0, 100, 80),
    "Sleep_Hours": (0, 24, 8),
    "Previous_Scores": (0, 100, 70),
    "Tutoring_Sessions": (0, 20, 1),
    "Physical_Activity": (0, 20, 5),
}

categorical_columns = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

with st.form("prediction_form"):
    st.markdown("### Academic information")
    values = {}

    left, right = st.columns(2)
    for i, (col, (minimum, maximum, default)) in enumerate(numeric_defaults.items()):
        with (left if i % 2 == 0 else right):
            values[col] = st.number_input(
                col.replace("_", " "),
                min_value=minimum,
                max_value=maximum,
                value=default,
            )

    st.markdown("### Student background")
    left, right = st.columns(2)
    for i, col in enumerate(categorical_columns):
        with (left if i % 2 == 0 else right):
            if col in df.columns:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
            else:
                options = []

            if not options:
                fallback = {
                    "Parental_Involvement": ["Low", "Medium", "High"],
                    "Access_to_Resources": ["Low", "Medium", "High"],
                    "Extracurricular_Activities": ["Yes", "No"],
                    "Motivation_Level": ["Low", "Medium", "High"],
                    "Internet_Access": ["Yes", "No"],
                    "Family_Income": ["Low", "Medium", "High"],
                    "Teacher_Quality": ["Low", "Medium", "High"],
                    "School_Type": ["Public", "Private"],
                    "Peer_Influence": ["Positive", "Neutral", "Negative"],
                    "Learning_Disabilities": ["Yes", "No"],
                    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
                    "Distance_from_Home": ["Near", "Moderate", "Far"],
                    "Gender": ["Male", "Female"],
                }
                options = fallback[col]

            values[col] = st.selectbox(col.replace("_", " "), options)

    submitted = st.form_submit_button("Predict Exam Score", type="primary")

if submitted:
    input_df = pd.DataFrame([values])
    try:
        prediction = float(model.predict(input_df)[0])
        st.success(f"### Predicted Exam Score: {prediction:.2f} / 100")
        st.caption(
            "This is a model estimate, not a guaranteed result. "
            "Predictions depend on the quality and representativeness of the training data."
        )
    except Exception as exc:
        st.error("Prediction failed. Check that the saved model matches the notebook features.")
        st.exception(exc)

st.divider()
st.caption("StudentScore AI | Ridge Regression")
