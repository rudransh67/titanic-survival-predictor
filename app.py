
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.joblib"
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("model.joblib not found. Please place it in the app folder.")
    st.stop()

# Application title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details below to predict "
    "whether the passenger would have survived the Titanic disaster."
)

st.divider()

# Passenger input section
st.subheader("Passenger Information")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0,
    step=1.0
)

# Prediction button
if st.button("Predict Survival", type="primary"):

    # Convert categorical input into numerical value
    sex_value = 1 if sex == "Female" else 0

    # Prepare input data in the same order as training
    input_data = pd.DataFrame(
        [[pclass, sex_value, age, fare]],
        columns=["Pclass", "Sex", "Age", "Fare"]
    )

    try:
        # Predict survival
        prediction = model.predict(input_data)[0]

        # Predict probability
        probabilities = model.predict_proba(input_data)[0]

        # Find probability corresponding to class 1
        survival_index = list(model.classes_).index(1)
        survival_probability = probabilities[survival_index] * 100

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Survival Probability",
            f"{survival_probability:.2f}%"
        )

        if prediction == 1:
            st.success("Prediction: Passenger Survived")
        else:
            st.warning("Prediction: Passenger Did Not Survive")

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

st.caption(
    "Machine Learning Project | Decision Tree Classifier | "
    "Kaggle Titanic Dataset"
)
