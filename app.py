import streamlit as st
import joblib
import re


# Mapping dictionary
condition_dict = {
    0: "Depression",
    1: "Diabetes, Type 2",
    2: "High Blood Pressure"
}

# Load model and vectorizer
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Page config
st.set_page_config(page_title="Health Condition Predictor", layout="centered")

# Custom style
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 2.2em;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub {
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .footer {
        font-size: 0.8rem;
        text-align: center;
        margin-top: 3rem;
        color: #999;
    }
    .description-box {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b3d9ff;
        font-size: 1.1rem;
        color: #333;
        margin-bottom: 2rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .description-box h3 {
        color: #0066cc;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .description-box ul {
        margin-top: 0;
        list-style-type: disc;
        margin-left: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown("<div class='title'>🩺 Health Condition Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Describe your symptoms or health issue, and we’ll predict a possible condition.</div>", unsafe_allow_html=True)

# Information about types of diseases
st.markdown("""
    <div class='description-box'>
        <h3>Conditions the model can predict:</h3>
        <p>This health condition predictor can identify the following conditions based on the symptoms you describe:</p>
        <ul>
            <li><strong>Depression</strong>: A mood disorder causing persistent feelings of sadness and loss of interest.</li>
            <li><strong>Diabetes, Type 2</strong>: A chronic condition affecting how your body processes blood sugar (glucose).</li>
            <li><strong>High Blood Pressure</strong>: A condition where the force of the blood against the artery walls is too high.</li>
        </ul>
        <p>Please provide a detailed description of your symptoms, and we'll predict a possible condition.</p>
    </div>
    """, unsafe_allow_html=True)

# Load model and vectorizer
model, vectorizer = load_artifacts()

# Input
user_input = st.text_area("💬 Describe your current symptoms or health issue:", height=150)

# Predict
if st.button("🔍 Predict Condition"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a health description.")
    else:
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        predicted_condition = condition_dict.get(prediction, "Unknown")
        st.success(f"🧠 **Predicted Condition**: `{predicted_condition}`")

# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit and spaCy</div>", unsafe_allow_html=True)
