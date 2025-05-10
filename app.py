
import streamlit as st
import joblib
import spacy
import re
import subprocess
import importlib.util

model_name = "en_core_web_sm"

try:
    # Try loading the model
    nlp = spacy.load(model_name)
except OSError:
    # If not found, download and then load
    subprocess.run(["python", "-m", "spacy", "download", model_name])
    nlp = spacy.load(model_name)

# Preprocess function
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num and not token.is_space and not token.is_bracket and not token.pos_ in ['SYM']
    ]
    return " ".join(tokens)

# Load model and vectorizer
@st.cache(allow_output_mutation=True)
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
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown("<div class='title'>🩺 Health Condition Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Describe your symptoms or health issue, and we’ll predict a possible condition.</div>", unsafe_allow_html=True)

# Load model and vectorizer
model, vectorizer = load_artifacts()

# Input
user_input = st.text_area("💬 Describe your current symptoms or health issue:", height=150)

# Predict
if st.button("🔍 Predict Condition"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a health description.")
    else:
        clean_text = preprocess_text(user_input)
        vectorized = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized)[0]
        st.success(f"🧠 **Predicted Condition**: `{prediction}`")

# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit and spaCy</div>", unsafe_allow_html=True)
