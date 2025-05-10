
import streamlit as st
import joblib
import spacy
import re
import spacy.cli


# Load spaCy model
@st.cache_resource
def load_spacy_model():
    # Make sure the model is linked
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        raise e
    return nlp

nlp = load_spacy_model()

# Mapping dictionary
condition_dict = {
    0: "Depression",
    1: "Diabetes, Type 2",
    2: "High Blood Pressure"
}

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
        predicted_condition = condition_dict.get(prediction, "Unknown")
        st.success(f"🧠 **Predicted Condition**: `{predicted_condition}`")

# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit and spaCy</div>", unsafe_allow_html=True)
