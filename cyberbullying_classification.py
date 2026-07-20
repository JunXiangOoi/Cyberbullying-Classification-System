import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download stopwords if not available
nltk.download('stopwords')


# ==========================
# Load Model
# ==========================

model = joblib.load("tfidf_logistic_regression_model.pkl")


# ==========================
# Text Preprocessing
# ==========================

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    text = str(text)

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # remove hashtag symbol only, keep the word
    text = re.sub(r"#", "", text)

    # remove non-alphabet characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # tokenization by split
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    # stemming
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)



# ==========================
# Streamlit Interface
# ==========================

st.title("Cyberbullying Classification System")


st.write(
    "Enter a text below to classify whether it contains cyberbullying "
    "and identify the cyberbullying type."
)


# User input box
user_text = st.text_area(
    "Enter text:",
    height=150,
    placeholder="Type your text here..."
)


# Proceed button
if st.button("Classify"):

    if user_text.strip() == "":
        st.warning("Please enter some text before proceeding.")

    elif len(user_text.split()) < 3:
        st.warning("Please enter a longer sentence for better classification accuracy.")
        st.stop()

    else:

        # Preprocess input
        cleaned_text = preprocess_text(user_text)


        # Check if text becomes empty after cleaning
        if cleaned_text == "":
            st.warning("The text becomes empty after preprocessing.")

        else:

            # Prediction
            prediction = model.predict([cleaned_text])

            if prediction == "other_cyberbullying":
                prediction = "other cyberbullying"
            
            elif prediction == "not_cyberbullying":
                prediction = "not cyberbullying"


            # Display result
            st.success("Classification Result:")

            st.write(
                f"**Cyberbullying Type: {prediction}**"
            )


            # Optional: show cleaned text
            with st.expander("View Processed Text"):
                st.write(cleaned_text)