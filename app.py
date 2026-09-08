# pip install streamlit pandas scikit-learn matplotlib seaborn requests

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# OLLAMA CONFIG
# ==============================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv('fake_job_dataset.csv')

# ==============================
# TRAIN RANDOM FOREST MODEL
# ==============================
def train_random_forest(data):

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'],
        test_size=0.2,
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    st.subheader("📊 Random Forest Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    st.pyplot()

    return model, vectorizer

# ==============================
# OLLAMA CLASSIFICATION
# ==============================
def classify_with_ollama(text):

    prompt = f"""
You are an expert job fraud detection system.

Classify the following job posting as FAKE or REAL.

Respond with ONLY one word:
FAKE or REAL

Job Posting:
{text}

Answer:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()
        return result["response"].strip()
    except:
        return "Ollama connection error"

# ==============================
# STREAMLIT UI
# ==============================
def main():

    st.title("🔍 Fake Job Detection ")

    data = load_data()

    # Train Random Forest once
    model, vectorizer = train_random_forest(data)

    st.subheader("📝 Enter Job Description")
    job_description = st.text_area("")

    if st.button("Classify"):

        if job_description.strip() == "":
            st.warning("Please enter job description.")
            return

        # -------- Random Forest Prediction --------
        job_vec = vectorizer.transform([job_description])
        rf_prediction = model.predict(job_vec)[0]

        # -------- Gemma3 Prediction --------
        gemma_prediction = classify_with_ollama(job_description)

        st.subheader("🔎 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### 🤖 Random Forest")
            if rf_prediction == 1:
                st.error("🚨 FAKE JOB")
            else:
                st.success("✅ REAL JOB")

        with col2:
            st.write("### 🧠 Gemma3 (Ollama)")
            if "FAKE" in gemma_prediction.upper():
                st.error("🚨 FAKE JOB")
            elif "REAL" in gemma_prediction.upper():
                st.success("✅ REAL JOB")
            else:
                st.write(gemma_prediction)

if __name__ == "__main__":
    main()
