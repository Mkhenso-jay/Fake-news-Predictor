import streamlit as st
import joblib
import numpy as np

# Load the saved model pipeline
model_data = joblib.load("experiments/models/RandomForest_tfidf.joblib")
pipeline = model_data['pipeline']
label_encoder = model_data['label_encoder']

# Streamlit app title
st.title("Fake News Detector 📰")
st.write("Type or paste news content below and click 'Predict' to check if it's REAL or FAKE.")

# Text input
user_input = st.text_area("Enter news text here:", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Prediction
        prediction = pipeline.predict([user_input])
        label = label_encoder.inverse_transform(prediction)[0]

        # Confidence (probability)
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            probs = pipeline.predict_proba([user_input])[0]
            conf = np.max(probs) * 100
            st.success(f"Prediction: **{label}** (Confidence: {conf:.2f}%)")
        else:
            # Some classifiers like LinearSVC don't support predict_proba
            st.success(f"Prediction: **{label}**")
