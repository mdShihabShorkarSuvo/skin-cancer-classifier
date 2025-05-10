import streamlit as st
from PIL import Image
import numpy as np
from utils import load_model, preprocess_image, CLASS_NAMES

st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")

st.title("🔬 Skin Cancer Classification App")
st.write("Upload a dermoscopic image. The app will predict one of 6 skin cancer categories.")

# Load model (download if not already present)
model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

    st.success(f"Prediction: **{CLASS_NAMES[class_id]}**")
    st.info(f"Confidence: **{confidence:.2%}**")
