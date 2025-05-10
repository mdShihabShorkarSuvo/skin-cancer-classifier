import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")

# Class labels
CLASS_NAMES = {
    0: "Basal Cell Carcinoma (BCC)",
    1: "Benign Keratosis-like Lesions (BKL)",
    2: "Dermatofibroma (DF)",
    3: "Melanoma (MEL)",
    4: "Melanocytic Nevi (NV)",
    5: "Others"
}

# Model path and URL
MODEL_PATH = "model/EfficientNetV2B0_Light_Image_Split.h5"
MODEL_URL = "https://github.com/mdShihabShorkarSuvo/skin-cancer-classifier/raw/main/model/EfficientNetV2B0_Light_Image_Split.h5"

# Function to download the model from GitHub
def download_model(path):
    try:
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
        st.success("Model downloaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")

# Load the model (cached for efficiency)
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        download_model(path)
    return tf.keras.models.load_model(path)

# Image preprocessing
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:  # Remove alpha if present
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# --- UI Layout ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 Skin Cancer Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a skin lesion image to classify the type of skin cancer.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a skin lesion image (JPG/PNG/GIF/BMP)", type=["jpg", "png", "gif", "bmp"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Loading model and processing image..."):
            model = load_model()
            image_array = preprocess_image(image)
            prediction = model.predict(image_array)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction, axis=1)[0])
            predicted_label = CLASS_NAMES.get(predicted_class, "Unknown")

        # Display result
        st.success("✅ Prediction Complete!")
        st.subheader("🔍 Result")
        st.markdown(f"<b>Predicted Class:</b> {predicted_class} - <b>{predicted_label}</b>", unsafe_allow_html=True)
        st.markdown(f"<b>Confidence:</b> {confidence * 100:.2f}%", unsafe_allow_html=True)
        st.progress(confidence)

        # Show all class probabilities
        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame(prediction[0], index=CLASS_NAMES.values(), columns=["Confidence"])
        st.bar_chart(prob_df)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
else:
    st.info("Please upload an image file to begin classification.")

# Footer .................
st.markdown("<hr><center>Developed by <b>Md. Shihab Shorkar</b> | Powered by EfficientNetV2B0</center>", unsafe_allow_html=True)
