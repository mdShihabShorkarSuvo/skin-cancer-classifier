import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

# Class labels
CLASS_NAMES = {
    0: "Basal Cell Carcinoma (BCC)",
    1: "Benign Keratosis-like Lesions (BKL)",
    2: "Dermatofibroma (DF)",
    3: "Melanoma (MEL)",
    4: "Melanocytic Nevi (NV)",
    5: "Others"
}

# Model path - It will be downloaded from GitHub automatically
MODEL_PATH = "model/EfficientNetV2B0_Light_Image_Split.h5"
MODEL_URL = "https://github.com/mdShihabShorkarSuvo/skin-cancer-classifier/raw/main/model/EfficientNetV2B0_Light_Image_Split.h5"

# Load model from path or GitHub if not found locally
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.write("Model not found locally. Downloading model...")
        download_model(path)
    return tf.keras.models.load_model(path)

# Function to download the model from GitHub
def download_model(path):
    try:
        # Download the model from the GitHub repository
        r = requests.get(MODEL_URL)
        r.raise_for_status()  # Raise an error if the download fails
        with open(path, "wb") as f:
            f.write(r.content)
        st.success("Model downloaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")

# Preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    # Remove alpha channel if it exists
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# App interface
st.title("Skin Cancer Classification")
st.write("Upload an image of a skin lesion and the model will predict the type of skin cancer.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "gif", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None

    if model is not None:
        # Preprocess image
        image_array = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        # Show results
        st.write(f"Prediction: {CLASS_NAMES[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
