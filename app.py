import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import pandas as pd

# Page config
st.set_page_config(page_title="🧬 Skin Cancer Classifier", layout="centered")

# Class labels
CLASS_NAMES = {
    0: "Basal Cell Carcinoma (BCC)",
    1: "Benign Keratosis-like Lesions (BKL)",
    2: "Dermatofibroma (DF)",
    3: "Melanoma (MEL)",
    4: "Melanocytic Nevi (NV)",
    5: "Others"
}

# Model paths
MODEL_PATH = "model/EfficientNetV2B0_Light_Image_Split.h5"
MODEL_URL = "https://github.com/mdShihabShorkarSuvo/skin-cancer-classifier/raw/main/model/EfficientNetV2B0_Light_Image_Split.h5"

# Download model if missing
def download_model(path):
    try:
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
        st.success("✅ Model downloaded!")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Download error: {e}")

# Load model with cache
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        download_model(path)
    return tf.keras.models.load_model(path)

# Preprocess image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:  # Remove alpha
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Validate image
def is_skin_like_image(image_array):
    return np.std(image_array) > 0.05

# Sidebar
with st.sidebar:
    st.markdown("## 📝 How to Use")
    st.markdown("""
    1. Upload a clear **skin lesion image** (JPG/PNG).
    2. The model will detect the type of skin cancer.
    3. View the predicted class, confidence, and a probability chart.
    """)
    st.markdown("---")
    st.markdown("📌 **Note**: Low-confidence predictions may require better quality images.")

# Header
st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <h1 style='color:#4CAF50;'>🧬 Skin Cancer Classifier</h1>
        <p>Upload a skin lesion image to classify the type of cancer using AI.</p>
    </div>
""", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Analyzing image..."):
            model = load_model()
            image_array = preprocess_image(image)

            if not is_skin_like_image(image_array):
                st.warning("⚠️ This image doesn’t appear to be a skin lesion. Please upload a valid lesion photo.")
                st.stop()

            prediction = model.predict(image_array)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction))
            predicted_label = CLASS_NAMES.get(predicted_class, "Unknown")

        # Results card
        st.markdown("""
            <div style='background-color: #f0f8ff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color:#4CAF50;'>🔍 Prediction Result</h3>
                <p><b>Predicted Class:</b> <span style='color:#2c3e50;'>{} - {}</span></p>
                <p><b>Confidence:</b> <span style='color:#2c3e50;'>{:.2f}%</span></p>
            </div>
        """.format(predicted_class, predicted_label, confidence * 100), unsafe_allow_html=True)

        if confidence < 0.60:
            st.warning("⚠️ The model is not very confident. Try using a better-quality lesion image.")

        # Probability chart
        st.markdown("### 📊 Class Probability Chart")
        prob_df = pd.DataFrame(prediction[0], index=CLASS_NAMES.values(), columns=["Confidence"])
        st.bar_chart(prob_df)

    except Exception as e:
        st.error(f"🚫 Error: {e}")
else:
    st.info("📷 Please upload a skin lesion image to get started.")

# Footer
st.markdown("""
    <hr>
    <div style='text-align:center'>
        Developed by <b>Md. Shihab Shorkar</b> | Powered by <b>EfficientNetV2B0</b>
    </div>
""", unsafe_allow_html=True)
