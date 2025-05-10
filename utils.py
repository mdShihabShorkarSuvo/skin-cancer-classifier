import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Class labels
CLASS_NAMES = {
    0: "Basal Cell Carcinoma (BCC)",
    1: "Benign Keratosis-like Lesions (BKL)",
    2: "Dermatofibroma (DF)",
    3: "Melanoma (MEL)",
    4: "Melanocytic Nevi (NV)",
    5: "Others"
}

MODEL_PATH = "model/EfficientNetV2B0_Light_Image_Split.h5"

def load_model(path=MODEL_PATH):
    return tf.keras.models.load_model(path)

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)
