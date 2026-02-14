import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gc
import requests
import os

from utils import preprocess_image, keep_largest_component

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    layout="centered"
)

st.title("🧠 Brain Tumor Segmentation (UNet Model)")

# --------------------------------------------------
# Ethical & Medical Disclaimer
# --------------------------------------------------
st.warning(
    "⚠️ This application is developed strictly for academic and research purposes.\n\n"
    "It is NOT intended for medical diagnosis, clinical decision-making, or treatment planning.\n\n"
    "Uploaded images are processed in-memory and are NOT stored."
)

# --------------------------------------------------
# Hugging Face Model URL
# --------------------------------------------------
MODEL_URL = "https://huggingface.co/sneha09004/brain-tumor-unet-model/resolve/main/brain_tumor_unet_inference.keras"
MODEL_PATH = "brain_tumor_unet_inference.keras"

# --------------------------------------------------
# Download Model (Only If Not Exists)
# --------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

# --------------------------------------------------
# Load Model (Cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    download_model()
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

model = load_model()

# --------------------------------------------------
# File Upload (In-Memory Only)
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Slice (Grayscale Image)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Load image into memory (NOT saved to disk)
    image = Image.open(uploaded_file)

    st.subheader("Input MRI Slice")
    st.image(image, use_container_width=True)

    # ---------------- Preprocessing ----------------
    input_tensor = preprocess_image(image, target_size=(128, 128))

    # ---------------- Prediction -------------------
    prediction = model.predict(input_tensor)[0, :, :, 0]
    binary_mask = (prediction > 0.5).astype(np.uint8)

    # ---------------- Post-processing --------------
    refined_mask = keep_largest_component(binary_mask)

    # ---------------- Overlay ----------------------
    original_gray = np.array(image.convert("L"))
    original_gray = cv2.resize(original_gray, (128, 128))

    overlay = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
    overlay[refined_mask == 1] = [255, 0, 0]

    # ---------------- Display ----------------------
    st.subheader("Segmentation Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Input MRI**")
    st.image(image, use_container_width=True)

with col2:
    st.markdown("**Predicted Mask**")
    st.image(refined_mask * 255, use_container_width=True)

with col3:
    st.markdown("**Overlay**")
    st.image(overlay, use_container_width=True)


    # ---------------- Memory Cleanup ---------------
    del input_tensor
    del prediction
    del binary_mask
    gc.collect()


