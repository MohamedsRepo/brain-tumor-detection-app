import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "brain_tumor_model.h5")
model = tf.keras.models.load_model(model_path)

st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI image to detect tumor presence.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize and preprocess
    image_resized = image.resize((150, 150))
    img_array = np.array(image_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"

    st.markdown("---")
    st.subheader("🔎 Result:")
    if prediction > 0.5:
        st.error(f"⚠️ {label} (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ {label} (Confidence: {1 - prediction:.2f})")
