import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "brain_tumor_model.h5")
model = tf.keras.models.load_model(model_path)

st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI image to detect tumor presence.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        st.image(image, caption="Uploaded MRI", use_column_width=True)

        # Preprocess image
        resized = cv2.resize(image, (150, 150)).reshape(1, 150, 150, 1) / 255.0

        # Predict
        prediction = model.predict(resized)[0][0]
        label = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"

        st.markdown("---")
        st.subheader("🔎 Result:")
        if prediction > 0.5:
            st.error(f"⚠️ {label} (Confidence: {prediction:.2f})")
        else:
            st.success(f"✅ {label} (Confidence: {1 - prediction:.2f})")
    else:
        st.warning("⚠️ Could not read the image.")
