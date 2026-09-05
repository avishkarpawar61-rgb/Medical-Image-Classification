import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

# Class names
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("medical_resnet50.keras")

model = load_model()

# Title
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to predict the class.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Display result
    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")