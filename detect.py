import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

st.title("🐠 Live Fish Detection App")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your trained model

model = load_model()

st.write("📷 Open your camera below and detect fish in real-time.")

# Use Streamlit's camera input (works on mobile too)
camera_input = st.camera_input("Take a picture of your aquarium")

if camera_input is not None:
    # Read image
    image = Image.open(camera_input)
    img_array = np.array(image)

    # Predict
    results = model.predict(img_array, conf=0.5)

    # Plot detection boxes
    annotated_frame = results[0].plot()

    # Display the result
    st.image(annotated_frame, caption="Detected Fish 🐠", use_column_width=True)
