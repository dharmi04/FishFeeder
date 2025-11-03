import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="🐠 Fish Detector", layout="centered")
st.title("🐟 Real-Time Fish Detection (via Camera)")

# Load model once and cache it
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # make sure best.pt is in the same folder

model = load_model()

st.write("📸 Capture your aquarium image below:")

# Capture image from camera
camera_input = st.camera_input("Take a photo")

if camera_input is not None:
    # Convert image data (bytes) → array for YOLO
    bytes_data = camera_input.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    img_array = np.array(image)

    # Run YOLO prediction
    with st.spinner("Detecting fish... 🐠"):
        results = model.predict(img_array, conf=0.5)

    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Display result
    st.image(annotated_frame, caption="🐠 Detected Fish", use_column_width=True)

    # Count fish
    num_fish = len(results[0].boxes)
    st.success(f"✅ Total Fish Detected: {num_fish}")
