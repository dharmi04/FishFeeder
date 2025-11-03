import streamlit as st
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

st.set_page_config(page_title="🐠 Fish Detector", layout="centered")
st.title("🐟 Real-Time Fish Detection (via Camera)")

# ✅ Load model (no torch_load_args)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.write("📸 Capture your aquarium image below:")

camera_input = st.camera_input("Take a photo")

if camera_input is not None:
    bytes_data = camera_input.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    img_array = np.array(image)

    with st.spinner("Detecting fish... 🐠"):
        results = model.predict(img_array, conf=0.5)

    annotated_frame = results[0].plot()
    st.image(annotated_frame, caption="🐠 Detected Fish", use_column_width=True)

    num_fish = len(results[0].boxes)
    st.success(f"✅ Total Fish Detected: {num_fish}")
