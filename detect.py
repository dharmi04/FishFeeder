import streamlit as st
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

# 🐠 Page setup
st.set_page_config(page_title="🐠 Fish Detector", layout="centered")
st.title("🐟 Real-Time Fish Detection")

# 🧠 Allow YOLO DetectionModel in new PyTorch versions (2.6+ fix)
try:
    import ultralytics.nn.tasks as tasks
    torch.serialization.add_safe_globals([tasks.DetectionModel])
except Exception as e:
    st.warning("Safe loading patch applied for YOLO model compatibility.")

# 🚀 Load model once and cache it
@st.cache_resource
def load_model():
    return YOLO("best.pt", torch_load_args={"weights_only": False})

model = load_model()

st.write("📸 Capture your aquarium image below:")

# 🎥 Take a photo using device camera
camera_input = st.camera_input("Take a photo")

if camera_input is not None:
    # Convert to image array
    bytes_data = camera_input.getvalue()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    img_array = np.array(image)

    # YOLO prediction
    with st.spinner("Detecting fish... 🐠"):
        results = model.predict(img_array, conf=0.5)

    # Annotate results
    annotated_frame = results[0].plot()

    # Show image and count
    st.image(annotated_frame, caption="🐠 Detected Fish", use_column_width=True)
    num_fish = len(results[0].boxes)
    st.success(f"✅ Total Fish Detected: {num_fish}")
