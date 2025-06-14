import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")

# UI Customizations
st.set_page_config(page_title="😷 Face Mask Detector", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #121212;
        color: #ffffff;
    }

    .main {
        background: linear-gradient(135deg, #1e1e1e 0%, #2c2c2c 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #00bcd4;
        text-align: center;
        font-weight: 700;
    }

    .stButton > button {
        background-color: #00bcd4;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        transition: background 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0097a7;
        color: #e0f7fa;
    }

    .stFileUploader {
        background-color: #1e1e1e !important;
        border-radius: 8px;
        padding: 1em;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("## 😷 Face Mask Detection App with YOLOv8")

# Radio: choose between image or video
option = st.radio("Choose input source:", ["Upload Image", "Upload Video"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(np.array(image), imgsz=640)[0]
            annotated_img = results.plot()
            st.image(annotated_img, caption="Detected Output", use_container_width=True)

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame_rgb, imgsz=640)[0]
                annotated_frame = results.plot()

                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()

st.markdown("</div>", unsafe_allow_html=True)
