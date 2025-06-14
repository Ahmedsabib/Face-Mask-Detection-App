import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model (replace with your trained model path)
model = YOLO("best.pt")

# UI Customizations
st.set_page_config(page_title="😷 Face Mask Detector", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
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

    .stFileUploader, .stRadio, .stCheckbox {
        background-color: #1e1e1e !important;
        border-radius: 8px;
        padding: 1em;
    }

    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("## 😷 Face Mask Detection App with YOLOv8")

# Upload image or use webcam
option = st.radio("Choose input source:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(np.array(image), imgsz=640)[0]
            annotated_img = results.plot()
            st.image(annotated_img, caption="Detected Output", use_column_width=True)

elif option == "Use Webcam":
    st.warning("Click the checkbox below to activate webcam.")
    run_webcam = st.checkbox("Start Webcam")

    if run_webcam:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame_rgb, imgsz=640)[0]
            annotated_frame = results.plot()

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()

st.markdown("</div>", unsafe_allow_html=True)
