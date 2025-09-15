import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from skimage import filters, morphology, exposure

st.set_page_config(page_title="Medical Image Analyzer", layout="wide")
st.title("🩻 Medical Image Analyzer (Beginner Project)")

# Upload image
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("📌 Original X-ray")
    st.image(image, caption="Original", use_container_width=True, channels="GRAY")

    # ---- Preprocessing ----
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    equalized = exposure.equalize_adapthist(blurred, clip_limit=0.03)

    # ---- Edge Detection ----
    edges = cv2.Canny((equalized * 255).astype(np.uint8), 50, 150)

    # ---- Segmentation ----
    thresh = filters.threshold_otsu(equalized)
    binary = equalized > thresh
    cleaned = morphology.remove_small_objects(binary, min_size=500)

    # ---- Highlight Abnormal Regions ----
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[cleaned] = [255, 0, 0]

    # ---- Display Results ----
    st.subheader("📊 Processing Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(blurred, caption="Noise Removed", use_container_width=True, channels="GRAY")
        st.image(equalized, caption="Contrast Enhanced", use_container_width=True, channels="GRAY")

    with col2:
        st.image(edges, caption="Edge Detection", use_container_width=True, channels="GRAY")
        st.image(cleaned, caption="Possible Abnormal Regions (Segmentation)", use_container_width=True, channels="GRAY")

    st.subheader("✅ Final Highlighted Image")
    st.image(overlay, caption="Abnormal Areas Marked in Red", use_container_width=True)