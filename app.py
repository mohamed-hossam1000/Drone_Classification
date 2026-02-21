import streamlit as st
import cv2
import numpy as np
from PIL import Image


st.title("Drone Detector")

st.write("Upload an image please")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Process Image"):
        # pipeline for processing the image
        if img_array is None:
            raise("Processing pipeline not implemented yet.")
        st.write("Processing the image... (This is a placeholder for the actual processing pipeline)")