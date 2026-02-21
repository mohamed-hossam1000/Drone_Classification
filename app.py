from pyexpat import model
from data_preprocessing import preprocess_image
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from model import build_model
import torchvision.models as models
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("resnet50_model.pth", map_location=device)
NN = build_model(num_classes=3, device=device)
NN.load_state_dict(state_dict)
NN.eval()
dict_classes = {0: 'Aeroplanes', 1: 'Birds', 2: 'Drones'}

st.title("Drone Detector")

st.write("Upload an image please")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img = preprocess_image(img_array)
    with torch.no_grad():
        output = NN(img.to(device))
        output = output.cpu().numpy()
        predicted_class = np.argmax(output, axis=1)
    
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Process Image"):
        # pipeline for processing the image
        st.write(f"Predicted Class: {dict_classes[predicted_class[0]]}")
        if img_array is None:
            raise("Processing pipeline not implemented yet.")