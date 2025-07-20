# -*- coding: utf-8 -*-
"""Crop_Disease_Detection with Gemini"""

import streamlit as st
import torch
import os
import gdown
from PIL import Image
import torchvision.transforms as transforms
import google.generativeai as genai

# ----------------- CONFIG -----------------
st.set_page_config(page_title="FarmConsultAI - Crop Disease", layout="centered")

# ----------------- API KEY -----------------
# Load Gemini API key securely
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ----------------- MODEL PATHS -----------------
DISEASE_MODEL_ID = "1O-K4s3tv3WTSouhUksPDA5u6gNQ_d0j1"
DISEASE_MODEL_PATH = "best_crop_disease_model.pt"

DISEASE_CLASSES = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight',
    'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast',
    'Sugarcane__Bacterial_Blight', 'Sugarcane__Healthy', 'Sugarcane__Red_Rot',
    'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust'
]

# ----------------- MODEL LOADER -----------------
def download_model():
    if not os.path.exists(DISEASE_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DISEASE_MODEL_ID}"
        gdown.download(url, DISEASE_MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    download_model()
    model = torch.load(DISEASE_MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    return model

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def predict_disease(image_file):
    model = load_model()
    input_tensor = preprocess_image(image_file)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    return DISEASE_CLASSES[predicted.item()]

# ----------------- GEMINI AI RESPONSE -----------------
def get_gemini_advice(disease_name):
    prompt = f"""
    A Nigerian farmer uploaded a photo and the crop disease was diagnosed as **{disease_name}**.

    As FarmConsultAI, explain:
    - What this disease is and how it affects the crop.
    - 2 or 3 simple treatments or control strategies.
    - One practical prevention tip.

    Use simple, clear, local Nigerian farmer language and sound like a friendly extension officer.
    """

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return None

# ----------------- STREAMLIT APP -----------------
def main():
    st.title("FarmConsultAI - Crop Disease")
    st.write("Upload an image of your crop to detect disease and get instant expert advice.")

    uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Your Uploaded Image", use_column_width=True)

        if st.button("Diagnose Disease"):
            with st.spinner("Running diagnosis..."):
                disease_prediction = predict_disease(uploaded_file)
            st.success(f"Predicted Disease: **{disease_prediction}**")

            with st.spinner("FarmConsultAI is writing advice..."):
                advice = get_gemini_advice(disease_prediction)
                if advice:
                    st.markdown("### FarmConsultAI Advice")
                    st.info(advice)

    st.markdown("---")
    st.caption("Powered by PyTorch + EfficientNet + Gemini | Built with ❤️ for Nigerian Farmers")

if __name__ == "__main__":
    main()
