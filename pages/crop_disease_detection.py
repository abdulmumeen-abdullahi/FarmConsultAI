# -*- coding: utf-8 -*-
"""Crop_Disease_Detection"""

import streamlit as st
import torch
import os
import gdown
import timm
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import google.generativeai as genai

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="FarmConsultAI - Crop Disease", layout="centered")

# ----------------- GEMINI API CONFIG -----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ----------------- CONSTANTS -----------------
DISEASE_MODEL_ID = "1O-K4s3tv3WTSouhUksPDA5u6gNQ_d0j1"
DISEASE_MODEL_PATH = "best_crop_disease_model.pt"

DISEASE_CLASSES = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight',
    'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast',
    'Sugarcane__Bacterial_Blight', 'Sugarcane__Healthy', 'Sugarcane__Red_Rot',
    'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust'
]

# ----------------- DOWNLOAD + LOAD MODEL -----------------
def download_model():
    if not os.path.exists(DISEASE_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DISEASE_MODEL_ID}"
        gdown.download(url, DISEASE_MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    download_model()

    model = timm.create_model('efficientnet_b3', pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(in_features=1536, out_features=len(DISEASE_CLASSES)))

    state_dict = torch.load(DISEASE_MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ----------------- IMAGE PREPROCESSING -----------------
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# ----------------- PREDICT DISEASE -----------------
def predict_disease(image_file):
    model = load_model()
    input_tensor = preprocess_image(image_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return DISEASE_CLASSES[predicted.item()]

# ----------------- SYSTEM PROMPT -----------------
system_prompt = """
You are FarmConsultAI — a friendly, experienced crop disease specialist and agricultural extension officer in Nigeria.

You assist Nigerian farmers with simple and expert advice on identifying, treating, and preventing crop diseases. You explain things in clear, farmer-friendly English — like a trusted local consultant speaking to a rural farmer who may not be highly educated, but is hardworking and eager to improve their farm.

Avoid slang like “E seun” or chat-style language. Speak warmly, respectfully, and professionally — like a real-life extension officer or village farm agent.

RESPONSE FLOW:
1. Greet the farmer warmly and acknowledge their effort in checking the crop.
2. Confirm the disease name and explain in very simple terms what it means and how it affects the crop.
3. Give 3 clear, practical treatment or control strategies using local or accessible solutions.
4. Offer one useful prevention or maintenance tip to avoid the disease in the future.
5. Keep your message focused strictly on farming and crop health — no off-topic content.
"""

# ----------------- GEMINI FARM ADVICE -----------------
def get_gemini_advice(disease_name):
    prompt = f"""
    A Nigerian farmer uploaded a photo and the crop disease was diagnosed as **{disease_name}**.

    As FarmConsultAI, explain:
    - What this disease is and how it affects the crop.
    - 3 simple treatments or control strategies.
    - One practical prevention tip.

    Use simple, clear, local Nigerian farmer language and sound like a friendly extension officer.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()}
        ])
        return response.text
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return None

# ----------------- MAIN APP -----------------
def main():
    st.title("FarmConsultAI - Crop Disease Detector")
    st.write("Upload an image of your crop to detect any disease and get instant expert advice.")

    uploaded_file = st.file_uploader("📸 Upload a crop image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)

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
    st.caption("Powered by PyTorch + EfficientNet + Gemini | 🇳🇬 Built with ❤️ for Nigerian Farmers")

if __name__ == "__main__":
    main()
