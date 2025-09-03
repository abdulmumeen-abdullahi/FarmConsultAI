# -*- coding: utf-8 -*-
"""FarmConsultAI - Crop Disease Detector"""

import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import google.generativeai as genai
from PIL import Image

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="FarmConsultAI - Crop Disease", layout="centered")

# ----------------- GEMINI API CONFIG -----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ----------------- INIT CHAT SESSION -----------------
if "disease_chat" not in st.session_state:
    st.session_state.disease_chat = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])
    st.session_state.disease_chat_history = []
    st.session_state.last_disease_advice = None

# ----------------- CONSTANTS -----------------
REPO_ID = "VisionaryQuant/5_Crop_Disease_Detection"
MODEL_FILENAME = "best_crop_disease_model.pt"

# Confidence threshold
CONF_THRESHOLD = 0.6

CLASSES = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight',
    'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast',
    'Sugarcane__Bacterial_Blight', 'Sugarcane__Healthy', 'Sugarcane__Red_Rot',
    'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust'
]

# ----------------- DOWNLOAD + LOAD MODEL -----------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)

    model = timm.create_model('efficientnet_b3', pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(in_features=1536, out_features=len(CLASSES)))

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ----------------- IMAGE PREPROCESSING -----------------
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        top_prob, predicted_idx = torch.max(probs, dim=1)

    # If below threshold, say Unknown
    if top_prob.item() < CONF_THRESHOLD:
        return "Unknown", top_prob.item()

    return CLASSES[predicted_idx], top_prob.item()


# ----------------- FORMAT DISEASE LABEL -----------------
def format_disease_label(raw_label):
    label = raw_label.replace("___", " - ").replace("__", " - ").replace("_", " ")
    return label.strip()

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
- One practical prevention tip to avoid future occurrence.

Use simple, clear, local Nigerian farmer language and sound like a friendly extension officer.
"""

    try:
        chat = st.session_state.disease_chat
        chat.send_message(system_prompt)
        reply = chat.send_message(prompt)
        st.session_state.last_disease_advice = reply.text
        st.session_state.disease_chat_history.append(("Farmer", f"My crop has {disease_name}"))
        st.session_state.disease_chat_history.append(("FarmConsultAI", reply.text))
        return reply.text
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return None

# ----------------- MAIN APP -----------------
def main():
    st.title("FarmConsultAI - Crop Disease Detector")
    st.write("Upload an image of your crop to detect any disease and get instant expert advice.")
    st.write("⚠️ NOTE: This app only supports Wheat, Sugarcane, Corn, Rice, and Potato leaves.")

    # File upload option
    uploaded_file = st.file_uploader("📸 Upload a crop image", type=["jpg", "jpeg", "png"])

    # Camera option
    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False

    if st.button("📸 Capture Image"):
        st.session_state.show_camera = True

    captured_img = None
    if st.session_state.show_camera:
        captured_img = st.camera_input("Take a photo")

    if uploaded_file:
        st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)

        if st.button("Diagnose Disease"):
            with st.spinner("Running diagnosis..."):
                prediction, confidence = predict_disease(uploaded_file)
                readable = format_disease_label(prediction)
                if prediction == "Unknown":
                    st.warning("It seems this image is not a crop leaf, and the model could not make a confident prediction.")
                else:
                    st.success(f"Predicted Disease: **{readable}** (Confidence: {confidence:.2f})")
                    with st.spinner("FarmConsultAI is writing advice..."):
                        get_gemini_advice(prediction)
    
    elif captured_img:
        st.image(captured_img, caption="Your Captured Image", use_container_width=True)

        if st.button("Diagnose Disease"):
            with st.spinner("Running diagnosis..."):
                prediction, confidence = predict_disease(captured_img)
                readable = format_disease_label(prediction)
                if prediction == "Unknown":
                    st.warning("It seems this image is not a crop leaf, and the model could not make a confident prediction.")
                else:
                    st.success(f"Predicted Disease: **{readable}** (Confidence: {confidence:.2f})")
                    with st.spinner("FarmConsultAI is writing advice..."):
                        get_gemini_advice(prediction)
        else:
            st.info("📸 Please upload an image or take a photo to start diagnosis.")

    # ----------------- DISPLAY CHAT -----------------
    if st.session_state.last_disease_advice:
        st.subheader("💬 FarmConsultAI Advice & Chat")
        for role, msg in st.session_state.disease_chat_history:
            with st.chat_message("user" if role == "Farmer" else "assistant"):
                st.markdown(msg)

        follow_up = st.chat_input("Ask a follow-up question about this disease")

        if follow_up:
            with st.chat_message("user"):
                st.markdown(follow_up)

            with st.spinner("FarmConsultAI is thinking..."):
                st.session_state.disease_chat_history.append(("Farmer", follow_up))
                reply = st.session_state.disease_chat.send_message(follow_up)
                st.session_state.disease_chat_history.append(("FarmConsultAI", reply.text))

            with st.chat_message("assistant"):
                st.markdown(reply.text)

    st.markdown("---")
    st.caption("Powered by Hugging Face + EfficientNet + Gemini | Built with ❤️ for Nigerian Farmers")

if __name__ == "__main__":
    main()
