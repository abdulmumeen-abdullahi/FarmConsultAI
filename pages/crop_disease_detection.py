# -*- coding: utf-8 -*-
"""FarmConsultAI - Crop Disease Detector"""

import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from google import genai
from google.genai import types

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="FarmConsultAI - Crop Disease", layout="centered")

# ----------------- GEMINI SETUP -----------------
@st.cache_resource
def get_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=60000,  # 60 seconds, in milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=3,  # includes the first try
                http_status_codes=[429, 500, 503, 504],
            ),
        ),
    )

client = get_client()
GEMINI_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

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
Always consult and reference authoritative Nigerian agricultural bodies and key players—such as the Federal Ministry of Agriculture and Rural Development (FMARD) and the International Institute of Tropical Agriculture (IITA)—before giving advice, to ensure accuracy and local relevance.

IMPORTANT: Use the RESPONSE FLOW above only for the first message after a diagnosis result.
For follow-up questions, answer the question directly and concisely, without repeating the greeting or the full structure.
If the farmer only says thanks or greets you, reply with one or two warm sentences and invite further farming questions.
Always remind farmers to confirm chemical names and dosages with a local extension officer or agro-dealer before use.
"""

# ----------------- INIT CHAT SESSION -----------------
if "disease_chat" not in st.session_state:
    st.session_state.disease_chat = client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    st.session_state.disease_chat_history = []
    st.session_state.last_disease_advice = None

def send_with_fallback(prompt):
    """Send a message; if the main model fails after retries, continue on the fallback model."""
    try:
        return st.session_state.disease_chat.send_message(prompt)
    except Exception:
        history = st.session_state.disease_chat.get_history()
        st.session_state.disease_chat = client.chats.create(
            model=FALLBACK_MODEL,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            history=history,
        )
        return st.session_state.disease_chat.send_message(prompt)

# ----------------- CONSTANTS -----------------
REPO_ID = "VisionaryQuant/5_Crop_Disease_Detection"
MODEL_FILENAME = "best_crop_disease_model.pt"
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
        top_prob, predicted_idx = torch.max(probs, dim=1)

    if top_prob.item() < CONF_THRESHOLD:
        return "Unknown", top_prob.item()

    return CLASSES[predicted_idx.item()], top_prob.item()

# ----------------- FORMAT DISEASE LABEL -----------------
def format_disease_label(raw_label):
    label = raw_label.replace("___", " - ").replace("__", " - ").replace("_", " ")
    return label.strip()

# ----------------- GEMINI FARM ADVICE -----------------
def get_gemini_advice(raw_label, readable_label):
    if "Healthy" in raw_label:
        prompt = f"""
A Nigerian farmer uploaded a photo and the crop was identified as: **{readable_label}**.

As FarmConsultAI, congratulate the farmer, then give:
- 3 simple tips to keep the crop healthy.
- One practical prevention tip against common diseases of this crop.

Use simple, clear, local Nigerian farmer language and sound like a friendly extension officer.
"""
    else:
        prompt = f"""
A Nigerian farmer uploaded a photo and the crop disease was diagnosed as **{readable_label}**.

As FarmConsultAI, explain:
- What this disease is and how it affects the crop.
- 3 simple treatments or control strategies.
- One practical prevention tip to avoid future occurrence.

Use simple, clear, local Nigerian farmer language and sound like a friendly extension officer.
"""

    try:
        reply = send_with_fallback(prompt)
        st.session_state.last_disease_advice = reply.text
        st.session_state.disease_chat_history.append(("Farmer", f"My crop result: {readable_label}"))
        st.session_state.disease_chat_history.append(("FarmConsultAI", reply.text))
        return reply.text
    except Exception as e:
        st.error("❌ The advice service is busy right now. Please try again in a moment.")
        st.caption(f"Details: {e}")
        return None

# ----------------- MAIN APP -----------------
def main():
    st.title("FarmConsultAI - Crop Disease Detector")
    st.write("Upload an image of your crop to detect any disease and get instant expert advice.")
    st.write("⚠️ NOTE: This app only supports Wheat, Sugarcane, Corn, Rice, and Potato leaves.")

    uploaded_file = st.file_uploader("📸 Upload a crop image", type=["jpg", "jpeg", "png"])

    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False

    if st.button("📸 Capture Image"):
        st.session_state.show_camera = True

    captured_img = None
    if st.session_state.show_camera:
        captured_img = st.camera_input("Take a photo")

    # Use the uploaded file first, otherwise the camera photo
    image_file = uploaded_file or captured_img

    if image_file:
        caption = "Your Uploaded Image" if uploaded_file else "Your Captured Image"
        st.image(image_file, caption=caption, use_container_width=True)

        if st.button("Diagnose Disease"):
            with st.spinner("Running diagnosis..."):
                prediction, confidence = predict_disease(image_file)

            readable = format_disease_label(prediction)
            if prediction == "Unknown":
                st.warning("It seems this image is not a crop leaf, and the model could not make a confident prediction.")
            else:
                st.success(f"Predicted: **{readable}** (Confidence: {confidence:.2f})")
                with st.spinner("FarmConsultAI is writing advice..."):
                    get_gemini_advice(prediction, readable)
    else:
        st.info("📸 Please upload an image or take a photo to start diagnosis.")

    # ----------------- DISPLAY CHAT -----------------
    if st.session_state.last_disease_advice:
        st.subheader("💬 FarmConsultAI Advice & Chat")
        for role, msg in st.session_state.disease_chat_history:
            with st.chat_message("user" if role == "Farmer" else "assistant"):
                st.markdown(msg)

        follow_up = st.chat_input("Ask a follow-up question about this result")

        if follow_up:
            with st.chat_message("user"):
                st.markdown(follow_up)

            try:
                with st.spinner("FarmConsultAI is thinking..."):
                    reply = send_with_fallback(follow_up)
                    st.session_state.disease_chat_history.append(("Farmer", follow_up))
                    st.session_state.disease_chat_history.append(("FarmConsultAI", reply.text))

                with st.chat_message("assistant"):
                    st.markdown(reply.text)
            except Exception as e:
                st.error("❌ The advice service is busy right now. Please try again in a moment.")
                st.caption(f"Details: {e}")

    st.markdown("---")
    st.caption("Advice is general guidance. Confirm treatments and dosages with a local extension officer. "
               "Powered by Hugging Face + EfficientNet + Gemini | Built with ❤️ for Nigerian Farmers")

if __name__ == "__main__":
    main()