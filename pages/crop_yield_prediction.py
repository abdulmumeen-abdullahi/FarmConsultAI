import streamlit as st
import pickle
import numpy as np
import os
import gdown
import google.generativeai as genai

# ----------------- CONFIG -----------------
st.set_page_config(page_title="FarmConsultAI - Crop Yield Prediction", layout="centered")
st.title("FarmConsultAI - Crop Yield Prediction")
st.write("Estimate your farm's crop yield using soil, weather, and nutrient data.")

# ----------------- SET GEMINI API KEY -----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize chat session if not in state
if "chat" not in st.session_state:
    st.session_state.chat = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])
    st.session_state.last_advice = None
    st.session_state.prediction_context = ""

chat = st.session_state.chat

# ----------------- LOAD MODEL -----------------
model_file_id = "1lDIpOAM4jLx7wbnBlB9360z98twaprvG"
model_url = f"https://drive.google.com/uc?id={model_file_id}"
model_path = "yield_best_random_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            gdown.download(model_url, model_path, quiet=False)
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ----------------- USER INPUTS -----------------
crop_type = st.selectbox("Crop Type", ["Corn", "Potato", "Rice", "Sugarcane", "Wheat"])
soil_type = st.selectbox("Soil Type", ["Clay", "Loamy", "Peaty", "Saline", "Sandy"])
soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
temperature = st.slider("Temperature (°C)", 10, 40, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 15)
n = st.number_input("Nitrogen (N)", min_value=0, value=60)
p = st.number_input("Phosphorus (P)", min_value=0, value=45)
k = st.number_input("Potassium (K)", min_value=0, value=31)
soil_quality = st.number_input("Soil Quality", min_value=0, max_value=100, value=50)

# ----------------- ENCODE INPUT -----------------
crop_encoded = {"Corn": 0, "Potato": 1, "Rice": 2, "Sugarcane": 3, "Wheat": 4}[crop_type]
soil_encoded = {"Clay": 0, "Loamy": 1, "Peaty": 2, "Saline": 3, "Sandy": 4}[soil_type]
input_data = np.array([[crop_encoded, soil_encoded, soil_ph, temperature, humidity,
                        wind_speed, n, p, k, soil_quality]])


# Session state to persist chat
if "chat" not in st.session_state:
    st.session_state.chat = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])

chat = st.session_state.chat

# ----------------- SYSTEM PROMPT -----------------
system_prompt = """
You are FarmConsultAI, a friendly and knowledgeable agricultural consultant, advisor and extension officer in Nigeria.
You assist farmers with expert advice on crop yield, farm conditions, and yield improvement.
Use clear, simple, local, and warm English, like a trusted and real-life rural consultant or farm advisor and extension officer would when advising farmers in Nigeria.
Avoid slang like "E seun" or overly casual expressions. Keep it natural, expert-like, and practical.

RESPONSE FLOW:
1. Greet the farmer respectfully in English (using a farmer-style tone that is warm but professional).
2. Acknowledge the predicted crop and yield and explain what the yield means.
3. Give 3 straightforward recommendations as a seasoned farm consultant would, based on the predicted data.
4. Give one prevention or maintenance tip for sustained yield.
5. Do not ask questions or offer further assistance.
"""

# ----------------- PREDICTION & GEMINI CHAT -----------------
if st.button("Predict Yield"):
    try:
        prediction = model.predict(input_data)
        yield_value = prediction[0]
        st.success(f"**Estimated Crop Yield:** {yield_value:.2f} tons/hectare")

        user_prompt = f"""
A Nigerian farmer is growing {crop_type} with the following:
- Soil: {soil_type}, pH: {soil_ph}, Quality: {soil_quality}/100
- Temp: {temperature}°C, Humidity: {humidity}%, Wind: {wind_speed} km/h
- Nutrients - N: {n}, P: {p}, K: {k}
- Predicted yield: {yield_value:.2f} tons/hectare

As FarmConsultAI, provide:
1. A warm greeting and confirm the predicted yield and crop.
2. What this yield level means for Nigerian farmers.
3. Three practical tips to improve this yield.
4. One prevention or maintenance tip for future consistency and improvement.
Use local, simple and clear terms.
"""

        with st.spinner("Getting expert advice from FarmConsultAI..."):
            chat.send_message(system_prompt)
            gemini_reply = chat.send_message(user_prompt)
            st.session_state.last_advice = gemini_reply.text
            st.session_state.prediction_context = user_prompt

        st.markdown("###FarmConsultAI Advice")
        st.info(st.session_state.last_advice)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ----------------- FOLLOW-UP QUESTION -----------------
if st.session_state.last_advice:
    st.markdown("### 💬 Ask Follow-up Question")
    follow_up = st.text_input("Have any more questions about your farm?", key="followup_input")

    if follow_up:
        with st.spinner("FarmConsultAI is thinking..."):
            followup_reply = chat.send_message(follow_up)
            st.markdown("**FarmConsultAI says:**")
            st.success(followup_reply.text)

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("Powered by Scikit-Learn + Gemini | Built with ❤️ for Nigerian Farmers")
