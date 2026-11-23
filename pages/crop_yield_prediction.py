import streamlit as st
import pickle
import numpy as np
from huggingface_hub import hf_hub_download
from google import genai
from google.genai import types

# ----------------- CONFIG -----------------
st.set_page_config(page_title="FarmConsultAI - Crop Yield Prediction", layout="centered")
st.title("FarmConsultAI - Crop Yield Prediction")
st.write("Estimate your farm's crop yield using soil, weather, and nutrient data.")

# ----------------- GEMINI SETUP -----------------
@st.cache_resource
def get_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=60000,
            retry_options=types.HttpRetryOptions(
                attempts=3,                           # includes the first try
                http_status_codes=[429, 500, 503, 504],
            ),
        ),
    )

client = get_client()
GEMINI_MODEL = "gemini-3.6-flash"

# ----------------- SYSTEM PROMPT -----------------
system_prompt = """
You are FarmConsultAI — a friendly, experienced Nigerian crop yield expert, agricultural consultant and crop advisor.

You help farmers understand the meaning of predicted crop yield and what actions they can take to improve their harvest.
You also explain which factors (like soil pH, nutrients, temperature, humidity, soil type) had the strongest influence on the prediction.

You speak in clear, simple English like a trusted farm extension officer advising local farmers in a village or rural setting. 
Keep the tone warm, practical, and professional — avoid slang like “E seun, E kaaro” and overly casual expressions.

RESPONSE FLOW:
1. Greet the farmer warmly and acknowledge their effort in checking the expected yield.
2. Mention the crop type and explain what the predicted yield means in simple terms (e.g., what it could look like in bags/hectares, and whether it’s good, average, or poor).
3. Give 3 practical and experience-based recommendations to improve or sustain yield, based on general best practices (e.g., soil management, planting method, fertilizer timing, watering).
4. Provide 3 helpful prevention or maintenance tips to secure better results in future harvests.
5. Stay focused only on farming, crops, and agricultural best practices. Do not discuss off-topic issues.
Always consult and reference authoritative Nigerian agricultural bodies and key players—such as the Federal Ministry of Agriculture and Rural Development (FMARD) and the International Institute of Tropical Agriculture (IITA)—before giving advice, to ensure accuracy and local relevance.

IMPORTANT: Use the RESPONSE FLOW above only for the first message after a yield prediction.
For follow-up questions, answer the question directly and concisely, without repeating the greeting or the full structure.
If the farmer only says thanks or greets you, reply with one or two warm sentences and invite further farming questions.
"""

# ----------------- SESSION STATE INIT -----------------
if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    st.session_state.last_advice = None
    st.session_state.prediction_context = ""
    st.session_state.chat_history = []

chat = st.session_state.chat

# ------------ Fall Back ----------------

FALLBACK_MODEL = "gemini-3.5-flash-lite"  # lighter model
def send_with_retry(prompt):
    try:
        return st.session_state.chat.send_message(prompt)
    except Exception:
        history = st.session_state.chat.get_history()
        st.session_state.chat = client.chats.create(
            model=FALLBACK_MODEL,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            history=history,
        )
        return st.session_state.chat.send_message(prompt)


# ----------------- LOAD MODEL -----------------
REPO_ID = "VisionaryQuant/Crop_Yield_Prediction"
MODEL_FILENAME = "crop_yield_best_random_model.pkl"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

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

# ----------------- ENCODE INPUT -----------------
crop_encoded = {"Corn": 0, "Potato": 1, "Rice": 2, "Sugarcane": 3, "Wheat": 4}[crop_type]
soil_encoded = {"Clay": 0, "Loamy": 1, "Peaty": 2, "Saline": 3, "Sandy": 4}[soil_type]
input_data = np.array([[crop_encoded, soil_encoded, soil_ph, temperature, humidity,
                        wind_speed, n]])

# ----------------- PREDICTION & GEMINI ADVICE -----------------
if st.button("Predict Yield"):
    try:
        prediction = model.predict(input_data)
        yield_value = prediction[0]
        st.success(f"**Estimated Crop Yield:** {yield_value:.2f} tons/hectare")

        user_prompt = f"""

A Nigerian farmer is growing {crop_type} with the following:
- Soil: {soil_type}, pH: {soil_ph}
- Temp: {temperature}°C, Humidity: {humidity}%, Wind: {wind_speed} km/h
- Nutrients - N: {n}
- Predicted yield: {yield_value:.2f} tons/hectare

As FarmConsultAI, provide:
1. A warm greeting and confirm the predicted yield and crop.
2. What this yield level means for Nigerian farmers.
3. Which 3 factors (from soil, weather, or nutrients) most influenced this prediction.
4. Three practical tips to improve this yield.
5. Three prevention or maintenance tip for future consistency and improvement.
Use local, simple, and clear terms.
"""
        with st.spinner("Getting expert advice from FarmConsultAI..."):
            gemini_reply = send_with_retry(user_prompt)
            st.session_state.last_advice = gemini_reply.text
            st.session_state.prediction_context = user_prompt
            st.session_state.chat_history.append(("FarmConsultAI", gemini_reply.text))

    except Exception as e:
        st.error("❌ The advice service is busy right now. Please try again in a moment.")
        st.caption(f"Details: {e}")

# ----------------- DISPLAY CHAT HISTORY -----------------
if st.session_state.last_advice:
    st.subheader("💬 FarmConsultAI Conversation")
    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "Farmer" else "assistant"):
            st.markdown(message)

    follow_up = st.chat_input("Got another question about your farm?")

    if follow_up:
        with st.chat_message("user"):
            st.markdown(follow_up)

        try:
            with st.spinner("FarmConsultAI is thinking..."):
                st.session_state.chat_history.append(("Farmer", follow_up))
                # followup_reply = chat.send_message(follow_up)
                followup_reply = send_with_retry(follow_up)
                st.session_state.chat_history.append(("FarmConsultAI", followup_reply.text))

            with st.chat_message("assistant"):
                st.markdown(followup_reply.text)
        except Exception as e:
            st.error(f"❌ Could not get a reply: {e}")

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("Powered by Scikit-Learn + Gemini | Built with ❤️ for Nigerian Farmers")