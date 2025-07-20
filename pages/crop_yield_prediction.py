import streamlit as st
import pickle
import numpy as np
import os
import gdown
import openai
from openai import OpenAI, RateLimitError

# ----------------- CONFIG -----------------
st.set_page_config(page_title="FarmConsultAI - Crop Yield Prediction", layout="centered")

# Load OpenAI Key securely
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ----------------- MODEL FILE IDs -----------------
model_file_id = "1lDIpOAM4jLx7wbnBlB9360z98twaprvG"
model_url = f"https://drive.google.com/uc?id={model_file_id}"
model_path = "crop_yield_model.pkl"

# ----------------- LOADERS -----------------
@st.cache_resource
def load_pickle_file(url, path, desc):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {desc}..."):
            gdown.download(url, path, quiet=False)
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_pickle_file(model_url, model_path, "Crop Yield Model")
except Exception as e:
    st.error(f"Error loading model components. Details: {e}")
    st.stop()

# ----------------- GPT CONTEXT -----------------
system_prompt = [
    {"role": "system", "content": """
        You are FarmConsultAI, a friendly and knowledgeable agricultural consultant and extension officer in Nigeria.
        You assist farmers with expert advice on crop yield, farm conditions, and yield improvement.
        Use simple, local, and warm language, like a trusted rural consultant and extension officer.

        RESPONSE FLOW:
        1. Greet and confirm the predicted yield and crop.
        2. Explain what the yield means.
        3. Recommend 3 ways to improve the yield in this context.
        4. Give one prevention or maintenance tip for sustained yield.
        5. Do not ask questions or offer further assistance.
    """}
]

def get_completions_from_messages(messages, model="gpt-3.5-turbo", stream=True):
    client = OpenAI()
    try:
        return client.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream
        )
    except RateLimitError:
        st.error("Rate limit reached. Please wait a few seconds and try again.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

# ----------------- STREAMLIT APP -----------------
st.title("FarmConsultAI - Crop Yield Prediction")
st.write("Estimate your farm's crop yield using soil, weather, and nutrient data.")

# --- Inputs ---
crop_type = st.selectbox("Crop Type", ["Corn", "Potato", "Rice", "Sugarcane", "Wheat"])
soil_type = st.selectbox("Soil Type", ["Clay", "Loamy", "Peaty", "Saline", "Sandy"])
soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
temperature = st.slider("Temperature (°C)", 10, 40, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 15)
n = st.number_input("Nitrogen (N)", min_value=0, value=60)
p = st.number_input("Phosphorus (P)", min_value=0, value=45)
k = st.number_input("Potassium (K)", min_value=0, value=31)
soil_quality = st.number_input("Soil Quality (0–100)", min_value=0, max_value=100, value=50)

# --- Encode Inputs ---
crop_encoded = {"Corn": 0, "Potato": 1, "Rice": 2, "Sugarcane": 3, "Wheat": 4}[crop_type]
soil_encoded = {"Clay": 0, "Loamy": 1, "Peaty": 2, "Saline": 3, "Sandy": 4}[soil_type]
input_features = np.array([[crop_encoded, soil_encoded, soil_ph, temperature, humidity,
                            wind_speed, n, p, k, soil_quality]])

# --- Predict Button ---
if st.button("Predict Yield"):
    try:
        prediction = model.predict(input_features)
        st.success(f"Predicted Crop Yield: **{prediction[0]:.2f} tons/hectare**")

        # --- GPT Prompt ---
        prompt = f"""
        A Nigerian farmer is growing {crop_type} with the following:
        - Soil: {soil_type}, pH: {soil_ph}, Quality: {soil_quality}/100
        - Temp: {temperature}°C, Humidity: {humidity}%, Wind: {wind_speed} km/h
        - Nutrients - N: {n}, P: {p}, K: {k}
        - Predicted yield: {prediction[0]:.2f} tons/hectare

        As FarmConsultAI, provide:
        1. A warm greeting and confirm the predicted yield and crop.
        2. What this yield level means for Nigerian farmers.
        3. Three practical tips to improve this yield.
        4. One prevention or maintenance tip for future consistency and improvement.
        Use local, simple and clear terms.
        """

        with st.spinner("FarmConsultAI is writing advice..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=system_prompt + [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            advice = response.choices[0].message["content"]
            st.markdown("### FarmConsultAI Advice")
            st.info(advice)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Scikit-Learn + OpenAI | Built with ❤️ for Nigerian Farmers")
