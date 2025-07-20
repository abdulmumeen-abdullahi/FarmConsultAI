import streamlit as st
import pickle
import numpy as np
import os
import gdown
import openai

st.set_page_config(page_title="NaijaFarmConsultAI - Crop Yield Prediction", layout="centered")

# --- Load model from Google Drive if not present ---
file_id = "1QXq2c3Gl2SjK1dM7bIOCvXSxK0hbVXnS"
model_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "crop_yield_best_random_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading crop yield model..."):
            gdown.download(model_url, model_path, quiet=False)
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model. Try again later.\n\nError: {e}")
    st.stop()

# --- UI Input Fields ---
st.title("Crop Yield Prediction")
st.write("Estimate the yield of major crops using soil, weather, and nutrient inputs.")

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

# Encode categories
crop_encoded = {"Corn": 0, "Potato": 1, "Rice": 2, "Sugarcane": 3, "Wheat": 4}[crop_type]
soil_encoded = {"Clay": 0, "Loamy": 1, "Peaty": 2, "Saline": 3, "Sandy": 4}[soil_type]

input_features = np.array([[crop_encoded, soil_encoded, soil_ph, temperature, humidity, wind_speed, n, p, k, soil_quality]])

# --- Prediction & GPT-based One-Time Advice ---
if st.button("Predict Yield"):
    prediction = model.predict(input_features)
    st.success(f"Predicted Crop Yield: **{prediction[0]:.2f} tons/hectare**")

    st.subheader("AI Yield Advice")
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    yield_prompt = (
        f"A Nigerian farmer is growing {crop_type} with the following conditions:\n"
        f"- Soil Type: {soil_type}\n"
        f"- Soil pH: {soil_ph}\n"
        f"- Temperature: {temperature}°C\n"
        f"- Humidity: {humidity}%\n"
        f"- Wind Speed: {wind_speed} km/h\n"
        f"- Nitrogen: {n}, Phosphorus: {p}, Potassium: {k}\n"
        f"- Soil Quality Score: {soil_quality}/100\n"
        f"- Predicted yield: {prediction[0]:.2f} tons/hectare\n\n"
        f"Act like a senior Nigerian agricultural consultant and assistant, give clear, expert advice on this yield, what it means, and 3 practical ways to improve it. "
        f"Do not ask the user any questions or offer further assistance. Keep it brief, local, and focused only on this result."
    )

    with st.spinner("Generating expert advice..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable AI agricultural consultant and assistant for Nigerian farmers. You give advice only once based on data provided."
                },
                {"role": "user", "content": yield_prompt}
            ],
            temperature=0.6,
            max_tokens=500
        )

        advice = response.choices[0].message["content"]
        st.info(advice)

# --- Footer ---
st.markdown("---")
st.caption("Powered by Scikit-Learn + GPT-3.5 | Built with ❤️ for Nigerian farmers")
