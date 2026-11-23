# 🌾 FarmConsultAI – An Intelligent Assistant for Better Farming Decisions

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-orange.svg)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9-yellow.svg)](https://scikit-learn.org/)
[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E.svg)](https://huggingface.co/VisionaryQuant)

<img width="1366" height="768" alt="FarmConsultAI Home Page" src="https://github.com/user-attachments/assets/8ed6f535-065c-4e1f-8149-8e8b59894c0d" />

Agriculture is the backbone of food security, yet millions of farmers—especially in developing regions like Nigeria—struggle with unpredictable weather, crop diseases, and limited access to expert advice. I built **FarmConsultAI** to tackle these critical challenges by putting the power of AI directly into the hands of farmers.

**🔗 Live App:** [**farmconsultai.streamlit.app**](https://farmconsultai.streamlit.app/)

---

## 💡 Project Overview

FarmConsultAI is an intelligent, AI-powered web application designed to help farmers, agropreneurs, and agri-investors make smarter, data-driven decisions. It integrates machine learning, computer vision, and generative AI to provide a holistic agricultural advisory service tailored to Nigerian farming contexts.

### Key Features

- **📈 Crop Yield Prediction:** A **Random Forest Regressor** model estimates crop yield in tons/hectare across **5 major crops** (*Corn, Potato, Rice, Sugarcane, and Wheat*) using crop type, soil type, soil pH, temperature, humidity, wind speed, and Nitrogen (N) levels.
- **🌿 Crop Disease Diagnosis:** An **EfficientNet-B3** computer vision model detects and classifies **17 disease and healthy leaf states** across the 5 supported crops via file upload or live camera capture.
- **💬 AI-Powered Agronomic Advice:** **Google's Gemini** (`gemini-3.6-flash` with automatic `gemini-3.5-flash-lite` fallback) translates model predictions into clear, practical recommendations and supports interactive follow-up Q&A grounded in FMARD and IITA guidelines.
- **🖥️ Interactive Web App:** A responsive multi-page interface built with **Streamlit** makes these tools accessible on mobile devices and desktops with an internet connection.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c35db63d-6207-4099-b245-8bb0eebba0e3" width="45%" alt="Yield Prediction Input">
  <img src="https://github.com/user-attachments/assets/db8b43b1-12f2-4a87-badd-164c4c92bce0" width="45%" alt="Yield Prediction Output">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/34a0acc2-bf8a-41e2-afdf-14ed74cf542e" width="45%" alt="Disease Detection Input">
  <img src="https://github.com/user-attachments/assets/d5f39648-19e8-4391-8538-16b5ab69fd13" width="45%" alt="Disease Detection Output">
</p>

---

## 🛠️ Tech Stack & Architecture

- **Frontend:** `Streamlit`
- **Yield Prediction:** `Scikit-learn`, `NumPy`
- **Disease Diagnosis:** `PyTorch`, `Torchvision`, `TIMM (PyTorch Image Models)`, `Pillow`
- **Model Hosting:** `Hugging Face Hub` (`huggingface_hub`)
- **Generative AI & Chat:** `Google GenAI SDK` (`gemini-3.6-flash` with `gemini-3.5-flash-lite` fallback)
- **Deployment:** `Streamlit Community Cloud`

### How It Works

1. **Input Capture:** The **Streamlit UI** collects farm parameters (crop type, soil type, soil pH, temperature, humidity, wind speed, and Nitrogen) or a crop leaf image (via file upload or live camera capture).
2. **Model Inference:** Cached models pulled from **Hugging Face Hub** run inference—a **Random Forest Regressor** estimates yield in tons/hectare, while an **EfficientNet-B3** classifier diagnoses leaf diseases across **5 supported crops** (*Corn, Potato, Rice, Sugarcane, and Wheat*) with a `0.60` confidence threshold to filter out non-leaf images.
3. **Agronomic Advisory:** Predictions are sent to the **Gemini API** with a domain-tuned agricultural extension prompt referencing FMARD and IITA guidelines.
4. **Interactive Follow-Up:** Gemini returns actionable treatment/yield recommendations and powers a stateful follow-up chat interface.

---

## 🚀 Usage & Local Installation

### Use Online (No Setup Required)

You can use the live application directly in your browser at **[https://farmconsultai.streamlit.app/](https://farmconsultai.streamlit.app/)**:
- **Crop Yield Prediction:** Select your crop (*Corn, Potato, Rice, Sugarcane, or Wheat*) and enter soil type, soil pH, temperature, humidity, wind speed, and Nitrogen (N) levels.
- **Crop Disease Diagnosis:** Upload a leaf photo or use the **Capture Image** camera option to diagnose 17 disease/healthy states across the 5 supported crops.

### Prerequisites (For Local Development)

Before setting up the project locally, ensure you have:
- **[Python 3.12](https://www.python.org/downloads/)** (or Python 3.10+) installed
- **Git** installed on your machine
- A **Google Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/))

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abdulmumeen-abdullahi/FarmConsultAI.git](https://github.com/abdulmumeen-abdullahi/FarmConsultAI.git)
   cd FarmConsultAI
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Gemini API key:**
   Create a `.streamlit/secrets.toml` file in the project root:
   ```bash
   mkdir -p .streamlit
   ```
   Open `.streamlit/secrets.toml` and add your API key:
   ```toml
   GEMINI_API_KEY = "YOUR_API_KEY_HERE"
   ```
   > [!IMPORTANT]
   > Ensure `.streamlit/secrets.toml` is listed in your `.gitignore` file so you never accidentally commit your API key to version control.

### Running Locally

1. **Start the Streamlit server** (with your `(venv)` environment active):
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open the app in your browser:**
   Streamlit will open the interface automatically, or you can navigate to `http://localhost:8501` and select a module from the home page or sidebar.

---

## 🎯 Impact & Relevance

FarmConsultAI directly addresses several UN Sustainable Development Goals (SDGs):
- **SDG 1 (No Poverty):** Improving farm productivity and reducing preventable crop losses.
- **SDG 2 (Zero Hunger):** Helping farmers grow more and waste less to strengthen local food systems.
- **SDG 13 (Climate Action):** Promoting climate-smart agriculture through data-driven decision-making.

In Nigeria, where farmers often lack timely access to extension officers, FarmConsultAI bridges a critical knowledge gap by democratizing practical, AI-driven agronomic insights.

---

## 🔗 Project Resources

- **Live Web App:** [**https://farmconsultai.streamlit.app/**](https://farmconsultai.streamlit.app/)
- **Crop Yield Prediction Sub-Repo:** [Crops-Yield-Prediction-with-Environmental-Factors-and-Land-Nutrients](https://github.com/abdulmumeen-abdullahi/Crops-Yield-Prediction-with-Environmental-Factors-and-Land-Nutrients)
- **Crop Disease Detection Sub-Repo:** [Crop-Disease-Identification-and-Classification](https://github.com/abdulmumeen-abdullahi/Crop-Disease-Identification-and-Classification)