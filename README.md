# 🌾 FarmConsultAI – An Intelligent Assistant for Better Farming Decisions

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-yellow.svg)](https://scikit-learn.org/)

<img width="1366" height="768" alt="FarmConsultAI Home Page" src="https://github.com/user-attachments/assets/8ed6f535-065c-4e1f-8149-8e8b59894c0d" />

Agriculture is the backbone of food security, yet millions of farmers—especially in developing regions like Nigeria—struggle with unpredictable weather, crop diseases, and limited access to expert advice. I built FarmConsultAI to tackle these critical challenges by putting the power of AI directly into the hands of farmers.

**🔗 Live Demo:** [**farmconsultai.streamlit.app**](https://farmconsultai.streamlit.app)


## 💡 Project Overview

FarmConsultAI is an intelligent, AI-powered web application designed to help farmers, agropreneurs, and agri-investors make smarter, data-driven decisions. It integrates machine learning, computer vision, and generative AI to provide a holistic farm advisory service.

### Key Features

* **📈 Crop Yield Prediction:** A **Random Forest Regressor** model predicts crop yield in tons/hectare based on soil type, weather conditions, and Nitrogen level.
* **🌿 Crop Disease Diagnosis:** An **EfficientNet-B3** computer vision model detects and classifies diseases from uploaded crop leaf images with high accuracy.
* **💬 AI-Powered Advice:** **Google's Gemini** provides actionable recommendations and conversational follow-ups, explaining complex insights in simple, local-friendly language.
* **🖥️ Interactive Web App:** A user-friendly interface built with **Streamlit** makes these powerful tools accessible on any device with an internet connection.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c35db63d-6207-4099-b245-8bb0eebba0e3" width="45%" alt="Yield Prediction Input">
  <img src="https://github.com/user-attachments/assets/db8b43b1-12f2-4a87-badd-164c4c92bce0" width="45%" alt="Yield Prediction Output">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/34a0acc2-bf8a-41e2-afdf-14ed74cf542e" width="45%" alt="Disease Detection Input">
  <img src="https://github.com/user-attachments/assets/d5f39648-19e8-4391-8538-16b5ab69fd13" width="45%" alt="Disease Detection Output">
</p>


## 🛠️ Tech Stack & Architecture

This project combines multiple AI disciplines into a seamless user experience.

* **Frontend:** `Streamlit`
* **Yield Prediction:** `Scikit-learn`, `Pandas`, `NumPy`
* **Disease Diagnosis:** `PyTorch`, `TIMM (PyTorch Image Models)`
* **Generative AI & Chat:** `Google Generative AI (Gemini)`
* **Deployment:** `Streamlit Community Cloud`

The application follows a simple, effective architecture:
1.  The **Streamlit UI** captures user inputs (form data for yield, images for diseases).
2.  The inputs are processed and fed into the appropriate backend model (**Random Forest** or **EfficientNet-B3**).
3.  The model's prediction is passed to the **Gemini API** along with a structured prompt.
4.  Gemini generates a detailed, easy-to-understand explanation and actionable advice.
5.  The final output and conversational chat interface are rendered back to the user in the Streamlit app.

---

## 🚀 Installation & Usage

### Prerequisites
Before setting up the project locally, ensure you have the following:
* **[Python](https://www.python.org/downloads/)** installed (recommended for PyTorch and Scikit-learn compatibility)
* **Git** installed on your machine
* A **Google Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/))

### Local Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/abdulmumeen-abdullahi/FarmConsultAI.git
    cd FarmConsultAI
    ```
2. **Create a Virtual Environment**
  
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
    ```

3. **Install the required Dependencies**
  
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure your Gemini API key:**
   Create a `.streamlit` directory in the project root and add a `secrets.toml` file inside it:
   ```bash
   mkdir -p .streamlit
   ```
   Open `.streamlit/secrets.toml` and add your API key:
   ```toml
   GEMINI_API_KEY = "YOUR_API_KEY_HERE"
   ```
   > [!IMPORTANT]
   > Ensure `.streamlit/secrets.toml` is listed in your `.gitignore` file so you never accidentally commit your API key to version control.

### Running the Application

1. **Start the Streamlit server** (with your `(venv)` environment still active):
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open the app in your browser:**
   Streamlit will launch the interface automatically, or you can navigate manually to `http://localhost:8501`. From the navigation menu, select either **Crop Yield Prediction** (enter soil, weather, and nitrogen parameters) or **Crop Disease Diagnosis** (upload a clear crop leaf image) to generate predictions and Gemini-powered agronomic advice.

---

## 🎯 Impact & Relevance

FarmConsultAI directly addresses several UN Sustainable Development Goals (SDGs):
* **SDG 1 (No Poverty):** By improving farm productivity and reducing crop losses.
* **SDG 2 (Zero Hunger):** Helping farmers grow more and waste less to feed communities.
* **SDG 13 (Climate Action):** Promoting climate-smart agriculture through informed decision-making.

In Nigeria, where farmers often lack access to timely agronomic expertise, this app bridges a critical knowledge gap. It democratizes access to AI-driven insights that were once out of reach, empowering a new generation of farmers.

---

## 🔗 Project Resources
* **Live Demo:** [**https://farmconsultai.streamlit.app**](https://farmconsultai.streamlit.app)
* **Crop Yield Prediction Sub-Repo:** [Link to GitHub Repository](https://github.com/abdulmumeen-abdullahi/Crops-Yield-Prediction-with-Environmental-Factors-and-Land-Nutrients)
* **Crop Disease Detection Sub-Repo:** [Link to GitHub Repository](https://github.com/abdulmumeen-abdullahi/Crop-Disease-Identification-and-Classification)
