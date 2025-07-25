# -*- coding: utf-8 -*-
"""streamlit_app

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NpA23k8u_hHl3Mn6smWNwZukVK2Qh8H3
"""

import streamlit as st

# Page configuration
st.set_page_config(page_title='FarmConsultAI', page_icon='🌾', layout='wide')

# Page links
st.page_link("streamlit_app.py", label="HOME")
st.page_link("pages/crop_yield_prediction.py", label="Crop Yield Prediction")
st.page_link("pages/crop_disease_detection.py", label="Crop Disease Diagnosis")

# Background color and style
background_color = "#F6FFF0"
text_color = "#2E382E"
highlight_color = "#A3C85A"

st.markdown(f"""
    <style>
        body {{
            background-color: {background_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {background_color};
        }}
        .block-container {{
            padding-top: 2rem;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {highlight_color};
        }}
        p, div, label {{
            color: {text_color};
            font-size: 1.1rem;
        }}
    </style>
""", unsafe_allow_html=True)

# Main content
st.title("FarmConsultAI - 3MTT July Knowledge Showcase")
st.header("Empowering Nigerian Farmers with AI")

st.write("""
Welcome to FarmConsultAI, your intelligent assistant for better farming decisions!
We combine advanced AI tools with local agricultural knowledge to help farmers ***diagnose crop diseases*** from images and ***predict crop yields*** using environmental and soil data.

Whether you're a farmer in Kaduna, an agri-entrepreneur in Abia, or a student learning agriculture in Kwara, FarmConsultAI is here to support you every step of the way!
""")

st.subheader("What Can FarmConsultAI Do?")
st.markdown("""
- **Crop Yield Prediction**: Estimate how much your farm will produce based on soil, weather, and nutrient data.
- **Crop Disease Diagnosis**: Upload a photo of a sick crop and get instant analysis and AI-powered treatment advice.
- **Gemini-1.5-flash Assistant**: Get personalized farming advice in simple terms, for free.

> Built for smallholder farmers, cooperatives, and agri-entrepreneurs who want **actionable insights, not buzzwords**.
""")

st.success("Let's help you grow more, waste less, and farm smarter!")

st.markdown("---")
st.caption("Powered by EfficientNet + Scikit-Learn + GPT-3.5 | Built with ❤️ for Nigerian farmers")
