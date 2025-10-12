# app/streamlit_app.py
"""
Quick Streamlit demo. Run:

streamlit run app/streamlit_app.py --server.port 8501
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from src.utils import load_mlb
import os

st.set_page_config(page_title="Poster Genre Predictor", layout="centered")

MODEL_PATH = st.sidebar.text_input("Model path", "models/resnet_genre.h5")
MLB_PATH = st.sidebar.text_input("MLB path", "data/splits/mlb.pkl")
IMAGE_SIZE = st.sidebar.number_input("Image size", value=224, step=1)

@st.cache_resource
def load_model_and_mlb(model_path, mlb_path):
    model = tf.keras.models.load_model(model_path)
    mlb = load_mlb(mlb_path)
    return model, mlb

st.title("🎬 Movie Poster Genre Predictor")
st.write("Upload a poster and I’ll guess the genres. Multi-label predictions (sigmoid).")

uploaded = st.file_uploader("Upload poster image", type=['png','jpg','jpeg'])
if st.button("Load model"):
    try:
        model, mlb = load_model_and_mlb(MODEL_PATH, MLB_PATH)
        st.success("Model & labels loaded")
    except Exception as e:
        st.error("Load failed: " + str(e))
        model = None
        mlb = None
else:
    model = None
    mlb = None

# try load automatically if model exists
if model is None and os.path.exists(MODEL_PATH) and os.path.exists(MLB_PATH):
    try:
        model, mlb = load_model_and_mlb(MODEL_PATH, MLB_PATH)
    except Exception:
        pass

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption="Uploaded poster", use_column_width=True)
    if model is None:
        st.warning("Model not loaded. Click 'Load model' or ensure paths are correct.")
    else:
        img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        x = np.array(img).astype('float32') / 255.0
        x = np.expand_dims(x, 0)
        preds = model.predict(x)[0]
        top_idx = np.argsort(preds)[::-1]
        st.subheader("Predictions")
        for i in top_idx[:10]:
            st.write(f"{mlb.classes_[i]}: {preds[i]:.3f}")
        st.write("Thresholded (0.5):")
        predicted = [mlb.classes_[i] for i, p in enumerate(preds) if p >= 0.5]
        if predicted:
            st.success(", ".join(predicted))
        else:
            st.info("No genre passed the 0.5 threshold.")
