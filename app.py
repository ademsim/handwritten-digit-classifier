import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps
from sklearn.datasets import load_digits

st.set_page_config(page_title="Handwritten Digit Classifier", page_icon="🔢")

MODEL_PATH = Path(__file__).parent / "digits_knn_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

digits = load_digits()

st.title("🔢 Handwritten Digit Classifier")
st.write("A K-Nearest Neighbors model trained on the scikit-learn digits dataset (8x8 grayscale images).")


def show_and_predict(pixels, true_label=None):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(pixels.reshape(8, 8), cmap="gray_r")
    ax.axis("off")
    st.pyplot(fig)

    x = pixels.reshape(1, -1)
    prediction = int(model.predict(x)[0])
    if true_label is None:
        st.success(f"Prediction: **{prediction}**")
    elif prediction == true_label:
        st.success(f"Prediction: **{prediction}** (true label: {true_label}, correct)")
    else:
        st.error(f"Prediction: **{prediction}** (true label: {true_label}, wrong)")

    proba = model.predict_proba(x)[0]
    st.bar_chart(pd.Series(proba, index=[str(c) for c in model.classes_], name="probability"))


tab_sample, tab_upload = st.tabs(["Sample image", "Upload your own"])

with tab_sample:
    if "index" not in st.session_state:
        st.session_state.index = 0

    if st.button("🎲 Random sample"):
        st.session_state.index = int(np.random.randint(0, len(digits.data)))

    index = st.slider("Sample index", 0, len(digits.data) - 1, key="index")
    show_and_predict(digits.data[index], int(digits.target[index]))

with tab_upload:
    st.caption(
        "Use a dark digit on a light background. The image is converted to 8x8 grayscale, "
        "so simple, centered digits work best. Real-world photos usually do not work well."
    )
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("L")
        img = ImageOps.autocontrast(img).resize((8, 8), Image.LANCZOS)
        pixels = 16 - np.array(img, dtype=float) / 255.0 * 16
        show_and_predict(pixels.flatten())
