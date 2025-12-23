import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from utils.style import load_css

# =========================
# GENRE MAP
# =========================
GENRE_MAP = {
    0: "Acoustic/Folk",
    1: "Alt Music",
    2: "Blues",
    3: "Bollywood",
    4: "Country",
    5: "HipHop",
    6: "Indie Alt",
    7: "Instrumental",
    8: "Metal",
    9: "Pop",
    10: "Rock"
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Mukti Layer Perceptron Base – Simulasi Prediksi Genre",
    layout="wide"
)

load_css()

# =========================
# LOAD MODEL ASSETS
# =========================
@st.cache_resource
def load_assets():
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    model = tf.keras.models.load_model("models/model_mlp_base.h5")
    return scaler, encoder, model

scaler, encoder, model = load_assets()

# =========================
# TITLE & NAVIGATION
# =========================
st.title("🧠 Multi Layer Perceptron – Simulasi Prediksi Genre")

if st.button("⬅️ Kembali ke Beranda"):
    st.switch_page("app.py")

st.write("---")

# =========================
# INPUT FORM
# =========================
st.subheader("🎛️ Simulasi Karakteristik Audio")

with st.form("form_mlp"):
    c1, c2 = st.columns(2)

    with c1:
        popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        loudness = st.number_input(
            "Loudness (dB)",
            min_value=-60.0,
            max_value=0.0,
            value=-10.0
        )

    with c2:
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.number_input(
            "Tempo (BPM)",
            min_value=50.0,
            max_value=250.0,
            value=120.0
        )
        key = st.number_input("Key (0–11)", 0, 11, 5)

    submit = st.form_submit_button("🎯 Prediksi Genre")

# =========================
# PREDICTION
# =========================
if submit:
    try:
        # ⚠️ HARUS SESUAI URUTAN SAAT TRAINING
        df_input = pd.DataFrame([{
            "Artist Name": "unknown",
            "Popularity": popularity,
            "danceability": danceability,
            "energy": energy,
            "key": key,
            "loudness": loudness,
            "mode": 1,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": valence,
            "tempo": tempo,
            "time_signature": 4,
            "duration_ms": 210000,
        }])

        # feature engineering
        df_input["energy_loudness"] = df_input["energy"] * df_input["loudness"]
        df_input["acoustic_valence"] = df_input["acousticness"] * df_input["valence"]

        # preprocessing
        X = scaler.transform(encoder.transform(df_input))

        # prediction
        probs = model.predict(X, verbose=0)[0]
        res_idx = int(np.argmax(probs))
        genre_name = GENRE_MAP[res_idx]
        confidence = probs[res_idx] * 100

        # output
        st.success(f"🎶 **Prediksi Genre: {genre_name}**")
        st.progress(float(confidence / 100))
        st.caption(f"Confidence: {confidence:.2f}% | Genre ID: {res_idx}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
