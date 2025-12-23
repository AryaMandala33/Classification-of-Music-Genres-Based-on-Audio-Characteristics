import streamlit as st
from utils.style import load_css

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Model Explanation",
    layout="wide"
)

# =========================
# LOAD GLOBAL CSS
# =========================
load_css()

# =========================
# PAGE CONTENT
# =========================
st.title("📘 Penjelasan Model Klasifikasi")

st.markdown(
    """
    Halaman ini menjelaskan arsitektur model machine learning
    yang digunakan dalam sistem klasifikasi genre musik.
    """
)

st.write("---")

# =========================
# MLP BASE
# =========================
st.header("🧠 MLP Base (Multi-Layer Perceptron)")

st.markdown(
    """
    **MLP (Multi-Layer Perceptron)** adalah arsitektur neural network dasar
    yang terdiri dari beberapa *dense layer* dan fungsi aktivasi non-linear.

    **Karakteristik:**
    - Menggunakan arsitektur feedforward
    - Semua fitur diproses secara bersamaan
    - Cocok sebagai *baseline model*

    **Kelebihan:**
    - Arsitektur sederhana
    - Cepat dilatih
    - Mudah diimplementasikan

    **Kekurangan:**
    - Kurang menangkap hubungan kompleks antar fitur
    - Sensitif terhadap skala data
    """
)

st.write("---")

# =========================
# TABNET
# =========================
st.header("🌲 TabNet")

st.markdown(
    """
    **TabNet** adalah model deep learning modern yang dirancang
    khusus untuk data tabular menggunakan mekanisme *attention*.

    Model ini memilih fitur secara dinamis pada setiap langkah keputusan.

    **Karakteristik:**
    - Menggunakan attention mechanism
    - Seleksi fitur adaptif
    - Lebih interpretable dibanding MLP

    **Kelebihan:**
    - Performa baik pada data tabular
    - Memberikan insight pentingnya fitur
    - Mengurangi kebutuhan feature engineering

    **Kekurangan:**
    - Waktu training lebih lama
    - Arsitektur lebih kompleks
    """
)

st.write("---")

# =========================
# RESIDUAL NN
# =========================
st.header("🧬 Residual Neural Network")

st.markdown(
    """
    **Residual Neural Network** menggunakan *skip connection*
    untuk mengatasi permasalahan *vanishing gradient*
    pada jaringan neural yang dalam.

    **Karakteristik:**
    - Memiliki koneksi residual (shortcut)
    - Lebih stabil saat training
    - Cocok untuk arsitektur yang lebih dalam

    **Kelebihan:**
    - Akurasi lebih baik pada data kompleks
    - Training lebih stabil dibanding MLP biasa

    **Kekurangan:**
    - Arsitektur lebih sulit dipahami
    - Overhead komputasi lebih besar
    """
)

st.write("---")

# =========================
# PENUTUP
# =========================
st.info(
    "📌 Setiap model memiliki kelebihan dan kekurangan masing-masing. "
    "Pemilihan model terbaik bergantung pada karakteristik data dan "
    "kebutuhan sistem klasifikasi."
)

if st.button("⬅️ Kembali ke Beranda"):
    st.switch_page("app.py")
