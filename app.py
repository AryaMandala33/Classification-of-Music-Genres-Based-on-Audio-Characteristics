import streamlit as st
import pandas as pd
from utils.style import load_css

# =========================
# GENRE MAP (WAJIB ADA DI app.py)
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

st.set_page_config(
    page_title="Classification of Musical Genres",
    layout="wide",
    page_icon="🎵"
)

load_css()


st.title("🎶 Klasifikasi Genre Musik Berdasarkan Karakteristik Audio")
st.markdown(
    "#### Welcome ! Pilih arsitektur model di bawah ini untuk memulai klasifikasi genre musik."
)

st.write("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(    
        '<div class="model-card">'
        '<p class="model-title">🧠 MLP Base</p>'
        '<p>Arsitektur Neural Network standar dengan Dense Layer.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    if st.button("Buka Model MLP", use_container_width=True):
        st.switch_page("pages/1_🧠_MLP_Base.py")

with col2:
    st.markdown(
        '<div class="model-card">'
        '<p class="model-title">🌲 TabNet</p>'
        '<p>Model modern untuk data tabular.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    if st.button("Buka Model TabNet", use_container_width=True):
        st.switch_page("pages/2_🌲_TabNet.py")

with col3:
    st.markdown(
        '<div class="model-card">'
        '<p class="model-title">🧬 Residual NN</p>'
        '<p>Neural Network dengan skip connection.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    if st.button("Buka Model Residual", use_container_width=True):
        st.switch_page("pages/3_🧬_Residual_NN.py")

st.subheader("🎼 Daftar Genre yang Didukung")

genre_df = pd.DataFrame({
    "Genre ID": list(GENRE_MAP.keys()),
    "Nama Genre": list(GENRE_MAP.values())
})

st.dataframe(genre_df, use_container_width=True)

st.markdown("""
  
📌Goals saya untuk mensimulasikan proses **klasifikasi genre musik berdasarkan karakteristik audio**
menggunakan tiga pendekatan model pembelajaran mesin:
**MLP Base**, **TabNet**, dan **Residual Neural Network**.
Pengguna dapat membandingkan bagaimana perbedaan arsitektur model
menghasilkan prediksi genre yang berbeda meskipun menggunakan input audio yang sama.
""")