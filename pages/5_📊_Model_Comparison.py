import streamlit as st
import pandas as pd
from utils.style import load_css

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

# =========================
# LOAD GLOBAL CSS
# =========================
load_css()

# =========================
# PAGE TITLE & DESCRIPTION
# =========================
st.title("📊 Perbandingan Performa Model")

st.markdown("""
Halaman ini menyajikan perbandingan performa tiga model klasifikasi genre musik
berdasarkan hasil eksperimen yang telah dilakukan.
""")

st.write("---")

# =========================
# DATA METRICS
# =========================
df_metrics = pd.DataFrame({
    "Model": ["Base MLP", "TabNet", "Residual Neural Network"],
    "Accuracy (%)": [71.85, 75.00, 68.74],
    "Weighted F1-score": [0.72, 0.75, 0.69],
    "Catatan": [
        "Baseline model dengan performa stabil",
        "Performa terbaik pada data tabular",
        "Model kompleks namun kurang optimal"
    ]
})

# =========================
# VISUALIZATION
# =========================
st.subheader("📊 Visualisasi Performa Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Accuracy (%)**")
    st.bar_chart(
        df_metrics.set_index("Model")[["Accuracy (%)"]],
        use_container_width=True
    )

with col2:
    st.markdown("**Weighted F1-score**")
    st.bar_chart(
        df_metrics.set_index("Model")[["Weighted F1-score"]],
        use_container_width=True
    )

st.caption("🔍 TabNet menunjukkan performa tertinggi pada kedua metrik evaluasi.")

st.write("---")

# =========================
# COMPARISON TABLE
# =========================
st.subheader("📋 Tabel Perbandingan Model")
st.dataframe(df_metrics, use_container_width=True)

st.write("---")

# =========================
# INTERPRETATION
# =========================
st.subheader("🧠 Interpretasi Hasil")

st.markdown("""
- **Base MLP** digunakan sebagai baseline dan menunjukkan performa yang cukup baik.
- **TabNet** memberikan performa terbaik dengan akurasi dan F1-score tertinggi,
  menunjukkan keunggulan arsitektur attention pada data tabular.
- **Residual Neural Network** memiliki arsitektur yang lebih kompleks,
  namun tidak memberikan peningkatan performa pada dataset ini.
""")

st.success(
    "📌 **Kesimpulan:** Berdasarkan hasil evaluasi, **TabNet** dipilih sebagai "
    "model terbaik untuk sistem klasifikasi genre musik pada eksperimen ini."
)

# =========================
# NAVIGATION
# =========================
if st.button("⬅️ Kembali ke Beranda"):
    st.switch_page("app.py")
