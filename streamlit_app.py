"""Türkiye çok sektörlü ağlı genel denge modeli — Streamlit uygulaması (giriş noktası)."""
import streamlit as st

st.set_page_config(page_title="Türkiye Çok Sektörlü Model", layout="wide")

nav = st.navigation([
    st.Page("pages/sok_gezgini.py", title="Şok gezgini", default=True),
    st.Page("pages/senaryo_analizi.py", title="Senaryo analizi"),
    st.Page("pages/model_sonuclari.py", title="Model sonuçları"),
    st.Page("pages/metodoloji.py", title="Metodoloji"),
])
nav.run()
