import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("model_raisin.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Prediksi Jenis Kismis (Raisin Classifier)")

st.markdown("""
Masukkan data morfologi kismis berdasarkan hasil pengukuran citra. Sistem akan memprediksi jenis kismis: **Besni** atau **Kecimen**.
""")

# Form input fitur
area = st.number_input("Area")
maj = st.number_input("MajorAxisLength")
minr = st.number_input("MinorAxisLength")
ecc = st.number_input("Eccentricity")
conv = st.number_input("ConvexArea")
ext = st.number_input("Extent")
peri = st.number_input("Perimeter")
ar = st.number_input("AspectRation")
roundness = st.number_input("Roundness")
compactness = st.number_input("Compactness")
sf1 = st.number_input("ShapeFactor1")
sf2 = st.number_input("ShapeFactor2")
sf3 = st.number_input("ShapeFactor3")
sf4 = st.number_input("ShapeFactor4")

if st.button("🔍 Prediksi"):
    input_data = np.array([[area, maj, minr, ecc, conv, ext, peri,
                            ar, roundness, compactness, sf1, sf2, sf3, sf4]])
    
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    label = "Besni" if pred == 0 else "Kecimen"
    st.success(f"Hasil prediksi: **{label}**")
