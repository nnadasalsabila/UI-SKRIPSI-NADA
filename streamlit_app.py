import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import datetime

st.set_page_config(layout="wide", page_title="Dashboard Prediksi Harga Cabai di Jawa Timur")

st.title("🌶️ Dashboard Prediksi Harga Komoditas Cabai di Jawa Timur")
st.markdown("Model: **ARIMA** vs **ARIMAX** | Komoditas: Cabai Rawit, Cabai Keriting, Cabai Merah Besar")

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

# Main logic
if uploaded_file:
    df = pd.read_excel(uploaded_file, parse_dates=['tanggal'])
    df = df.sort_values('tanggal')
    df.set_index('tanggal', inplace=True)

    st.subheader("📊 Visualisasi Dataset")
    komoditas = st.selectbox("Pilih Komoditas", ['cabai_rawit', 'cabai_keriting', 'cabai_merah_besar'])
    st.line_chart(df[komoditas])

    st.subheader("⚙️ Pemodelan dan Evaluasi (ARIMA vs ARIMAX)")

    # Dummy ARIMA dan ARIMAX (ganti dengan parameter terbaikmu)
    train = df[komoditas].iloc[:-30]
    test = df[komoditas].iloc[-30:]

    # ARIMA
    model_arima = ARIMA(train, order=(1, 1, 1)).fit()
    pred_arima = model_arima.forecast(steps=30)
    mape_arima = mean_absolute_percentage_error(test, pred_arima) * 100

    # ARIMAX dengan dummy variabel tanggal (contoh: hari besar dummy)
    df['hari_besar'] = df.index.dayofweek == 6  # Misal minggu dianggap hari besar
    exog_train = df['hari_besar'].iloc[:-30]
    exog_test = df['hari_besar'].iloc[-30:]

    model_arimax = SARIMAX(train, order=(1,1,1), exog=exog_train).fit()
    pred_arimax = model_arimax.predict(start=len(train), end=len(train)+29, exog=exog_test)
    mape_arimax = mean_absolute_percentage_error(test, pred_arimax) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAPE ARIMA (%)", f"{mape_arima:.2f}")
    with col2:
        st.metric("MAPE ARIMAX (%)", f"{mape_arimax:.2f}")

    st.subheader("📈 Visualisasi Hasil Prediksi")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, test.values, label="Aktual", color='black')
    ax.plot(test.index, pred_arima, label="Prediksi ARIMA", linestyle='--')
    ax.plot(test.index, pred_arimax, label="Prediksi ARIMAX", linestyle='--')
    ax.set_title(f"Hasil Prediksi Harga - {komoditas.replace('_', ' ').title()}")
    ax.set_ylabel("Harga")
    ax.set_xlabel("Tanggal")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Silakan upload dataset CSV terlebih dahulu. Dataset harus memiliki kolom 'tanggal' dan komoditas (cabai_rawit, cabai_keriting, cabai_merah_besar).")
