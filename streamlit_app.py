import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import datetime

st.set_page_config(layout="wide", page_title="Dashboard Prediksi Harga Cabai di Jawa Timur")

st.title("🌶️ Dashboard Prediksi Harga Komoditas Cabai di Jawa Timur")
st.markdown("Model: **ARIMA** vs **ARIMAX** | Komoditas: Cabai Rawit, Cabai Keriting, Cabai Merah Besar")

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, parse_dates=['tanggal'])
    df = df.sort_values('tanggal')
    df.set_index('tanggal', inplace=True)

    st.subheader("📊 Visualisasi Dataset")
    komoditas = st.selectbox("Pilih Komoditas", ['cabai_rawit', 'cabai_keriting', 'cabai_merah_besar'])
    st.line_chart(df[komoditas])

    st.subheader("⚙️ Pemodelan ARIMAX Otomatis Berdasarkan Signifikansi")

    # Feature dummy hari besar (misalnya hari Minggu dianggap hari besar)
    df['hari_besar'] = df.index.dayofweek == 6

    # Split train-test
    y = df[komoditas]
    x = df[['hari_besar']]

    y_train = y[:-30]
    y_test = y[-30:]
    x_train = x[:-30]
    x_test = x[-30:]

    # Range parameter orde
    p_values = range(0, 5)
    d_values = range(1, 2)
    q_values = range(0, 8)

    results = []

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = SARIMAX(y_train, order=(p, d, q), exog=x_train)
                    model_fit = model.fit(disp=False)
                    pvalues = model_fit.pvalues

                    if all(pvalues < 0.05):
                        results.append({
                            'Order (p,d,q)': (p, d, q),
                            'AIC': model_fit.aic,
                            'p-values': pvalues.to_dict(),
                            'model_fit': model_fit
                        })
                        st.write(f"Order ({p},{d},{q}) - SIGNIFICANT")
                    else:
                        st.write(f"Order ({p},{d},{q}) - NOT SIGNIFICANT")

                except Exception as e:
                    st.warning(f"Error on order ({p},{d},{q}): {e}")
                    continue

    if results:
        signif_df = pd.DataFrame(results)
        best_model = min(results, key=lambda x: x['AIC'])
        best_fit = best_model['model_fit']

        # Forecast
        pred_arimax = best_fit.predict(start=len(y_train), end=len(y_train)+29, exog=x_test)
        mape_arimax = mean_absolute_percentage_error(y_test, pred_arimax) * 100

        st.metric("MAPE ARIMAX Terbaik (%)", f"{mape_arimax:.2f}")

        st.subheader("📈 Visualisasi Hasil Prediksi ARIMAX Terbaik")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_test.index, y_test.values, label="Aktual", color='black')
        ax.plot(y_test.index, pred_arimax, label="Prediksi ARIMAX", linestyle='--')
        ax.set_title(f"Prediksi ARIMAX Terbaik - {komoditas.replace('_', ' ').title()}")
        ax.set_ylabel("Harga")
        ax.set_xlabel("Tanggal")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Tidak ada model ARIMAX yang signifikan pada kombinasi parameter ini.")

else:
    st.info("Silakan upload dataset Excel (.xlsx) terlebih dahulu. Dataset harus memiliki kolom 'tanggal' dan komoditas (cabai_rawit, cabai_keriting, cabai_merah_besar).")
