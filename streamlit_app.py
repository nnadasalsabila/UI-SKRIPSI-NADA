import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(layout="wide", page_title="Dashboard Prediksi Harga Cabai")

st.title("🌶️ Dashboard Prediksi Harga Cabai dengan ARIMAX")
st.markdown("Prediksi harga berdasarkan variabel eksogen: **Idul Adha** dan **Natal**")

# Sidebar upload
st.sidebar.header("📁 Upload File Excel")
uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Baca file
    df = pd.read_excel(uploaded_file)

    # Cek apakah kolom penting tersedia
    required_cols = ['Tanggal', 'Harga', 'Idul Adha', 'Natal']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ File harus memiliki kolom: {', '.join(required_cols)}")
        st.stop()

    # Format datetime dan set index
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal')
    df.set_index('Tanggal', inplace=True)

    st.subheader("📊 Visualisasi Harga Cabai")
    st.line_chart(df['Harga'])

    # Split berdasarkan tanggal tertentu
    split_date = '2024-12-25'

    y = df['Harga']
    x = df[['Idul Adha', 'Natal']]

    y_train = y.loc[y.index < split_date]
    y_test = y.loc[y.index >= split_date]
    x_train = x.loc[x.index < split_date]
    x_test = x.loc[x.index >= split_date]

    st.write("Jumlah data y_train:", len(y_train))
    st.write("Jumlah data y_test :", len(y_test))
    st.write("Jumlah data x_train:", len(x_train))
    st.write("Jumlah data x_test :", len(x_test))

    st.subheader("⚙️ Pemodelan ARIMAX Otomatis")

    # Grid search parameter
    p_values = range(0, 5)
    d_values = range(1, 2)
    q_values = range(0, 8)

    results = []

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = SARIMAX(y_train, order=(p, d, q), exog=x_train)
                    result = model.fit(disp=False)
                    pvalues = result.pvalues

                    if all(pvalues < 0.05):
                        results.append({
                            'Order (p,d,q)': (p, d, q),
                            'AIC': result.aic,
                            'model_fit': result
                        })
                        st.write(f"Order ({p},{d},{q}) - ✅ SIGNIFICANT")
                    else:
                        st.write(f"Order ({p},{d},{q}) - ❌ NOT SIGNIFICANT")

                except Exception as e:
                    st.warning(f"Error on order ({p},{d},{q}): {e}")
                    continue

    # Evaluasi model terbaik
    if results:
        best_model = min(results, key=lambda x: x['AIC'])
        best_fit = best_model['model_fit']

        # Prediksi
        pred = best_fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=x_test)
        mape = mean_absolute_percentage_error(y_test, pred) * 100

        st.metric("MAPE ARIMAX Terbaik (%)", f"{mape:.2f}")

        # Plot hasil
        st.subheader("📈 Visualisasi Prediksi vs Aktual")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_test.index, y_test.values, label='Aktual', color='black')
        ax.plot(y_test.index, pred, label='Prediksi ARIMAX', linestyle='--', color='blue')
        ax.set_title("Prediksi Harga Cabai")
        ax.set_ylabel("Harga")
        ax.set_xlabel("Tanggal")
        ax.legend()
        st.pyplot(fig)

    else:
        st.error("Tidak ada model ARIMAX yang signifikan ditemukan.")

else:
    st.info("Silakan upload file Excel (.xlsx) yang berisi kolom: 'Tanggal', 'Harga', 'Idul Adha', dan 'Natal'.")
