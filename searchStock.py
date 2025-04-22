import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
url = 'https://raw.githubusercontent.com/blueboxHuang/Screener/main/kode_saham.csv'
df = pd.read_csv(url)
st.set_page_config(page_title="Screener Saham Indonesia", layout="wide")
st.title("📊 Screener Saham Ala Ala BlueBox")

# === Fungsi ambil data ===
@st.cache_data(show_spinner=False)
def get_fundamental_data(ticker_list):
    data_list = []
    progress = st.progress(0, text="📡 Mengambil data saham...")
    for i, ticker in enumerate(ticker_list):
        try:
            saham = yf.Ticker(ticker + ".JK")
            info = saham.info
            hist = saham.history(period="1mo")
            avg_volume = hist["Volume"].mean() if not hist.empty else 0

            data_list.append({
                "Kode": ticker,
                "Nama": info.get("longName", ""),
                "PER": float(info.get("trailingPE")) if info.get("trailingPE") else None,
                "PBV": float(info.get("priceToBook")) if info.get("priceToBook") else None,
                "ROE": float(info.get("returnOnEquity")) if info.get("returnOnEquity") else None,
                "DER": float(info.get("debtToEquity")) if info.get("debtToEquity") else None,
                "Avg Volume": avg_volume
            })
        except Exception as e:
            data_list.append({
                "Kode": ticker,
                "Nama": f"Error: {e}",
                "PER": None,
                "PBV": None,
                "ROE": None,
                "DER": None,
                "Avg Volume": 0
            })
        progress.progress((i + 1) / len(ticker_list), text=f"📡 Mengambil data {i+1}/{len(ticker_list)} saham...")
    progress.empty()
    return pd.DataFrame(data_list)

# === Fungsi prediksi sederhana ===
def predict_price(ticker):
    try:
        df = yf.download(ticker + ".JK", period="30d", interval="1d", progress=False)
        if df.empty:
            st.warning(f"⚠️ Data kosong untuk {ticker}")
            return None, None, "❌ Data kosong"

        df = df.reset_index()
        df["Day"] = np.arange(len(df))
        X = df[["Day"]]
        y = df["Close"]

        model = LinearRegression().fit(X, y)
        next_day = [[len(df) + 7]]
        predicted_price = model.predict(next_day)[0]
        last_price = df["Close"].iloc[-1]

        if float(predicted_price) > float(last_price):
            rekom = "⬆️ Naik"
        else:
            rekom = "⬇️ Turun"

        return round(float(last_price), 2), round(float(predicted_price), 2), rekom
    except Exception as e:
        st.error(f"❌ Gagal prediksi {ticker}: {e}")
        return None, None, f"❌ Error: {e}"



# === Upload CSV ===
uploaded_file = st.file_uploader("📂 Upload file CSV berisi kode saham (kolom: 'Kode')", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    if "Kode" not in df_input.columns:
        st.error("⚠️ Kolom 'Kode' tidak ditemukan.")
    else:
        ticker_list = df_input["Kode"].dropna().astype(str).str.upper().tolist()
        st.info(f"Jumlah kode saham ditemukan: {len(ticker_list)}")

        if st.button("🔍 Ambil Data"):
            df_result = get_fundamental_data(ticker_list)
            st.session_state["df_result"] = df_result
            st.success("✅ Data berhasil diambil!")

# === Tampilkan filter & prediksi ===
if "df_result" in st.session_state:
    df_result = st.session_state["df_result"]

    opsi_volume = st.checkbox("📈 Hanya saham dengan volume harian > 200.000")
    opsi_fundamental = st.checkbox("✅ Filter berdasarkan rasio fundamental (PER, PBV, ROE, DER)")

    df_filtered = df_result.copy()
    if opsi_volume:
        df_filtered = df_filtered[df_filtered["Avg Volume"] > 2_000_000]

    if opsi_fundamental:
        df_filtered = df_filtered[
            (df_filtered["PER"].fillna(9999) <= 15) &
            (df_filtered["PBV"].fillna(9999) <= 2) &
            (df_filtered["ROE"].fillna(0) * 100 >= 10) &
            (df_filtered["DER"].fillna(9999) <= 2)
        ]

    st.success(f"✅ {len(df_filtered)} saham terfilter ditemukan.")
    st.dataframe(df_filtered)

    if not df_filtered.empty:
        if st.button("🤖 Jalankan Prediksi AI (Harga 7 Hari ke Depan)"):
            hasil_prediksi = []
            for kode in df_filtered["Kode"]:
                st.write(f"⏳ Memproses: {kode}")
                harga_akhir, prediksi, rekom = predict_price(kode)
                if harga_akhir is not None:
                    hasil_prediksi.append({
                        "Kode": kode,
                        "Harga Terakhir": harga_akhir,
                        "Prediksi 7 Hari": prediksi,
                        "Rekomendasi": rekom
                    })

            if hasil_prediksi:
                df_prediksi = pd.DataFrame(hasil_prediksi)
                st.subheader("📈 Hasil Prediksi AI")
                st.dataframe(df_prediksi)
                st.download_button("⬇️ Download hasil prediksi", data=df_prediksi.to_csv(index=False).encode("utf-8"), file_name="prediksi_ai.csv", mime="text/csv")
            else:
                st.warning("⚠️ Tidak ada saham yang berhasil diprediksi.")
