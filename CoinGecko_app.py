import streamlit as st
import pandas as pd
from sklearn.linear_model 
import LinearRegression
from sklearn.model_selection 
import train_test_split
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Prediksi Harga Crypto CoinGecko", layout="wide")
st.title("📈 Prediksi Harga Crypto (Aman untuk Deploy)")

@st.cache_data(ttl=600)
def get_data():
    url = "https://api.coingecko.com/api/v3/klines"

📊 Untuk halaman khusus seperti FAQ atau API, URL-nya bisa menjadi:
- Tanya Jawab: https://www.coingecko.com/id/faq
- API: https://www.coingecko.com/en/api
Kalau kamu sedang membuat presentasi atau dokumentasi dan ingin menulis URL dengan rapi, pastikan:
- Tidak ada spasi
- Gunakan huruf kecil
- Hindari karakter khusus kecuali yang diperbolehkan (seperti /, -, ?, =)
Mau saya bantu membuat daftar URL CoinGecko yang relevan untuk topik tertentu seperti harga koin, grafik, atau API?

    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 500}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            st.error("Gagal mengambil data dari CoinGecko. Status code: " + str(response.status_code))
            return None
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", " "close", "volume",
            "_1", "_2", "_3", "_4", "_5", "_6"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(hours=7)
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df["target"] = df["close"].shift(-1)
        return df.dropna()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        return None

df = get_data()
if df is None or len(df) < 10:
    st.stop()

X = df[["open", "high", "low", "volume"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
akurasi = model.score(X_test, y_test)
st.success(f"✅ Akurasi Model: {akurasi:.2f}")

st.subheader("📊 Data Terbaru")
st.dataframe(df.tail())

st.subheader("🕯️ Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(
    x=df["timestamp"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])
fig.update_layout(xaxis_title="Waktu", yaxis_title="Harga", title="Grafik Candlestick BTC/USDT")
st.plotly_chart(fig, use_container_width=True)

last_data = X.tail(1)
prediksi = model.predict(last_data)
st.success(f"🎯 Prediksi Harga Selanjutnya: ${prediksi[0]:,.2f}")