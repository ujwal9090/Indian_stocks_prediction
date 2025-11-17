import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

st.title("📈 Stock Forecast Dashboard using ARIMA")
st.write("Enter a stock name (e.g., RELIANCE.NS, TCS.NS, SBIN.NS) and get predictions.")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="RELIANCE.NS")

start_date = st.sidebar.date_input("Start Date", dt.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date(2025, 11, 15))

forecast_days = st.sidebar.number_input("Forecast Days", min_value=5, max_value=60, value=10)

run_button = st.sidebar.button("Run Forecast")

# -------------------------------
# FUNCTIONS
# -------------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    if result[1] < 0.05:
        return "Stationary (p-value < 0.05)"
    else:
        return "Not Stationary (p-value ≥ 0.05)"

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if run_button:

    st.subheader(f"📌 Downloading Data for {ticker}")

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("❌ Invalid Ticker or No Data Found. Try another ticker.")
    else:
        data = data.reset_index()
        data = data.set_index("Date")

        st.write("### Raw Data")
        st.dataframe(data.head())

        # Use Close Prices
        df = data[["Close"]].copy()
        df["Return"] = df["Close"].pct_change()
        df.dropna(inplace=True)

        st.write("---")
        st.write("### Stationarity Test (ADF Test)")
        stationarity = check_stationarity(df["Close"])
        st.info(stationarity)

        # Differencing if needed
        df["Close_Diff"] = df["Close"].diff()
        df.dropna(inplace=True)

        # Fit ARIMA Model
        st.write("### Training ARIMA Model (5,1,0)...")
        model = ARIMA(df["Close"], order=(5,1,0))
        model_fit = model.fit()

        # Forecasting
        forecast = model_fit.forecast(steps=forecast_days)
        future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq="B")[1:]

        # -------------------------
        # PLOTTING
        # -------------------------
        st.write("### 📉 Actual vs Predicted Prices")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["Close"], label="Actual Price")
        ax.plot(future_dates, forecast, label="Predicted Price", linestyle="dashed")
        ax.set_title(f"{ticker} Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        st.pyplot(fig)

        st.success("Forecast completed!")

