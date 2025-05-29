# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
from models import forecast_with_prophet, forecast_with_arima, forecast_with_lstm

st.set_page_config(page_title="Drug Forecasting", layout="centered")

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type")
        return None

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecast CSV</a>'

def main():
    st.title("💊 Drug Demand Forecasting")

    uploaded_file = st.file_uploader("📤 Upload Your Data File (CSV/Excel)", type=["csv", "xls", "xlsx"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("📈 Preview of Uploaded Data")
        st.dataframe(df.head())

        forecast_days = st.number_input("⏳ Number of Days to Forecast", min_value=1, max_value=365, value=30)
        model_type = st.selectbox("🧠 Choose Forecasting Model", ["Prophet", "ARIMA", "LSTM"])

        if st.button("🚀 Run Forecast"):
            try:
                with st.spinner("Running model..."):
                    if model_type == "Prophet":
                        forecast = forecast_with_prophet(df, forecast_days)
                    elif model_type == "ARIMA":
                        forecast = forecast_with_arima(df, forecast_days)
                    else:
                        forecast = forecast_with_lstm(df, forecast_days)

                st.success("✅ Forecast Complete!")
                st.subheader("🔍 Forecast Results")
                st.dataframe(forecast)

                st.subheader("📊 Forecast Plot")
                plt.figure(figsize=(10, 5))
                plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
                plt.xlabel("Date")
                plt.ylabel("Demand")
                plt.legend()
                st.pyplot(plt)

                st.markdown(get_download_link(forecast), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
