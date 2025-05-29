# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# App title
st.title("Drug Demand Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    # Column selection
    date_col = st.selectbox("Select Date Column", df.columns)
    value_col = st.selectbox("Select Demand Column", df.columns)

    # Forecasting horizon
    periods = st.slider("Months to Forecast", min_value=1, max_value=24, value=12)

    # Model selection
    model_type = st.selectbox("Select Forecasting Model", ["Prophet", "ARIMA", "LSTM"])

    # Forecast button
    if st.button("Run Forecast"):
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})

        if model_type == "Prophet":
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=periods, freq='M')
            forecast = m.predict(future)
            st.subheader("Forecast Plot")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)

        elif model_type == "ARIMA":
            df_arima = df.set_index('ds')['y'].asfreq('M').fillna(method='ffill')
            model = ARIMA(df_arima, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)
            st.line_chart(pd.concat([df_arima, forecast.rename("Forecast")]))

        elif model_type == "LSTM":
            data = df.set_index("ds").asfreq("M").fillna(method='ffill')
            values = data["y"].values.reshape(-1, 1)

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)

            def create_dataset(series, time_step=5):
                X, Y = [], []
                for i in range(len(series) - time_step - 1):
                    X.append(series[i:(i + time_step), 0])
                    Y.append(series[i + time_step, 0])
                return np.array(X), np.array(Y)

            time_step = 5
            X, y = create_dataset(scaled, time_step)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(X, y, epochs=10, batch_size=1, verbose=0)

            x_input = scaled[-time_step:].reshape(1, time_step, 1)
            predictions = []
            for _ in range(periods):
                next_val = model.predict(x_input)[0][0]
                predictions.append(next_val)
                x_input = np.append(x_input[:, 1:, :], [[[next_val]]], axis=1)

            forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            forecast_dates = pd.date_range(start=df['ds'].iloc[-1], periods=periods + 1, freq='M')[1:]
            forecast_df = pd.DataFrame({'ds': forecast_dates, 'Forecast': forecast.flatten()})

            st.line_chart(pd.concat([df.set_index("ds")["y"], forecast_df.set_index("ds")["Forecast"]], axis=1))

 
