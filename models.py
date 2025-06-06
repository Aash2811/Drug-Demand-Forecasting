# models.py

import pandas as pd


def forecast_with_prophet(df, forecast_horizon):
    from prophet import Prophet
    df = df.rename(columns={"ds": "ds", "y": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_horizon)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(forecast_horizon)

def forecast_with_arima(df, forecast_horizon):
    from pmdarima import auto_arima
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds")
    model = auto_arima(df["y"], seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=forecast_horizon)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    return pd.DataFrame({"ds": future_dates, "yhat": forecast})

def forecast_with_lstm(df, forecast_horizon):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds")[["y"]]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    def create_sequences(data, seq_length=30):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    sequence_length = 30
    X, y = create_sequences(data_scaled, sequence_length)
    X = X.reshape((X.shape[0], sequence_length, 1))

    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    last_seq = data_scaled[-sequence_length:]
    predictions = []

    for _ in range(forecast_horizon):
        pred = model.predict(last_seq.reshape(1, sequence_length, 1), verbose=0)[0][0]
        predictions.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    return pd.DataFrame({"ds": future_dates, "yhat": forecast.flatten()})
 
