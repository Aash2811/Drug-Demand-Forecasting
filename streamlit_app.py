import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet

st.set_page_config(page_title="Prophet Forecasting App", layout="wide")
st.title("📈 Drug Demand Forecasting with Prophet")

# Load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv("drug_demand_forecasting_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# Sidebar UI
st.sidebar.header("🧪 Model Settings")
drug_list = df["drug_name"].unique()
selected_drug = st.sidebar.selectbox("Select Drug", drug_list)
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

# Optional Seasonality
use_weekly = st.sidebar.checkbox("Use Weekly Seasonality", value=True)
use_yearly = st.sidebar.checkbox("Use Yearly Seasonality", value=False)
use_monthly = st.sidebar.checkbox("Add Custom Monthly Seasonality", value=True)

# Regressors
use_temp = st.sidebar.checkbox("Include Temperature", value=True)
use_humidity = st.sidebar.checkbox("Include Humidity", value=True)
use_event = st.sidebar.checkbox("Include Event Flag", value=True)

# Trend flexibility
cp_scale = st.sidebar.slider("Changepoint Prior Scale (Trend Flexibility)", 0.001, 0.5, 0.05)

# Filter data
filtered_df = df[df["drug_name"] == selected_drug].copy().sort_values("date")

# Aggregate data to avoid duplicates
agg_columns = {"units_sold": "sum"}
if use_temp:
    agg_columns["temperature"] = "mean"
if use_humidity:
    agg_columns["humidity"] = "mean"
if use_event:
    agg_columns["event_flag"] = "max"

grouped = filtered_df.groupby("date").agg(agg_columns).reset_index()
grouped = grouped.rename(columns={"date": "ds", "units_sold": "y"})
data = grouped.copy()

# Initialize Prophet
model = Prophet(
    yearly_seasonality=use_yearly,
    weekly_seasonality=use_weekly,
    changepoint_prior_scale=cp_scale
)
if use_monthly:
    model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
if use_temp:
    model.add_regressor("temperature")
if use_humidity:
    model.add_regressor("humidity")
if use_event:
    model.add_regressor("event_flag")

model.fit(data)

# Create future dataframe
future = model.make_future_dataframe(periods=forecast_days)
for var in ["temperature", "humidity", "event_flag"]:
    if var in data.columns:
        future = future.merge(grouped[["ds", var]], on="ds", how="left")
        future[var] = future[var].fillna(method='ffill')

# Predict
forecast = model.predict(future)

# Merge actuals for enhanced plotting
forecast_merged = forecast.set_index("ds").copy()
actuals = data.set_index("ds").copy()
forecast_merged["actual"] = actuals["y"]

# Enhanced Forecast Plot
st.subheader(f"🔮 Forecast for {selected_drug} - Next {forecast_days} Days")

fig, ax = plt.subplots(figsize=(7, 4))

# Actual values (dotted)
forecast_merged["actual"].plot(ax=ax, label="Actual", color="black", linestyle="dotted", marker="o", alpha=0.6)

# Forecasted values
forecast_merged["yhat"].plot(ax=ax, label="Forecast", color="blue")

# Confidence intervals
ax.fill_between(forecast.index, forecast["yhat_lower"], forecast["yhat_upper"], 
                color="blue", alpha=0.2, label="Confidence Interval")

# Forecast start marker
cutoff_date = actuals.index.max()
ax.axvline(cutoff_date, color="gray", linestyle="--", alpha=0.7)
ax.text(cutoff_date, ax.get_ylim()[1]*0.9, 'Forecast Start', rotation=90, color='gray')

# Format
ax.set_title(f"Forecast vs Actual: {selected_drug}", fontsize=14)
ax.set_ylabel("Units Sold")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

st.pyplot(fig)
