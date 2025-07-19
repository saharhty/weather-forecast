import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Germany Weather Forecast", layout="wide")

# Set background by weather type
def set_background(weather_type):
    if weather_type == "sunny":
        bg_url = "https://images.unsplash.com/photo-1501973801540-537f08ccae7b"
    elif weather_type == "rainy":
        bg_url = "https://images.unsplash.com/photo-1561484930-998b6a7a2f9b"
    elif weather_type == "cloudy":
        bg_url = "https://images.unsplash.com/photo-1583132331877-caec1e68f7f3"
    else:
        bg_url = ""

    if bg_url:
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url('{bg_url}');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
        """, unsafe_allow_html=True)

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("weather_forecast_model.pkl")
    feature_order = joblib.load("model_features.pkl")
    return model, feature_order

model, feature_order = load_model()

# Load weather data
@st.cache_data
def load_data():
    df = pd.read_csv("weather_data.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    df.dropna(subset=["datetime"], inplace=True)
    df["name"] = df["name"].replace("Colonge", "Cologne")
    return df

df = load_data()

# UI: city and date selector
st.title("🌦️ Weather Forecast Dashboard")
city = st.selectbox("Select City", sorted(df["name"].unique()))

# Generate 7-day forecast
@st.cache_data
def load_forecast():
    last_date = df["datetime"].max()
    city_data = df[df["name"] == city].sort_values("datetime")
    last_row = city_data.iloc[-1]

    forecast_rows = []
    for i in range(1, 8):
        future_date = last_date + timedelta(days=i)
        features = {
            "year": future_date.year,
            "month": future_date.month,
            "day": future_date.day,
            "dayofweek": future_date.weekday(),
            "is_weekend": int(future_date.weekday() >= 5),
            "season": future_date.month % 12 // 3 + 1,
            "prev_temp": last_row["temp"],
            "prev_precip": last_row["precip"]
        }

        for c in ["Berlin", "Cologne", "Frankfurt", "Hamburg", "Munich"]:
            features[f"name_{c}"] = int(city == c)

        X = pd.DataFrame([features])
        X = X.reindex(columns=feature_order, fill_value=0)
        y_pred = model.predict(X)[0]

        forecast_rows.append({
            "date": future_date.strftime("%a %d %b"),
            "temp": round(y_pred[0], 1),
            "humidity": round(y_pred[1], 1),
            "precip": round(y_pred[2], 1),
            "windspeed": round(y_pred[3], 1),
        })

    return pd.DataFrame(forecast_rows)

forecast_df = load_forecast()

# Determine today's weather type for background
weather_type = (
    "sunny" if forecast_df.iloc[0]["temp"] > 22 and forecast_df.iloc[0]["precip"] < 1
    else "rainy" if forecast_df.iloc[0]["precip"] > 2
    else "cloudy"
)
set_background(weather_type)

# Display current day
today = forecast_df.iloc[0]
st.markdown(f"### {city} – {datetime.now().strftime('%A, %d %B')}")
st.metric("🌡️ Temp (°C)", f"{today['temp']:.1f}°")
f"{today['humidity']:.1f}%"  
f"{today['precip']:.1f} mm"  
f"{today['windspeed']:.1f} km/h"

# 7-day forecast strip
st.markdown("---")
st.markdown("### 7-Day Forecast")
cols = st.columns(7)
for i, day in forecast_df.iterrows():
    with cols[i]:
        st.markdown(f"**{day['date']}**")
        st.markdown(f"🌡️ **{day['temp']:.1f}°C**")
        st.markdown(f"💧 {day['humidity']:.1f}%")
        st.markdown(f"☔ {day['precip']:.1f} mm")
        st.markdown(f"💨 {day['windspeed']:.1f} km/h")
