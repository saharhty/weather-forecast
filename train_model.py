import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load data
df = pd.read_csv("weather_data.csv")
df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
df.dropna(subset=["datetime"], inplace=True)
df["name"] = df["name"].replace("Colonge", "Cologne")

# Feature engineering
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["dayofweek"] = df["datetime"].dt.dayofweek
df["is_weekend"] = df["dayofweek"] >= 5
df["season"] = df["month"] % 12 // 3 + 1

# Optional lag features
df = df.sort_values(by=["name", "datetime"])
df["prev_temp"] = df.groupby("name")["temp"].shift(1)
df["prev_precip"] = df.groupby("name")["precip"].shift(1)
df.dropna(inplace=True)

# One-hot encode city
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
city_encoded = encoder.fit_transform(df[["name"]])
city_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(["name"]))

# Combine features
X = pd.concat([
    df[["year", "month", "day", "dayofweek", "is_weekend", "season", "prev_temp", "prev_precip"]].reset_index(drop=True),
    city_df.reset_index(drop=True)
], axis=1)

# Sort columns to ensure consistent order
X = X.sort_index(axis=1)

# Define feature order for future use
expected_features = X.columns.tolist()

# Targets
y = df[["temp", "humidity", "precip", "windspeed"]].reset_index(drop=True)

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42))
model.fit(X_train, y_train)

# Save model and expected features
joblib.dump(model, "weather_forecast_model.pkl")
joblib.dump(expected_features, "model_features.pkl")

print("✅ Improved model trained and saved as weather_forecast_model.pkl")
