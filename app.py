import streamlit as st
import pandas as pd
import pickle

st.title("🚧 Pothole Risk Predictor")
st.write("Adjust key conditions below to predict pothole risk:")

# -----------------------------
# Load Model and Features
# -----------------------------
with open("C:\\Users\\sumit\\pothole_project\\pothole_rf_model2.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:\\Users\\sumit\\pothole_project\\features.pkl", "rb") as f:
    features = pickle.load(f)

# -----------------------------
# Default values for all features (except target)
# -----------------------------
default_values = {
    'temperature_2m': 30.0,
    'relative_humidity': 50.0,
    'dew_point': 20.0,
    'wind_speed': 5.0,
    'max_temp': 35.0,
    'min_temp': 25.0,
    'avg_temp': 30.0,
    'precipitation': 0.0,
    'pressure': 1013.0,
    'PCU_City_Wide_Avg_Volume': 90000.0,
    'Traffic_Data_Source': 0,  # numeric encoding
    'Is_Peak_Traffic_Hour': 0,
    'Year': 2023,
    'Month': 1,
    'Day': 1,
    'Hour': 12,
    'DayOfWeek': 1,
    'Is_Weekend': 0,
    'Temp_Range': 10.0,
    'Humidex': 25.0,
    'Precipitation_Log': 0.0
}

# -----------------------------
# Collect user input (only most important)
# -----------------------------
user_input = {}
user_input['relative_humidity'] = st.slider('Relative Humidity (%)', 0, 100, 50)
user_input['temperature_2m'] = st.slider('Temperature (°C)', -10, 50, 30)
user_input['precipitation'] = st.slider('Precipitation (mm)', 0.0, 50.0, 0.0)
user_input['Precipitation_Log'] = st.slider('Precipitation Log', 0.0, 5.0, 0.0)

# -----------------------------
# Fill remaining features with defaults
# -----------------------------
for col in features:
    if col not in user_input and col != 'Pothole_Risk':
        user_input[col] = default_values[col]

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Risk"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[features]  # reorder exactly as training

    pred = model.predict(input_df)[0]

    # Categorize risk
    def risk_category(value):
        if value < 0.33:
            return "Low"
        elif value < 0.66:
            return "Medium"
        else:
            return "High"

    st.success(f"Predicted Pothole Probability: {pred:.2f} ({risk_category(pred)})")
