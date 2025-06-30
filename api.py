import streamlit as st
import pickle
import numpy as np

# ---------- Page Config ----------
st.set_page_config(
    page_title="Solar Power Predictor Pro",
    page_icon="🌞",
    layout="wide"
)

# ---------- Constants ----------
FEATURES = [
    'distance_to_solar_noon', 'temperature', 'wind_direction',
    'wind_speed', 'sky_cover', 'humidity', 'average_pressure',
    'visibility'
]

VALID_RANGES = {
    'distance_to_solar_noon': (0.0, 1.5),
    'temperature': (0.0, 80.0),
    'wind_direction': (0, 360),
    'wind_speed': (0.0, 20.0),
    'sky_cover': (0, 4),
    'humidity': (0, 100),
    'average_pressure': (29.0, 30.5),
    'visibility':(0,2)
}

# ---------- Model Loading ----------
@st.cache_resource
def load_model():
    try:
        with open("LGBM.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'LGBM.pkl' not found!")
        st.stop()

model = load_model()

# ---------- Interface ----------
st.title("🌞 Solar Power Prediction System")
st.markdown("Predict 3-hour energy production (Joules) using environmental metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Solar Position")
    distance_to_solar_noon = st.slider(
        "🌞 Solar Noon Distance (radians)", 
        *VALID_RANGES['distance_to_solar_noon'], 0.5
    )
    sky_cover = st.select_slider(
        "☁️ Sky Cover", 
        options=range(5),
        help="0 = Clear, 4 = Overcast"
    )

with col2:
    st.header("Weather Conditions")
    temperature = st.number_input(
        "🌡️ Temperature (°C)", 
        *VALID_RANGES['temperature'], 25.0
    )
    wind_speed = st.number_input(
        "💨 Wind Speed (m/s)", 
        *VALID_RANGES['wind_speed'], 5.0
    )

with col3:
    st.header("Atmospheric Metrics")
    humidity = st.number_input(
        "💧 Humidity (%)", 
        *VALID_RANGES['humidity'], 50
    )
    average_pressure = st.number_input(
        "📉 Pressure (inHg)", 
        *VALID_RANGES['average_pressure'], 29.92, 
        format="%.2f"
    )
    wind_direction = st.number_input(
        "🧭 Wind Direction (°)", 
        *VALID_RANGES['wind_direction'], 180
    )
      # Add visibility input
    visibility = st.select_slider(
        "👁️ Visibility",
        options=[(0, "High"), (1, "Medium"), (2, "Low")],
        format_func=lambda x: x[1],
        help="Atmospheric visibility level"
    )[0]

# ---------- Prediction ----------
# Prediction
if st.button("🔍 Predict Energy Output", type="primary"):
    # Explicitly map inputs (safer than eval)
    inputs = {
        'distance_to_solar_noon': distance_to_solar_noon,
        'temperature': temperature,
        'wind_direction': wind_direction,
        'wind_speed': wind_speed,
        'sky_cover': sky_cover,
        'humidity': humidity,
        'average_pressure': average_pressure,
        'visibility':visibility
    }
    
    # Validate input
    for feature, value in inputs.items():
        if not (VALID_RANGES[feature][0] <= value <= VALID_RANGES[feature][1]):
            st.error(f"Invalid {feature.replace('_', ' ')}: Must be between {VALID_RANGES[feature][0]} and {VALID_RANGES[feature][1]}")
            st.stop()

    # Prepare input for model
    input_array = np.array([[inputs[feat] for feat in FEATURES]], dtype=np.float32)
    
    try:
        with st.spinner("Analyzing environmental factors..."):
            joules = model.predict(input_array)[0]
            kwh = joules / 3_600_000

        st.success(f"## ⚡ Estimated Energy: {joules:,.0f} J ({kwh:.2f} kWh)")

    except Exception as e:
        st.error(f"Prediction error: {e}")