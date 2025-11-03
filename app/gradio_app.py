import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import joblib
import requests
import gradio as gr
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from feature_columns import FEATURE_COLUMNS

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
LATITUDE = float(os.getenv("LATITUDE"))
LONGITUDE = float(os.getenv("LONGITUDE"))
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION"))
MODEL_NAMES = {
    "24h": "aqi_rf_24h",
    "48h": "aqi_rf_48h",
    "72h": "aqi_rf_72h"
}
MODEL_VERSION = 4
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Global variables for caching
models = {}
project = None

def initialize_hopsworks():
    """Initialize Hopsworks connection and load all models."""
    global models, project
    
    try:
        print("Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        for horizon, model_name in MODEL_NAMES.items():
            print(f"Loading model '{model_name}' (v{MODEL_VERSION})...")
            aqi_model = mr.get_model(model_name, version=MODEL_VERSION)
            model_path = aqi_model.download()
            models[horizon] = joblib.load(os.path.join(model_path, f"{model_name}.pkl"))
        
        print("Hopsworks initialization successful.")
        return True
    except Exception as e:
        print(f"Error initializing Hopsworks: {e}")
        return False

def fetch_current_data():
    """Fetch current air pollution data from OpenWeatherMap."""
    try:
        url = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            "lat": LATITUDE,
            "lon": LONGITUDE,
            "appid": OPENWEATHER_API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("list", [])[0] if data.get("list") else None
    except Exception as e:
        print(f"Error fetching current data: {e}")
        return None

def prepare_features(raw_data):
    """Prepare features from raw data for model prediction."""
    try:
        # Extract data
        dt = raw_data.get("dt")
        components = raw_data.get("components", {})
        
        # Create datetime
        dt_obj = datetime.fromtimestamp(dt)
        
        # Extract features
        features = {
            'hour': dt_obj.hour,
            'day_of_week': dt_obj.weekday(),
            'month': dt_obj.month,
            'year': dt_obj.year,
            'co': float(components.get('co', 0)),
            'no': float(components.get('no', 0)),
            'no2': float(components.get('no2', 0)),
            'o3': float(components.get('o3', 0)),
            'so2': float(components.get('so2', 0)),
            'pm2_5': float(components.get('pm2_5', 0)),
            'pm10': float(components.get('pm10', 0)),
            'nh3': float(components.get('nh3', 0)),
            'aqi_lag_1h': np.nan, # Cannot be computed without Feature Store lookup
            'aqi_change_rate': np.nan, # Cannot be computed without Feature Store lookup
            'pm2_5_roll_24h': np.nan, # Cannot be computed without Feature Store lookup
            'pm10_roll_24h': np.nan, # Cannot be computed without Feature Store lookup
        }
        
        return features, dt_obj
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None, None

def get_aqi_level(aqi_value):
    """Convert AQI value to AQI level description."""
    if aqi_value <= 1:
        return "Good"
    elif aqi_value <= 2:
        return "Fair"
    elif aqi_value <= 3:
        return "Moderate"
    elif aqi_value <= 4:
        return "Poor"
    else:
        return "Very Poor"

def get_aqi_color(aqi_value):
    """Get color for AQI visualization."""
    if aqi_value <= 1:
        return "#00E400"  # Green
    elif aqi_value <= 2:
        return "#FFFF00"  # Yellow
    elif aqi_value <= 3:
        return "#FF7E00"  # Orange
    elif aqi_value <= 4:
        return "#FF0000"  # Red
    else:
        return "#8F3F97"  # Purple

def predict_aqi():
    """Predict AQI for the next 24, 48, and 72 hours."""
    try:
        if not models:
            return "Error: Models not loaded", None, None
        
        # Fetch current data
        raw_data = fetch_current_data()
        if not raw_data:
            return "Error: Failed to fetch current data", None, None
        
        # Prepare features
        features, dt_obj = prepare_features(raw_data)
        # Create DataFrame for prediction
        X = pd.DataFrame([features])
        
        # Fill NaN values with 0 for prediction (as the model was trained without lookback features)
        X = X.fillna(0)
        
        # Ensure correct data types and order
        X = X[FEATURE_COLUMNS]
        
        # Make predictions for all horizons
        forecast_data = []
        current_aqi = raw_data.get("main", {}).get("aqi", 0)
        
        for horizon, model in models.items():
            hours = int(horizon.replace('h', ''))
            prediction = model.predict(X)[0]
            forecast_time = dt_obj + timedelta(hours=hours)
            
            forecast_data.append([
                f"{hours} Hours",
                forecast_time.strftime("%Y-%m-%d %H:%M"),
                f"{prediction:.2f}",
                get_aqi_level(prediction)
            ])
        
        # Create summary (using 72h prediction for the main summary)
        prediction_72h = models['72h'].predict(X)[0]
        summary = f"""
        **Current AQI:** {current_aqi}
        **Predicted AQI (72-hour):** {prediction_72h:.2f}
        **AQI Level (72-hour):** {get_aqi_level(prediction_72h)}
        **Timestamp:** {dt_obj.strftime("%Y-%m-%d %H:%M:%S")}
        **Location:** Karachi, Pakistan ({LATITUDE}, {LONGITUDE})
        """
        
        return summary, forecast_data, None
    except Exception as e:
        return f"Error: {str(e)}", None, None

# Initialize model on startup
initialize_hopsworks()

# Create Gradio interface
with gr.Blocks(title="AQI Predictor - Karachi") as demo:
    gr.Markdown("# Air Quality Index (AQI) Predictor for Karachi")
    gr.Markdown("Predict the air quality for the next **24, 48, and 72 Hours** using machine learning.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Prediction")
            predict_btn = gr.Button("Get Multi-Horizon Forecast (24h, 48h, 72h)", variant="primary")
        
    with gr.Row():
        with gr.Column():
            summary_output = gr.Markdown("Click the button to get predictions...")
        
    with gr.Row():
        with gr.Column():
            forecast_table = gr.Dataframe(
                headers=["Horizon", "Timestamp", "Predicted AQI", "AQI Level"],
                label="Multi-Horizon Forecast (24h, 48h, 72h)"
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### About")
            gr.Markdown("""
            This application uses a Random Forest model trained on historical air pollution data
            from OpenWeatherMap to predict the AQI for Karachi over the next 3 hours.
            
            **AQI Levels:**
            - **Good (0-1):** Air quality is satisfactory
            - **Fair (1-2):** Air quality is acceptable
            - **Moderate (2-3):** Members of sensitive groups may experience health effects
            - **Poor (3-4):** Some members of the general public may begin to experience health effects
            - **Very Poor (4+):** Health alert; the entire population is more likely to be affected
            """)
    
    predict_btn.click(
        fn=predict_aqi,
        outputs=[summary_output, forecast_table, gr.State()]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)
