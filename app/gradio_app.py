# import os
# import sys
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# import hopsworks
# import joblib
# import requests
# import gradio as gr
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
# from feature_columns import FEATURE_COLUMNS

# # Load environment variables
# load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# # Configuration
# HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
# LATITUDE = float(os.getenv("LATITUDE"))
# LONGITUDE = float(os.getenv("LONGITUDE"))
# FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
# FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION"))
# MODEL_NAMES = {
#     "24h": "aqi_rf_24h",
#     "48h": "aqi_rf_48h",
#     "72h": "aqi_rf_72h"
# }
# MODEL_VERSION = 4
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# # Global variables for caching
# models = {}
# project = None

# def initialize_hopsworks():
#     """Initialize Hopsworks connection and load all models."""
#     global models, project
    
#     try:
#         print("Connecting to Hopsworks...")
#         project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
#         mr = project.get_model_registry()
        
#         for horizon, model_name in MODEL_NAMES.items():
#             print(f"Loading model '{model_name}' (v{MODEL_VERSION})...")
#             aqi_model = mr.get_model(model_name, version=MODEL_VERSION)
#             model_path = aqi_model.download()
#             models[horizon] = joblib.load(os.path.join(model_path, f"{model_name}.pkl"))
        
#         print("Hopsworks initialization successful.")
#         return True
#     except Exception as e:
#         print(f"Error initializing Hopsworks: {e}")
#         return False

# def fetch_current_data():
#     """Fetch current air pollution data from OpenWeatherMap."""
#     try:
#         url = "http://api.openweathermap.org/data/2.5/air_pollution"
#         params = {
#             "lat": LATITUDE,
#             "lon": LONGITUDE,
#             "appid": OPENWEATHER_API_KEY
#         }
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         data = response.json()
#         return data.get("list", [])[0] if data.get("list") else None
#     except Exception as e:
#         print(f"Error fetching current data: {e}")
#         return None

# def prepare_features(raw_data):
#     """Prepare features from raw data for model prediction."""
#     try:
#         # Extract data
#         dt = raw_data.get("dt")
#         components = raw_data.get("components", {})
        
#         # Create datetime
#         dt_obj = datetime.fromtimestamp(dt)
        
#         # Extract features
#         features = {
#             'hour': dt_obj.hour,
#             'day_of_week': dt_obj.weekday(),
#             'month': dt_obj.month,
#             'year': dt_obj.year,
#             'co': float(components.get('co', 0)),
#             'no': float(components.get('no', 0)),
#             'no2': float(components.get('no2', 0)),
#             'o3': float(components.get('o3', 0)),
#             'so2': float(components.get('so2', 0)),
#             'pm2_5': float(components.get('pm2_5', 0)),
#             'pm10': float(components.get('pm10', 0)),
#             'nh3': float(components.get('nh3', 0)),
#             'aqi_lag_1h': np.nan, # Cannot be computed without Feature Store lookup
#             'aqi_change_rate': np.nan, # Cannot be computed without Feature Store lookup
#             'pm2_5_roll_24h': np.nan, # Cannot be computed without Feature Store lookup
#             'pm10_roll_24h': np.nan, # Cannot be computed without Feature Store lookup
#         }
        
#         return features, dt_obj
#     except Exception as e:
#         print(f"Error preparing features: {e}")
#         return None, None

# def get_aqi_level(aqi_value):
#     """Convert AQI value to AQI level description."""
#     if aqi_value <= 1:
#         return "Good"
#     elif aqi_value <= 2:
#         return "Fair"
#     elif aqi_value <= 3:
#         return "Moderate"
#     elif aqi_value <= 4:
#         return "Poor"
#     else:
#         return "Very Poor"

# def get_aqi_color(aqi_value):
#     """Get color for AQI visualization."""
#     if aqi_value <= 1:
#         return "#00E400"  # Green
#     elif aqi_value <= 2:
#         return "#FFFF00"  # Yellow
#     elif aqi_value <= 3:
#         return "#FF7E00"  # Orange
#     elif aqi_value <= 4:
#         return "#FF0000"  # Red
#     else:
#         return "#8F3F97"  # Purple

# def predict_aqi():
#     """Predict AQI for the next 24, 48, and 72 hours."""
#     try:
#         if not models:
#             return "Error: Models not loaded", None, None
        
#         # Fetch current data
#         raw_data = fetch_current_data()
#         if not raw_data:
#             return "Error: Failed to fetch current data", None, None
        
#         # Prepare features
#         features, dt_obj = prepare_features(raw_data)
#         # Create DataFrame for prediction
#         X = pd.DataFrame([features])
        
#         # Fill NaN values with 0 for prediction (as the model was trained without lookback features)
#         X = X.fillna(0)
        
#         # Ensure correct data types and order
#         X = X[FEATURE_COLUMNS]
        
#         # Make predictions for all horizons
#         forecast_data = []
#         current_aqi = raw_data.get("main", {}).get("aqi", 0)
        
#         for horizon, model in models.items():
#             hours = int(horizon.replace('h', ''))
#             prediction = model.predict(X)[0]
#             forecast_time = dt_obj + timedelta(hours=hours)
            
#             forecast_data.append([
#                 f"{hours} Hours",
#                 forecast_time.strftime("%Y-%m-%d %H:%M"),
#                 f"{prediction:.2f}",
#                 get_aqi_level(prediction)
#             ])
        
#         # Create summary (using 72h prediction for the main summary)
#         prediction_72h = models['72h'].predict(X)[0]
#         summary = f"""
#         **Current AQI:** {current_aqi}
#         **Predicted AQI (72-hour):** {prediction_72h:.2f}
#         **AQI Level (72-hour):** {get_aqi_level(prediction_72h)}
#         **Timestamp:** {dt_obj.strftime("%Y-%m-%d %H:%M:%S")}
#         **Location:** Karachi, Pakistan ({LATITUDE}, {LONGITUDE})
#         """
        
#         return summary, forecast_data, None
#     except Exception as e:
#         return f"Error: {str(e)}", None, None

# # Initialize model on startup
# initialize_hopsworks()

# # Create Gradio interface
# with gr.Blocks(title="AQI Predictor - Karachi") as demo:
#     gr.Markdown("# Air Quality Index (AQI) Predictor for Karachi")
#     gr.Markdown("Predict the air quality for the next **24, 48, and 72 Hours** using machine learning.")
    
#     with gr.Row():
#         with gr.Column():
#             gr.Markdown("### Prediction")
#             predict_btn = gr.Button("Get Multi-Horizon Forecast (24h, 48h, 72h)", variant="primary")
        
#     with gr.Row():
#         with gr.Column():
#             summary_output = gr.Markdown("Click the button to get predictions...")
        
#     with gr.Row():
#         with gr.Column():
#             forecast_table = gr.Dataframe(
#                 headers=["Horizon", "Timestamp", "Predicted AQI", "AQI Level"],
#                 label="Multi-Horizon Forecast (24h, 48h, 72h)"
#             )
    
#     with gr.Row():
#         with gr.Column():
#             gr.Markdown("### About")
#             gr.Markdown("""
#             This application uses a Random Forest model trained on historical air pollution data
#             from OpenWeatherMap to predict the AQI for Karachi over the next 3 days.
            
#             **AQI Levels:**
#             - **Good (0-1):** Air quality is satisfactory
#             - **Fair (1-2):** Air quality is acceptable
#             - **Moderate (2-3):** Members of sensitive groups may experience health effects
#             - **Poor (3-4):** Some members of the general public may begin to experience health effects
#             - **Very Poor (4+):** Health alert; the entire population is more likely to be affected
#             """)
    
#     predict_btn.click(
#         fn=predict_aqi,
#         outputs=[summary_output, forecast_table, gr.State()]
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7863, share=False)


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
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "4"))
MODEL_NAMES = {
    "24h": "aqi_predictor_24h",
    "48h": "aqi_predictor_48h",
    "72h": "aqi_predictor_72h"
}
MODEL_VERSION = 1
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Global variables for caching
models = {}
model_types = {}
project = None

def initialize_hopsworks():
    """Initialize Hopsworks connection and load all models."""
    global models, model_types, project
    
    try:
        print("Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        for horizon, model_name in MODEL_NAMES.items():
            print(f"\nLoading model '{model_name}' (v{MODEL_VERSION})...")
            
            # Retry logic for model download
            max_retries = 3
            retry_count = 0
            model_loaded = False
            
            while retry_count < max_retries and not model_loaded:
                try:
                    if retry_count > 0:
                        print(f"  Retry attempt {retry_count}/{max_retries-1}...")
                    
                    aqi_model = mr.get_model(model_name, version=MODEL_VERSION)
                    
                    # Download with timeout handling
                    print(f"  Downloading model (this may take a minute)...")
                    model_dir = aqi_model.download()
                    
                    # Try to load the model file
                    model_file = os.path.join(model_dir, f"{model_name}_v{MODEL_VERSION}.pkl")
                    if not os.path.exists(model_file):
                        # Try alternate naming pattern
                        model_file = os.path.join(model_dir, f"{model_name}.pkl")
                    
                    if not os.path.exists(model_file):
                        raise FileNotFoundError(f"Model file not found in {model_dir}")
                    
                    print(f"  Loading model from disk...")
                    models[horizon] = joblib.load(model_file)
                    
                    # Extract model type from description if available
                    description = getattr(aqi_model, 'description', '')
                    if 'Model Type:' in description:
                        model_type = description.split('Model Type:')[1].split('.')[0].strip()
                        model_types[horizon] = model_type
                    else:
                        model_types[horizon] = "Unknown"
                    
                    print(f"  ✓ Successfully loaded {model_types[horizon]} model for {horizon} horizon")
                    model_loaded = True
                    
                except Exception as e:
                    retry_count += 1
                    print(f"  ✗ Error loading model (attempt {retry_count}/{max_retries}): {e}")
                    
                    if retry_count < max_retries:
                        import time
                        wait_time = 5 * retry_count  # Exponential backoff
                        print(f"  Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ✗ Failed to load model for {horizon} after {max_retries} attempts")
                        print(f"  You can continue with partial functionality (models loaded: {list(models.keys())})")
        
        if not models:
            print("\n✗ No models were loaded successfully.")
            return False
        
        print(f"\n✓ Hopsworks initialization complete. Loaded {len(models)}/{len(MODEL_NAMES)} models.")
        if len(models) < len(MODEL_NAMES):
            print(f"  Warning: Some models failed to load. Available horizons: {list(models.keys())}")
        return True
        
    except Exception as e:
        print(f"\n✗ Critical error initializing Hopsworks: {e}")
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
        
        # Extract features in the correct order as per FEATURE_COLUMNS
        features = {
            'hour': dt_obj.hour,
            'day_of_week': dt_obj.weekday(),
            'month': dt_obj.month,
            'year': dt_obj.year,
            'aqi_lag_1h': 0.0,  # Set to 0 for inference (trained with these as NaN/0)
            'aqi_change_rate': 0.0,  # Set to 0 for inference
            'pm2_5_roll_24h': float(components.get('pm2_5', 0)),  # Use current as approximation
            'pm10_roll_24h': float(components.get('pm10', 0)),  # Use current as approximation
            'co': float(components.get('co', 0)),
            'no': float(components.get('no', 0)),
            'no2': float(components.get('no2', 0)),
            'o3': float(components.get('o3', 0)),
            'so2': float(components.get('so2', 0)),
            'pm2_5': float(components.get('pm2_5', 0)),
            'pm10': float(components.get('pm10', 0)),
            'nh3': float(components.get('nh3', 0)),
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

def get_health_recommendation(aqi_level):
    """Get health recommendations based on AQI level."""
    recommendations = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities!",
        "Fair": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "Moderate": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        "Poor": "Some members of the general public may experience health effects. Sensitive groups should limit outdoor exertion.",
        "Very Poor": "Health alert: Everyone may experience more serious health effects. Avoid outdoor activities."
    }
    return recommendations.get(aqi_level, "No recommendation available.")

def predict_aqi():
    """Predict AQI for the next 24, 48, and 72 hours."""
    try:
        if not models:
            return "❌ **Error:** No models loaded. Please check the logs and restart the app.", None
        
        # Fetch current data
        raw_data = fetch_current_data()
        if not raw_data:
            return "❌ **Error:** Failed to fetch current data from OpenWeatherMap.", None
        
        # Prepare features
        features, dt_obj = prepare_features(raw_data)
        if features is None:
            return "❌ **Error:** Failed to prepare features from raw data.", None
        
        # Create DataFrame for prediction
        X = pd.DataFrame([features])
        
        # Ensure correct column order
        X = X[FEATURE_COLUMNS]
        
        # Make predictions for all available horizons
        forecast_data = []
        current_aqi = raw_data.get("main", {}).get("aqi", 0)
        current_level = get_aqi_level(current_aqi)
        
        predictions = {}
        available_horizons = sorted(models.keys(), key=lambda x: int(x.replace('h', '')))
        
        for horizon in available_horizons:
            model = models[horizon]
            hours = int(horizon.replace('h', ''))
            prediction = model.predict(X)[0]
            predictions[horizon] = prediction
            forecast_time = dt_obj + timedelta(hours=hours)
            aqi_level = get_aqi_level(prediction)
            
            forecast_data.append([
                f"{hours} Hours",
                forecast_time.strftime("%Y-%m-%d %H:%M"),
                f"{prediction:.2f}",
                aqi_level,
                model_types.get(horizon, "Unknown")
            ])
        
        # Create detailed summary with available predictions
        summary_parts = [
            "## 🌍 Current Air Quality",
            f"- **Location:** Karachi, Pakistan ({LATITUDE}°N, {LONGITUDE}°E)",
            f"- **Timestamp:** {dt_obj.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Current AQI:** {current_aqi} ({current_level})",
            f"- **Current Health Impact:** {get_health_recommendation(current_level)}",
            "",
            "---",
            "",
            "## 🔮 Forecast Summary"
        ]
        
        # Add predictions for available horizons
        if '24h' in predictions:
            level_24h = get_aqi_level(predictions['24h'])
            summary_parts.append(f"- **24-Hour Prediction:** {predictions['24h']:.2f} ({level_24h})")
        
        if '48h' in predictions:
            level_48h = get_aqi_level(predictions['48h'])
            summary_parts.append(f"- **48-Hour Prediction:** {predictions['48h']:.2f} ({level_48h})")
        
        if '72h' in predictions:
            level_72h = get_aqi_level(predictions['72h'])
            summary_parts.append(f"- **72-Hour Prediction:** {predictions['72h']:.2f} ({level_72h})")
        
        # Add warning if not all models loaded
        if len(predictions) < 3:
            missing = set(['24h', '48h', '72h']) - set(predictions.keys())
            summary_parts.append("")
            summary_parts.append(f"⚠️ *Note: Models for {', '.join(missing)} are not available*")
        
        summary_parts.extend([
            "",
            "---",
            "",
            "## 💡 Health Recommendation"
        ])
        
        # Use the longest horizon available for recommendations
        longest_horizon = available_horizons[-1] if available_horizons else None
        if longest_horizon:
            longest_level = get_aqi_level(predictions[longest_horizon])
            summary_parts.append(f"**{longest_horizon} Forecast:** {get_health_recommendation(longest_level)}")
        
        summary_parts.extend([
            "",
            "---",
            "",
            "## 🤖 Model Information"
        ])
        
        for horizon in available_horizons:
            summary_parts.append(f"- **{horizon} Model:** {model_types.get(horizon, 'Unknown')}")
        
        summary = "\n".join(summary_parts)
        
        return summary, forecast_data
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict_aqi: {error_details}")
        return f"❌ **Error:** {str(e)}\n\nPlease check the console for details.", None

# Initialize models on startup
print("\n" + "="*60)
print("INITIALIZING AQI PREDICTOR APP")
print("="*60)
if initialize_hopsworks():
    print("✓ App ready!")
else:
    print("✗ Failed to initialize. Please check your configuration.")
print("="*60 + "\n")

# Create Gradio interface
with gr.Blocks(title="AQI Predictor - Karachi", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌫️ Air Quality Index (AQI) Predictor for Karachi
    
    Predict air quality for the next **24, 48, and 72 hours** using state-of-the-art machine learning models.
    """)
    
    with gr.Row():
        with gr.Column():
            predict_btn = gr.Button(
                "🔮 Get Multi-Horizon Forecast",
                variant="primary",
                size="lg"
            )
    
    with gr.Row():
        with gr.Column():
            summary_output = gr.Markdown("Click the button above to get predictions...")
    
    with gr.Row():
        with gr.Column():
            forecast_table = gr.Dataframe(
                headers=["Horizon", "Forecast Time", "Predicted AQI", "AQI Level", "Model Type"],
                label="📊 Multi-Horizon Forecast Details",
                wrap=True
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ---
            
            ### 📖 About This App
            
            This application uses machine learning models trained on historical air pollution data
            from OpenWeatherMap to predict the AQI for Karachi, Pakistan. The system automatically
            selects the best performing model (XGBoost, Random Forest, Gradient Boosting, or Linear Regression)
            for each prediction horizon.
            
            #### AQI Scale Reference:
            
            | AQI Value | Level | Health Impact | Color |
            |-----------|-------|---------------|-------|
            | 1 | Good | Air quality is satisfactory | 🟢 Green |
            | 2 | Fair | Air quality is acceptable | 🟡 Yellow |
            | 3 | Moderate | Sensitive groups may be affected | 🟠 Orange |
            | 4 | Poor | General public may be affected | 🔴 Red |
            | 5 | Very Poor | Health alert for everyone | 🟣 Purple |
            
            #### Features Used:
            - Time-based features (hour, day of week, month, year)
            - Current pollutant levels (PM2.5, PM10, CO, NO, NO2, O3, SO2, NH3)
            - Historical patterns (lag features and rolling averages)
            
            #### Data Source:
            Real-time air quality data from [OpenWeatherMap API](https://openweathermap.org/)
            
            ---
            
            *Predictions are for informational purposes only. For health-related decisions, 
            please consult official air quality monitoring agencies.*
            """)
    
    predict_btn.click(
        fn=predict_aqi,
        outputs=[summary_output, forecast_table]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)