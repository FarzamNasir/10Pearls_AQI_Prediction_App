import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
LATITUDE = float(os.getenv("LATITUDE"))
LONGITUDE = float(os.getenv("LONGITUDE"))
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION"))
MODEL_NAME = "aqi_random_forest_predictor"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="AQI Predictor API",
    description="Air Quality Index Prediction API for Karachi",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
model = None
feature_view = None
project = None

def initialize_hopsworks():
    """Initialize Hopsworks connection and load model and features."""
    global model, feature_view, project
    
    try:
        print("Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        # Load the model from Model Registry
        print(f"Loading model '{MODEL_NAME}'...")
        aqi_model = mr.get_model(MODEL_NAME, version=1)
        model_path = aqi_model.download()
        model = joblib.load(os.path.join(model_path, "aqi_rf_model.pkl"))
        
        # Get the feature view
        try:
            feature_view = fs.get_feature_view(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        except:
            print(f"Feature view '{FEATURE_GROUP_NAME}' not found.")
            feature_view = None
        
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
        main = raw_data.get("main", {})
        
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
            'aqi_lag_1h': np.nan,  # Will be fetched from Feature Store if available
            'aqi_change_rate': np.nan,  # Will be computed if aqi_lag_1h is available
        }
        
        return features, dt_obj
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None, None

@app.on_event("startup")
async def startup_event():
    """Initialize Hopsworks connection on app startup."""
    initialize_hopsworks()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AQI Predictor API",
        "endpoints": {
            "predict": "/predict",
            "current": "/current",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/current")
async def get_current_aqi():
    """Get current AQI data."""
    try:
        raw_data = fetch_current_data()
        if not raw_data:
            raise HTTPException(status_code=500, detail="Failed to fetch current data")
        
        main = raw_data.get("main", {})
        components = raw_data.get("components", {})
        
        return {
            "timestamp": datetime.fromtimestamp(raw_data.get("dt")).isoformat(),
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current_aqi": main.get("aqi"),
            "pollutants": {
                "co": components.get("co"),
                "no": components.get("no"),
                "no2": components.get("no2"),
                "o3": components.get("o3"),
                "so2": components.get("so2"),
                "pm2_5": components.get("pm2_5"),
                "pm10": components.get("pm10"),
                "nh3": components.get("nh3"),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def predict_aqi():
    """Predict AQI for the next 3 hours."""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Fetch current data
        raw_data = fetch_current_data()
        if not raw_data:
            raise HTTPException(status_code=500, detail="Failed to fetch current data")
        
        # Prepare features
        features, dt_obj = prepare_features(raw_data)
        if features is None:
            raise HTTPException(status_code=500, detail="Failed to prepare features")
        
        # Create DataFrame for prediction
        X = pd.DataFrame([features])
        
        # Fill NaN values with 0 for prediction (as the model was trained without lookback features)
        X = X.fillna(0)
        
        # Ensure correct data types
        X['hour'] = X['hour'].astype('int32')
        X['day_of_week'] = X['day_of_week'].astype('int32')
        X['month'] = X['month'].astype('int32')
        X['year'] = X['year'].astype('int32')
        
        for col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'aqi_lag_1h', 'aqi_change_rate']:
            X[col] = X[col].astype('float64')
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Generate 3-hour forecast (simplified: assume constant prediction for each hour)
        forecast = []
        for i in range(3):
            forecast_time = dt_obj + timedelta(hours=i+1)
            forecast.append({
                "hour": i + 1,
                "timestamp": forecast_time.isoformat(),
                "predicted_aqi": float(prediction),
                "aqi_level": get_aqi_level(prediction)
            })
        
        return {
            "current_timestamp": dt_obj.isoformat(),
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current_aqi": raw_data.get("main", {}).get("aqi"),
            "forecast": forecast,
            "model_version": 1,
            "model_name": MODEL_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
