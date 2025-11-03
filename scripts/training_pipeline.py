import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from feature_columns import FEATURE_COLUMNS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# --- Configuration ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
FEATURE_GROUP_VERSION = 3
MODEL_NAME = "aqi_random_forest_predictor"

# --- Utility Functions ---

def evaluate_model(y_true, y_pred):
    """Calculates and returns RMSE, MAE, and R2 scores."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_and_register_model(target_col, model_version):
    """Fetches data, trains a model, evaluates it, and registers it."""
    print("Connecting to Hopsworks...")
    try:
        # Connect to Hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        # 1. Fetch historical (features, targets) from the Feature Store
        print(f"Fetching feature view for {FEATURE_GROUP_NAME} version {FEATURE_GROUP_VERSION}...")
        
        # Get feature group
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        
        # Read all data from the feature group
        query = fg.select_all()
        df = query.read()
        df = df.dropna(subset=["target_aqi_72h"])
        
        # Define the feature view for model registration later
        try:
            feature_view = fs.get_feature_view(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        except:
            feature_view = fs.create_feature_view(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
                query=query,
                labels=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
            )
        
        print(f"Successfully fetched {len(df)} rows of training data.")

        # Prepare data for training
        # Use the canonical feature list to ensure order
        X = df[FEATURE_COLUMNS]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Train and evaluate the best ML model
        print("Training RandomForestRegressor model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate_model(y_test, y_pred)
        
        print(f"\nModel Evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")
        
        # 3. Store the trained model in the Model Registry
        model_name = f"aqi_rf_{target_col.split('_')[-1]}"
        model_path = f"{model_name}.pkl"
        joblib.dump(model, model_path)
        
        print(f"Saving model and registering to Model Registry as '{model_name}'...")
        
        # Register the model
        aqi_model = mr.python.create_model(
            name=model_name,
            version=model_version,
            metrics={"RMSE": rmse, "MAE": mae, "R2": r2},
            description=f"Random Forest Regressor for AQI prediction in Karachi ({target_col})."
        )
        
        aqi_model.save(
            model_path=model_path
        )
        
        print(f"Model '{model_name}' successfully registered with version {aqi_model.version}.")
        
    except Exception as e:
        print(f"An error occurred during the training pipeline: {e}")

if __name__ == "__main__":
    # Train and register models for all three horizons
    TARGETS = {
        "target_aqi_24h": 4,
        "target_aqi_48h": 4,
        "target_aqi_72h": 4
    }
    
    for target_col, model_version in TARGETS.items():
        train_and_register_model(target_col, model_version)
