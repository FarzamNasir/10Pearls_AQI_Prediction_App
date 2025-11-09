# import os
# import pandas as pd
# from dotenv import load_dotenv
# import hopsworks
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from feature_columns import FEATURE_COLUMNS
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib
# import numpy as np

# # --- Configuration ---
# load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
# HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
# FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
# FEATURE_GROUP_VERSION = 3
# MODEL_NAME = "aqi_random_forest_predictor"

# # --- Utility Functions ---

# def evaluate_model(y_true, y_pred):
#     """Calculates and returns RMSE, MAE, and R2 scores."""
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return rmse, mae, r2

# def train_and_register_model(target_col, model_version):
#     """Fetches data, trains a model, evaluates it, and registers it."""
#     print("Connecting to Hopsworks...")
#     try:
#         # Connect to Hopsworks
#         project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
#         fs = project.get_feature_store()
#         mr = project.get_model_registry()
        
#         # 1. Fetch historical (features, targets) from the Feature Store
#         print(f"Fetching feature view for {FEATURE_GROUP_NAME} version {FEATURE_GROUP_VERSION}...")
        
#         # Get feature group
#         fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        
#         # Read all data from the feature group
#         query = fg.select_all()
#         df = query.read()
#         df = df.dropna(subset=["target_aqi_72h"])
        
#         # Define the feature view for model registration later
#         try:
#             feature_view = fs.get_feature_view(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
#         except:
#             feature_view = fs.create_feature_view(
#                 name=FEATURE_GROUP_NAME,
#                 version=FEATURE_GROUP_VERSION,
#                 query=query,
#                 labels=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
#             )
        
#         print(f"Successfully fetched {len(df)} rows of training data.")

#         # Prepare data for training
#         # Use the canonical feature list to ensure order
#         X = df[FEATURE_COLUMNS]
#         y = df[target_col]
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # 2. Train and evaluate the best ML model
#         print("Training RandomForestRegressor model...")
#         model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         model.fit(X_train, y_train)
        
#         # Predict and evaluate
#         y_pred = model.predict(X_test)
#         rmse, mae, r2 = evaluate_model(y_test, y_pred)
        
#         print(f"\nModel Evaluation:")
#         print(f"  RMSE: {rmse:.4f}")
#         print(f"  MAE: {mae:.4f}")
#         print(f"  R2: {r2:.4f}")
        
#         # 3. Store the trained model in the Model Registry
#         model_name = f"aqi_rf_{target_col.split('_')[-1]}"
#         model_path = f"{model_name}.pkl"
#         joblib.dump(model, model_path)
        
#         print(f"Saving model and registering to Model Registry as '{model_name}'...")
        
#         # Register the model
#         aqi_model = mr.python.create_model(
#             name=model_name,
#             version=model_version,
#             metrics={"RMSE": rmse, "MAE": mae, "R2": r2},
#             description=f"Random Forest Regressor for AQI prediction in Karachi ({target_col})."
#         )
        
#         aqi_model.save(
#             model_path=model_path
#         )
        
#         print(f"Model '{model_name}' successfully registered with version {aqi_model.version}.")
        
#     except Exception as e:
#         print(f"An error occurred during the training pipeline: {e}")

# if __name__ == "__main__":
#     # Train and register models for all three horizons
#     TARGETS = {
#         "target_aqi_24h": 4,
#         "target_aqi_48h": 4,
#         "target_aqi_72h": 4
#     }
    
#     for target_col, model_version in TARGETS.items():
#         train_and_register_model(target_col, model_version)

import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from feature_columns import FEATURE_COLUMNS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# --- Configuration ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
FEATURE_GROUP_VERSION = 4  # Updated to match your feature pipeline
MODEL_VERSION = 1  # Will be incremented for each target

# --- Model Configurations ---
MODELS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    "LinearRegression": LinearRegression(
        n_jobs=-1
    )
}

# --- Utility Functions ---

def evaluate_model(y_true, y_pred):
    """Calculates and returns RMSE, MAE, and R2 scores."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_and_compare_models(X_train, X_test, y_train, y_test, target_col):
    """Trains multiple models and returns the best one with its metrics."""
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Training models for {target_col}")
    print(f"{'='*60}\n")
    
    for model_name, model in MODELS.items():
        print(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Evaluate
        rmse, mae, r2 = evaluate_model(y_test, y_pred)
        
        results[model_name] = {
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "predictions": y_pred
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}\n")
    
    # Find the best model based on RMSE (lower is better)
    best_model_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best_result = results[best_model_name]
    
    print(f"{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"  RMSE: {best_result['rmse']:.4f}")
    print(f"  MAE:  {best_result['mae']:.4f}")
    print(f"  R2:   {best_result['r2']:.4f}")
    print(f"{'='*60}\n")
    
    return best_model_name, best_result

def train_and_register_best_model(target_col, model_version):
    """Fetches data, trains multiple models, selects the best, and registers it."""
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
        
        # Drop rows with missing target values
        df = df.dropna(subset=[target_col])
        
        print(f"Successfully fetched {len(df)} rows of training data.")
        
        # Define the feature view for model registration later
        try:
            feature_view = fs.get_feature_view(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            print(f"Feature view '{FEATURE_GROUP_NAME}' found.")
        except:
            print(f"Creating feature view '{FEATURE_GROUP_NAME}'...")
            feature_view = fs.create_feature_view(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
                query=query,
                labels=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
            )
            print("Feature view created successfully.")

        # Prepare data for training
        # Use the canonical feature list to ensure order
        X = df[FEATURE_COLUMNS]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # 2. Train multiple models and select the best one
        best_model_name, best_result = train_and_compare_models(
            X_train, X_test, y_train, y_test, target_col
        )
        
        # 3. Store the best model in the Model Registry
        horizon = target_col.split('_')[-1]  # Extract '24h', '48h', or '72h'
        model_name = f"aqi_predictor_{horizon}"
        model_path = f"{model_name}_v{model_version}.pkl"
        
        # Save the model locally
        joblib.dump(best_result["model"], model_path)
        
        print(f"Saving best model ({best_model_name}) and registering to Model Registry as '{model_name}'...")
        
        # Register the model with detailed metadata
        # Note: metrics can only contain numeric values
        aqi_model = mr.python.create_model(
            name=model_name,
            version=model_version,
            metrics={
                "RMSE": float(best_result['rmse']),
                "MAE": float(best_result['mae']),
                "R2": float(best_result['r2'])
            },
            description=f"Best AQI predictor for {horizon} forecast horizon in Karachi. Model Type: {best_model_name}. Trained on {len(X_train)} samples."
        )
        
        aqi_model.save(model_path)
        
        print(f"✓ Model '{model_name}' successfully registered with version {model_version}.")
        print(f"  Model Type: {best_model_name}")
        print(f"  RMSE: {best_result['rmse']:.4f}")
        print(f"  MAE: {best_result['mae']:.4f}")
        print(f"  R2: {best_result['r2']:.4f}\n")
        
        # Clean up local model file
        if os.path.exists(model_path):
            os.remove(model_path)
        
    except Exception as e:
        print(f"An error occurred during the training pipeline: {e}")
        raise

if __name__ == "__main__":
    # Train and register the best model for all three horizons
    TARGETS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
    
    print("\n" + "="*60)
    print("AQI PREDICTION MODEL TRAINING PIPELINE")
    print("="*60 + "\n")
    
    for idx, target_col in enumerate(TARGETS, start=1):
        print(f"\n{'#'*60}")
        print(f"# Training Pipeline {idx}/3: {target_col}")
        print(f"{'#'*60}\n")
        
        train_and_register_best_model(target_col, MODEL_VERSION)
    
    print("\n" + "="*60)
    print("ALL TRAINING PIPELINES COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")