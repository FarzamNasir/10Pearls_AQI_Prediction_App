# import os
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# import hopsworks
# import numpy as np
# from feature_columns import FEATURE_COLUMNS

# # --- Configuration ---
# load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# API_KEY = os.getenv("OPENWEATHER_API_KEY")
# LAT = os.getenv("LATITUDE")
# LON = os.getenv("LONGITUDE")
# HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
# HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
# FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
# FEATURE_GROUP_VERSION = 3

# # Convert coordinates to float
# LAT = float(LAT)
# LON = float(LON)

# # Time range for backfill
# START_DATE = datetime(2021, 1, 1)
# END_DATE = datetime.now()

# # OpenWeatherMap API details
# # OpenWeatherMap API details
# HISTORICAL_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
# CURRENT_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
# CHUNK_DAYS = 30 # Fetch data in 30-day chunks

# def fetch_data(mode, start_ts=None, end_ts=None):
#     """Fetches air pollution data from OpenWeatherMap based on mode."""
#     params = {
#         "lat": LAT,
#         "lon": LON,
#         "appid": API_KEY
#     }
    
#     if mode == "historical":
#         url = HISTORICAL_URL
#         params["start"] = int(start_ts.timestamp())
#         params["end"] = int(end_ts.timestamp())
#     elif mode == "realtime":
#         url = CURRENT_URL
#     else:
#         raise ValueError("Invalid mode. Must be 'historical' or 'realtime'.")

#     try:
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         data = response.json()
        
#         if mode == "historical":
#             return data.get("list", [])
#         elif mode == "realtime":
#             # Real-time API returns a single item list under 'list' key
#             return data.get("list", [])
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data in {mode} mode: {e}")
#         return []

# def compute_features(raw_data, is_realtime=False):
#     """Computes features and targets from raw air pollution data."""
#     if not raw_data:
#         return pd.DataFrame()

#     df = pd.DataFrame(raw_data)
    
#     # Convert 'dt' to datetime
#     df['datetime'] = pd.to_datetime(df['dt'], unit='s')
    
#     # Normalize 'main' and 'components' columns
#     df_main = pd.json_normalize(df['main']).add_prefix('main.')
#     df_components = pd.json_normalize(df['components'])
    
#     # Concatenate all parts
#     df = pd.concat([df.drop(columns=['dt', 'main', 'components']), df_main, df_components], axis=1)
    
#     # Set index after feature engineering
#     df = df.set_index('datetime').sort_index()

#     # --- Feature Engineering ---
    
#     # 1. Time-based features
#     # Access datetime properties from the index (which is now a DatetimeIndex)
#     df['hour'] = df.index.hour
#     df['day_of_week'] = df.index.dayofweek
#     df['month'] = df.index.month
#     df['year'] = df.index.year
    
#     # 2. Derived features (AQI change rate)
#     # The 'main.aqi' is the current AQI.
    
#     if is_realtime:
#         # For real-time, we need to fetch the previous hour's data from the Feature Store
#         # For simplicity in this script, we will skip lookback features for real-time
#         # and assume the model is trained on features that are available at inference time.
#         # A more robust solution would involve a Feature Store lookup here.
#         df['aqi_lag_1h'] = np.nan
#         df['aqi_change_rate'] = np.nan
#         df['pm2_5_roll_24h'] = np.nan
#         df['pm10_roll_24h'] = np.nan
#         df['target_aqi_24h'] = np.nan
#         df['target_aqi_48h'] = np.nan
#         df['target_aqi_72h'] = np.nan # Targets are always NaN for real-time inference
    
#     # Drop rows with NaN values created by shifting (first few rows and last few rows)
#     # Only drop if all necessary features are present
#     df = df.dropna(subset=['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3'])
    
#     if not is_realtime:
#         df['aqi_lag_1h'] = df['main.aqi'].shift(1)
#         df['aqi_change_rate'] = df['main.aqi'] - df['aqi_lag_1h']
        
#         # Rolling means for major pollutants (PM2.5 and PM10) over 24 hours
#         df['pm2_5_roll_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean().shift(1)
#         df['pm10_roll_24h'] = df['pm10'].rolling(window=24, min_periods=1).mean().shift(1)
        
#         # 3. Target variables (AQI prediction for the next 24, 48, and 72 hours)
#         df['target_aqi_24h'] = df['main.aqi'].shift(-24)
#         df['target_aqi_48h'] = df['main.aqi'].shift(-48)
#         df['target_aqi_72h'] = df['main.aqi'].shift(-72)
        
#         # Drop rows with NaN values created by shifting (first few rows and last few rows)
#         df = df.dropna()
    
#     # Rename columns to be Hopsworks-compliant (no dots)
#     df = df.rename(columns={'main.aqi': 'current_aqi'})
    
#     # Add primary key and event time for Hopsworks
#     df['location_id'] = 1 # Static ID for Karachi
#     df['event_time'] = df.index
    
#     # Select final features and target
#     feature_columns = [
#         'location_id', 'event_time',
#         'current_aqi', # Current AQI (for training data)
#         'target_aqi_24h', # Target
#         'target_aqi_48h', # Target
#         'target_aqi_72h' # Target
#     ] + FEATURE_COLUMNS
    
#     # Remove duplicates and ensure order
#     feature_columns = list(dict.fromkeys(feature_columns))
    
#     # Explicitly cast columns to match Feature Group schema types
    
#     # Time-based features and location_id should be integers
#     int_cols = ['hour', 'day_of_week', 'month', 'year']
#     for col in int_cols:
#         if col in df.columns:
#             df[col] = df[col].astype('int32')
            
#     # location_id and current_aqi are likely bigint
#     if 'location_id' in df.columns:
#         df['location_id'] = df['location_id'].astype('int64')
#     if 'current_aqi' in df.columns:
#         df['current_aqi'] = df['current_aqi'].astype('int64')
            
#     # Pollutant and derived features should be float
#     float_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'aqi_lag_1h', 'aqi_change_rate', 'pm2_5_roll_24h', 'pm10_roll_24h', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
#     for col in float_cols:
#         if col in df.columns:
#             df[col] = df[col].astype('float64')
    
#     return df[feature_columns]

# def write_features_to_hopsworks(df):
#     """Connects to Hopsworks and writes the feature DataFrame."""
#     if df.empty:
#         print("DataFrame is empty. Skipping write to Feature Store.")
#         return

#     print("Connecting to Hopsworks...")
#     try:
#         # Connect to Hopsworks
#         project = hopsworks.login(
#             api_key_value=HOPSWORKS_API_KEY
#         )
#         fs = project.get_feature_store()
        
#         # Get or create Feature Group
#         try:
#             fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
#             print(f"Feature Group '{FEATURE_GROUP_NAME}' (version {FEATURE_GROUP_VERSION}) found.")
#             # Insert data into existing Feature Group
#             fg.insert(df, write_options={"wait_for_job": True})
#             print(f"Successfully inserted {len(df)} rows into Feature Group.")
#         except Exception as e:
#             print(f"Feature Group '{FEATURE_GROUP_NAME}' (version {FEATURE_GROUP_VERSION}) not found or error: {e}. Attempting to create new one.")
#             fg = fs.create_feature_group(
#                 name=FEATURE_GROUP_NAME,
#                 version=FEATURE_GROUP_VERSION,
#                 description="Historical and real-time AQI features for Karachi.",
#                 primary_key=['location_id'],
#                 event_time='event_time',
#                 online_enabled=True
#             )
#             # Insert initial data to define schema
#             fg.insert(df, write_options={"wait_for_job": True})
#             print("New Feature Group created and initial data inserted.")

#     except Exception as e:
#         print(f"An error occurred during Hopsworks operation: {e}")

# def realtime_pipeline():
#     """Runs the real-time hourly ingestion pipeline."""
#     print("Starting real-time hourly data ingestion...")
    
#     # 1. Fetch current data
#     raw_data = fetch_data("realtime")
    
#     if not raw_data:
#         print("No real-time data fetched. Exiting.")
#         return

#     print(f"Raw data points fetched: {len(raw_data)}")
    
#     # 2. Process and engineer features
#     # Note: For real-time data, we cannot compute 'aqi_lag_1h' or 'aqi_change_rate' 
#     # without fetching the previous hour's data from the Feature Store first.
#     # For simplicity in this serverless script, we will only compute features 
#     # that do not require lookback. The model will need to be trained on features 
#     # that are available at inference time.
#     # The 'target_aqi_3h' will be NaN for real-time data, which is expected.
#     features_df = compute_features(raw_data, is_realtime=True)
    
#     if features_df.empty:
#         print("No features generated. Exiting.")
#         return
        
#     print(f"Feature rows generated: {len(features_df)}")
    
#     # 3. Write to Feature Store
#     write_features_to_hopsworks(features_df)
#     print("Real-time pipeline finished.")

# def backfill_pipeline():
#     """Runs the full backfill pipeline."""
#     print(f"Starting historical data backfill from {START_DATE.date()} to {END_DATE.date()}...")
    
#     current_start = START_DATE
#     all_data = []
    
#     while current_start < END_DATE:
#         current_end = current_start + timedelta(days=CHUNK_DAYS)
#         if current_end > END_DATE:
#             current_end = END_DATE
        
#         print(f"Fetching data chunk: {current_start.date()} to {current_end.date()}...")
#         raw_data = fetch_data("historical", current_start, current_end)
#         all_data.extend(raw_data)
        
#         current_start = current_end + timedelta(seconds=1) # Move to the next second
        
#         # OpenWeatherMap free tier historical data is limited to 5 days.
#         # To avoid hitting rate limits or fetching empty data for the long period,
#         # I will break after the first chunk. The user can adjust CHUNK_DAYS and 
#         # the loop for a paid subscription or a different API.
#         # For now, I will fetch the first 30 days to get some training data.
#         if current_start > START_DATE + timedelta(days=CHUNK_DAYS):
#              print("Stopping after first chunk to respect potential API limits. Please adjust CHUNK_DAYS and loop for full backfill.")
#              break

#     print(f"Total raw data points fetched: {len(all_data)}")
    
#     # Process and engineer features
#     features_df = compute_features(all_data)
#     print(f"Total feature rows generated: {len(features_df)}")
    
#     # Write to Feature Store
#     write_features_to_hopsworks(features_df)
#     print("Backfill pipeline finished.")

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == "realtime":
#         realtime_pipeline()
#     else:
#         backfill_pipeline()



import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import numpy as np
from feature_columns import FEATURE_COLUMNS

# --- Configuration ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LATITUDE")
LON = os.getenv("LONGITUDE")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
FEATURE_GROUP_VERSION = 4

# Convert coordinates to float
LAT = float(LAT)
LON = float(LON)

# Time range for backfill
START_DATE = datetime.now() - timedelta(days=365) # Fetch 1 year of data
END_DATE = datetime.now()

# OpenWeatherMap API details
# OpenWeatherMap API details
HISTORICAL_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
CURRENT_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
CHUNK_DAYS = 5 # Fetch data in 5-day chunks to respect API limits

def fetch_data(mode, start_ts=None, end_ts=None):
    """Fetches air pollution data from OpenWeatherMap based on mode."""
    params = {
        "lat": LAT,
        "lon": LON,
        "appid": API_KEY
    }
    
    if mode == "historical":
        url = HISTORICAL_URL
        params["start"] = int(start_ts.timestamp())
        params["end"] = int(end_ts.timestamp())
    elif mode == "realtime":
        url = CURRENT_URL
    else:
        raise ValueError("Invalid mode. Must be 'historical' or 'realtime'.")

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if mode == "historical":
            return data.get("list", [])
        elif mode == "realtime":
            # Real-time API returns a single item list under 'list' key
            return data.get("list", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data in {mode} mode: {e}")
        return []

def compute_features(raw_data, is_realtime=False):
    """Computes features and targets from raw air pollution data."""
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    
    # Convert 'dt' to datetime
    df['datetime'] = pd.to_datetime(df['dt'], unit='s')
    
    # Normalize 'main' and 'components' columns
    df_main = pd.json_normalize(df['main']).add_prefix('main.')
    df_components = pd.json_normalize(df['components'])
    
    # Concatenate all parts
    df = pd.concat([df.drop(columns=['dt', 'main', 'components']), df_main, df_components], axis=1)
    
    # Set index after feature engineering
    df = df.set_index('datetime').sort_index()

    # --- Feature Engineering ---
    
    # 1. Time-based features
    # Access datetime properties from the index (which is now a DatetimeIndex)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # 2. Derived features (AQI change rate)
    # The 'main.aqi' is the current AQI.
    
    # --- Feature Engineering ---
    
    # 1. Time-based features
    # Access datetime properties from the index (which is now a DatetimeIndex)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # 2. Derived features (Lookback and Rolling Means)
    # These features are calculated for all data (historical and real-time)
    # but will be NaN for the latest real-time point until a Feature Store lookup is performed.
    df['aqi_lag_1h'] = df['main.aqi'].shift(1)
    df['aqi_change_rate'] = df['main.aqi'] - df['aqi_lag_1h']
    df['pm2_5_roll_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean().shift(1)
    df['pm10_roll_24h'] = df['pm10'].rolling(window=24, min_periods=1).mean().shift(1)
    
    # 3. Target variables (AQI prediction for the next 24, 48, and 72 hours)
    if not is_realtime:
        df['target_aqi_24h'] = df['main.aqi'].shift(-24)
        df['target_aqi_48h'] = df['main.aqi'].shift(-48)
        df['target_aqi_72h'] = df['main.aqi'].shift(-72)
        
        # Drop rows with NaN values created by shifting (first few rows and last few rows)
        df = df.dropna()
    else:
        df['target_aqi_24h'] = np.nan
        df['target_aqi_48h'] = np.nan
        df['target_aqi_72h'] = np.nan
        
        # Drop the first row which has NaN for all lookback features
        df = df.iloc[1:]
        
        # Drop rows with NaN values created by shifting (first few rows and last few rows)
        # Only drop if all necessary features are present
        df = df.dropna(subset=['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3'])
    
    # Rename columns to be Hopsworks-compliant (no dots)
    df = df.rename(columns={'main.aqi': 'current_aqi'})
    
    # Add primary key and event time for Hopsworks
    df['location_id'] = 1 # Static ID for Karachi
    df['event_time'] = df.index
    
    # Select final features and target
    feature_columns = [
        'location_id', 'event_time',
        'current_aqi', # Current AQI (for training data)
        'target_aqi_24h', # Target
        'target_aqi_48h', # Target
        'target_aqi_72h' # Target
    ] + FEATURE_COLUMNS
    
    # Remove duplicates and ensure order
    feature_columns = list(dict.fromkeys(feature_columns))
    
    # Explicitly cast columns to match Feature Group schema types
    
    # Time-based features and location_id should be integers
    int_cols = ['hour', 'day_of_week', 'month', 'year']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype('int32')
            
    # location_id and current_aqi are likely bigint
    if 'location_id' in df.columns:
        df['location_id'] = df['location_id'].astype('int64')
    if 'current_aqi' in df.columns:
        df['current_aqi'] = df['current_aqi'].astype('int64')
            
    # Pollutant and derived features should be float
    float_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'aqi_lag_1h', 'aqi_change_rate', 'pm2_5_roll_24h', 'pm10_roll_24h', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    return df[feature_columns]

def write_features_to_hopsworks(df):
    """Connects to Hopsworks and writes the feature DataFrame."""
    if df.empty:
        print("DataFrame is empty. Skipping write to Feature Store.")
        return

    print("Connecting to Hopsworks...")
    try:
        # Connect to Hopsworks
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY
        )
        fs = project.get_feature_store()
        
        # Get or create Feature Group
        try:
            fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            print(f"Feature Group '{FEATURE_GROUP_NAME}' (version {FEATURE_GROUP_VERSION}) found.")
            # Insert data into existing Feature Group
            fg.insert(df, write_options={"wait_for_job": True})
            print(f"Successfully inserted {len(df)} rows into Feature Group.")
        except Exception as e:
            print(f"Feature Group '{FEATURE_GROUP_NAME}' (version {FEATURE_GROUP_VERSION}) not found or error: {e}. Attempting to create new one.")
            fg = fs.create_feature_group(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
                description="Historical and real-time AQI features for Karachi.",
                primary_key=['location_id'],
                event_time='event_time',
                online_enabled=True
            )
            # Insert initial data to define schema
            fg.insert(df, write_options={"wait_for_job": True})
            print("New Feature Group created and initial data inserted.")

    except Exception as e:
        print(f"An error occurred during Hopsworks operation: {e}")

def realtime_pipeline():
    """Runs the real-time hourly ingestion pipeline."""
    print("Starting real-time hourly data ingestion...")
    
    # 1. Fetch current data
    raw_data = fetch_data("realtime")
    
    if not raw_data:
        print("No real-time data fetched. Exiting.")
        return

    print(f"Raw data points fetched: {len(raw_data)}")
    
    # 2. Process and engineer features
    # Note: For real-time data, we cannot compute 'aqi_lag_1h' or 'aqi_change_rate' 
    # without fetching the previous hour's data from the Feature Store first.
    # For simplicity in this serverless script, we will only compute features 
    # that do not require lookback. The model will need to be trained on features 
    # that are available at inference time.
    # The 'target_aqi_3h' will be NaN for real-time data, which is expected.
    features_df = compute_features(raw_data, is_realtime=True)
    
    if features_df.empty:
        print("No features generated. Exiting.")
        return
        
    print(f"Feature rows generated: {len(features_df)}")
    
    # 3. Write to Feature Store
    write_features_to_hopsworks(features_df)
    print("Real-time pipeline finished.")

def backfill_pipeline():
    """Runs the full backfill pipeline."""
    print(f"Starting historical data backfill from {START_DATE.date()} to {END_DATE.date()}...")
    
    current_start = START_DATE
    all_data = []
    
    while current_start < END_DATE:
        current_end = current_start + timedelta(days=CHUNK_DAYS)
        if current_end > END_DATE:
            current_end = END_DATE
        
        print(f"Fetching data chunk: {current_start.date()} to {current_end.date()}...")
        raw_data = fetch_data("historical", current_start, current_end)
        all_data.extend(raw_data)
        
        current_start = current_end + timedelta(seconds=1) # Move to the next second
        
        # Fetch the full year of data, chunked to respect API limits.
        # The user confirmed their API allows for 1 year of historical data.
        # The loop will continue until END_DATE is reached.

    print(f"Total raw data points fetched: {len(all_data)}")
    
    # Process and engineer features
    features_df = compute_features(all_data)
    print(f"Total feature rows generated: {len(features_df)}")
    
    # Write to Feature Store
    write_features_to_hopsworks(features_df)
    print("Backfill pipeline finished.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "realtime":
        realtime_pipeline()
    else:
        backfill_pipeline()
