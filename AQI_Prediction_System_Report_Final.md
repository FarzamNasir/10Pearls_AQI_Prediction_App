# End-to-End Serverless MLOps Pipeline for Multi-Horizon AQI Prediction in Karachi

This report details the final implementation of the serverless MLOps pipeline for predicting the Air Quality Index (AQI) in Karachi, Pakistan, with multi-horizon forecasts (24 hours, 48 hours, and 72 hours).

## 1. System Architecture

The system is built on a **100% serverless stack** following modern MLOps principles, integrating data engineering, model training, and deployment.

| Component                    | Technology                             | Role                                                                                                                                                                                               |
| :--------------------------- | :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Source**              | OpenWeatherMap Air Pollution API       | Provides current and historical hourly air quality data.                                                                                                                                           |
| **Feature Store & Registry** | Hopsworks                              | Centralized storage for features, targets, and trained models.                                                                                                                                     |
| **Feature Pipeline**         | Python Script (`feature_pipeline.py`)  | Fetches raw data, engineers features (including rolling means), and writes to the Feature Store.                                                                                                   |
| **Training Pipeline**        | Python Script (`training_pipeline.py`) | Fetches features, trains multiple ML models (XGBoost, Random Forest, Gradient Boosting, Linear Regression), selects the best performer for each horizon, and registers them in the Model Registry. |
| **Automation**               | GitHub Actions                         | Schedules hourly feature ingestion and daily model retraining.                                                                                                                                     |
| **Prediction Service**       | FastAPI + Gradio                       | Loads models and features from Hopsworks, computes multi-horizon predictions, and displays results on an interactive dashboard.                                                                    |

## 2. Feature Pipeline Enhancements

The `feature_pipeline.py` script was significantly enhanced to support multi-horizon targets and richer features.

### 2.1. Multi-Horizon Targets

The script now calculates three distinct target variables by shifting the `main.aqi` column:

- `target_aqi_24h`: AQI value 24 hours in the future (`.shift(-24)`).
- `target_aqi_48h`: AQI value 48 hours in the future (`.shift(-48)`).
- `target_aqi_72h`: AQI value 72 hours in the future (`.shift(-72)`).

### 2.2. Enhanced Feature Engineering

To improve model accuracy, the following features were added:

- **Rolling Means:** 24-hour rolling mean of major pollutants (`pm2_5` and `pm10`) to capture daily trends and smooth out noise.
  - `pm2_5_roll_24h`
  - `pm10_roll_24h`
- **Lagged AQI and Change Rate:** `aqi_lag_1h` and `aqi_change_rate` to capture short-term momentum.
- **Time-based Features:** `hour`, `day_of_week`, `month`, and `year`.
- **Pollutant Features:** Current levels of CO, NO, NO2, O3, SO2, PM2.5, PM10, and NH3.

### 2.3. Data Collection

The system collected **one year of historical data** (365 days) from OpenWeatherMap, resulting in **8,447 training samples** after feature engineering and cleaning.

The features and targets were stored in **Hopsworks Feature Group Version 4**.

## 3. Training Pipeline and Model Selection

The `training_pipeline.py` script was designed with an **automated model selection process** to ensure optimal performance for each prediction horizon.

### 3.1. Multi-Algorithm Training

For each of the three prediction horizons, four different machine learning algorithms were trained:

- **XGBoost Regressor** (n_estimators=200, max_depth=10, learning_rate=0.1)
- **Random Forest Regressor** (n_estimators=200, max_depth=15)
- **Gradient Boosting Regressor** (n_estimators=200, max_depth=10)
- **Linear Regression** (baseline model)

The system automatically selected the best-performing model based on RMSE (Root Mean Squared Error) and registered only the winning model for each horizon in the Hopsworks Model Registry.

### 3.2. Training Results and Best Model Selection

#### Dataset Split

- **Training Set:** 6,757 samples (80%)
- **Test Set:** 1,690 samples (20%)

#### 24-Hour Prediction Models

| Model            | RMSE       | MAE        | R² Score   |
| :--------------- | :--------- | :--------- | :--------- |
| RandomForest     | 0.4360     | 0.2822     | 0.8590     |
| **XGBoost** ✓    | **0.3815** | **0.2299** | **0.8920** |
| GradientBoosting | 0.3853     | 0.2298     | 0.8899     |
| LinearRegression | 0.7150     | 0.5415     | 0.6207     |

**Winner:** XGBoost achieved the lowest RMSE and highest R² score, explaining **89.20%** of the variance in 24-hour AQI predictions.

#### 48-Hour Prediction Models

| Model            | RMSE       | MAE        | R² Score   |
| :--------------- | :--------- | :--------- | :--------- |
| RandomForest     | 0.3847     | 0.2509     | 0.8925     |
| **XGBoost** ✓    | **0.3283** | **0.2027** | **0.9217** |
| GradientBoosting | 0.3448     | 0.2099     | 0.9136     |
| LinearRegression | 0.8228     | 0.6511     | 0.5082     |

**Winner:** XGBoost significantly outperformed other models, achieving an impressive **92.17% R² score**, making it the most accurate 48-hour predictor.

#### 72-Hour Prediction Models

| Model            | RMSE       | MAE        | R² Score   |
| :--------------- | :--------- | :--------- | :--------- |
| RandomForest     | 0.4283     | 0.2649     | 0.8659     |
| **XGBoost** ✓    | **0.3845** | **0.2239** | **0.8919** |
| GradientBoosting | 0.3923     | 0.2209     | 0.8875     |
| LinearRegression | 0.8962     | 0.7192     | 0.4130     |

**Winner:** XGBoost maintained superior performance for long-term predictions, explaining **89.19%** of variance in 72-hour AQI forecasts.

### 3.3. Registered Models

All three winning models were registered in the Hopsworks Model Registry as **Version 1**:

- `aqi_predictor_24h` (XGBoost, RMSE: 0.3815, R²: 0.8920)
- `aqi_predictor_48h` (XGBoost, RMSE: 0.3283, R²: 0.9217)
- `aqi_predictor_72h` (XGBoost, RMSE: 0.3845, R²: 0.8919)

### 3.4. Key Insights

1. **XGBoost Dominated:** XGBoost was selected as the best model for all three horizons, demonstrating its superior capability in handling complex temporal patterns in air quality data.

2. **Gradient Boosting as Runner-Up:** Gradient Boosting consistently performed as the second-best model, with performance very close to XGBoost.

3. **Strong Predictive Power:** All ensemble models (XGBoost, Random Forest, Gradient Boosting) achieved R² scores above 0.85, indicating strong predictive capabilities.

4. **48-Hour Sweet Spot:** Interestingly, the 48-hour prediction achieved the highest R² score (0.9217), suggesting that this time horizon captures optimal temporal patterns without being too far into the future.

5. **Linear Baseline:** Linear Regression performed significantly worse (R² < 0.62), confirming the non-linear nature of AQI dynamics and justifying the use of ensemble methods.

## 4. Prediction Service (FastAPI + Gradio)

Two deployment options were implemented for real-time predictions:

### 4.1. Gradio Web Application (`gradio_app.py`)

An interactive web interface that:

1.  **Loads All Models:** Connects to Hopsworks and downloads all three XGBoost models with retry logic and exponential backoff.
2.  **Fetches Real-Time Data:** Retrieves current air quality data from OpenWeatherMap API.
3.  **Feature Preparation:** Prepares the 16-feature input vector in the correct order as defined in `feature_columns.py`.
4.  **Multi-Horizon Predictions:** Generates predictions for 24h, 48h, and 72h horizons simultaneously.
5.  **Interactive Dashboard:** Displays:
    - Current AQI with health impact assessment
    - Multi-horizon forecast table with timestamps and AQI levels
    - Health recommendations based on predicted AQI
    - Model type information (shows which algorithm was selected)
    - Comprehensive AQI scale reference

### 4.2. FastAPI REST API (`main.py`)

RESTful API endpoints for programmatic access:

- `GET /` - API information and available endpoints
- `GET /health` - Health check and model loading status
- `GET /models` - Information about loaded models
- `GET /current` - Current AQI data with pollutant levels
- `GET /predict` - Multi-horizon predictions (24h, 48h, 72h)
- `GET /predict/{horizon}` - Specific horizon prediction (e.g., `/predict/24h`)

Both services include:

- Automatic retry logic for model downloads (3 attempts with exponential backoff)
- Partial loading support (works even if some models fail to load)
- Comprehensive error handling and logging
- Health recommendations based on AQI levels

## 5. Automation (GitHub Actions)

Two automated workflows ensure the system stays up-to-date:

### 5.1. Feature Pipeline Workflow

- **Schedule:** Runs hourly
- **Actions:**
  - Fetches latest air quality data from OpenWeatherMap
  - Engineers features with proper temporal ordering
  - Inserts new data into Feature Group Version 4
- **Purpose:** Maintains a continuously updated feature store for training and inference

### 5.2. Training Pipeline Workflow

- **Schedule:** Runs daily
- **Actions:**
  - Fetches all historical data from Feature Group
  - Trains 4 models × 3 horizons = 12 total model training runs
  - Evaluates all models using RMSE, MAE, and R² metrics
  - Selects and registers only the best model for each horizon
  - Automatically increments model version numbers
- **Purpose:** Ensures models continuously improve as more data becomes available

## 6. Model Performance Analysis

### 6.1. Performance Trends Across Horizons

| Metric       | 24h    | 48h    | 72h    | Trend    |
| :----------- | :----- | :----- | :----- | :------- |
| **RMSE**     | 0.3815 | 0.3283 | 0.3845 | 48h best |
| **MAE**      | 0.2299 | 0.2027 | 0.2239 | 48h best |
| **R² Score** | 0.8920 | 0.9217 | 0.8919 | 48h best |

**Key Observation:** The 48-hour model achieved the best performance across all metrics, which is counterintuitive but can be explained by:

- Optimal balance between short-term noise and long-term uncertainty
- Sufficient temporal distance to capture meaningful patterns
- Weather pattern cycles that align with this time horizon

### 6.2. Error Magnitude Interpretation

On the OpenWeatherMap AQI scale (1-5):

- **RMSE of 0.38:** Predictions are typically within ±0.38 AQI units
- **MAE of 0.22-0.23:** Average prediction error is less than a quarter of an AQI level
- **R² > 0.89:** Models explain over 89% of AQI variance

This level of accuracy is highly suitable for practical air quality forecasting applications.

## 7. Feature Importance

Based on XGBoost's inherent feature importance ranking (from training logs), the most influential features for AQI prediction are:

1.  **PM2.5 and PM10 levels:** Primary contributors to AQI
2.  **Rolling means (pm2_5_roll_24h, pm10_roll_24h):** Capture temporal trends
3.  **Time-based features (hour, day_of_week, month):** Capture cyclical patterns
4.  **Other pollutants (NO2, O3, SO2):** Secondary but important contributors

## 8. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenWeatherMap API                        │
│                  (Real-time Air Quality Data)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Feature Pipeline (Hourly)                      │
│  • Fetch data • Engineer features • Update Feature Store    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Hopsworks Feature Store (Version 4)                │
│           • 8,447 samples • 16 features • 3 targets         │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             ▼                                ▼
┌──────────────────────────┐    ┌───────────────────────────┐
│  Training Pipeline       │    │  Prediction Services      │
│  (Daily)                 │    │  • Gradio Web App         │
│  • Train 4 algorithms    │    │  • FastAPI REST API       │
│  • Select best models    │    │  • Load models v1         │
│  • Register to Registry  │    │  • Real-time predictions  │
└──────────┬───────────────┘    └───────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│           Hopsworks Model Registry (Version 1)               │
│  • aqi_predictor_24h (XGBoost, R²: 0.8920)                  │
│  • aqi_predictor_48h (XGBoost, R²: 0.9217)                  │
│  • aqi_predictor_72h (XGBoost, R²: 0.8919)                  │
└─────────────────────────────────────────────────────────────┘
```

## 9. Technical Specifications

### 9.1. Model Hyperparameters (XGBoost Winners)

```python
XGBRegressor(
    n_estimators=200,      # Number of boosting rounds
    max_depth=10,          # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    subsample=0.8,         # Fraction of samples per tree
    colsample_bytree=0.8,  # Fraction of features per tree
    random_state=42,       # Reproducibility
    n_jobs=-1              # Parallel processing
)
```

### 9.2. Feature Vector (16 dimensions)

```python
FEATURE_COLUMNS = [
    'hour', 'day_of_week', 'month', 'year',           # Time features (4)
    'aqi_lag_1h', 'aqi_change_rate',                  # Lag features (2)
    'pm2_5_roll_24h', 'pm10_roll_24h',               # Rolling means (2)
    'co', 'no', 'no2', 'o3', 'so2',                  # Gas pollutants (5)
    'pm2_5', 'pm10', 'nh3'                           # Particulate matter (3)
]
```

### 9.3. AQI Level Mapping

| AQI Value | Level     | Health Impact                    |
| :-------- | :-------- | :------------------------------- |
| 1         | Good      | Air quality is satisfactory      |
| 2         | Fair      | Acceptable for most people       |
| 3         | Moderate  | Sensitive groups may be affected |
| 4         | Poor      | General public may be affected   |
| 5         | Very Poor | Health alert for everyone        |

## 10. Future Enhancements

1. **Extended Horizons:** Add 96-hour (4-day) and 120-hour (5-day) predictions
2. **Weather Integration:** Incorporate weather forecasts (temperature, humidity, wind) as features
3. **Ensemble Stacking:** Combine multiple models for improved predictions
4. **Real-time Monitoring:** Add model drift detection and automatic retraining triggers
5. **Multi-Location:** Extend to other cities in Pakistan (Lahore, Islamabad, etc.)
6. **Mobile Application:** Develop iOS/Android apps for broader accessibility
7. **Alert System:** Push notifications when AQI is predicted to reach unhealthy levels

## 11. Conclusion

The project successfully delivers a **production-ready, serverless MLOps solution** for multi-horizon AQI prediction in Karachi. Key achievements include:

✅ **Automated Model Selection:** XGBoost emerged as the best algorithm across all horizons through systematic comparison  
✅ **High Accuracy:** R² scores exceeding 89% demonstrate strong predictive power  
✅ **Robust Architecture:** Feature Store, Model Registry, and automated pipelines ensure maintainability  
✅ **Real-time Predictions:** Both web interface and REST API provide instant multi-horizon forecasts  
✅ **Continuous Improvement:** Hourly data ingestion and daily retraining keep models fresh

The system provides actionable air quality forecasts that can help Karachi residents make informed decisions about outdoor activities and health precautions.

---

## 12. Project Information

**Location:** Karachi, Pakistan  
**Coordinates:** Latitude 24.86°N, Longitude 67.00°E  
**Data Source:** OpenWeatherMap Air Pollution API  
**Training Data:** 8,447 hourly samples (1 year)  
**Feature Store:** Hopsworks Version 4  
**Model Registry:** Hopsworks Version 1  
**Best Algorithm:** XGBoost (all horizons)  
**Deployment:** Gradio + FastAPI  
**Automation:** GitHub Actions (Hourly + Daily)

**Model URLs:**

- [24h Model](https://c.app.hopsworks.ai:443/p/1274004/models/aqi_predictor_24h/1)
- [48h Model](https://c.app.hopsworks.ai:443/p/1274004/models/aqi_predictor_48h/1)
- [72h Model](https://c.app.hopsworks.ai:443/p/1274004/models/aqi_predictor_72h/1)

---

_Report Last Updated: November 9, 2025_  
_Project Status: Production Ready_
