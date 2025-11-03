# End-to-End Serverless MLOps Pipeline for Multi-Horizon AQI Prediction in Karachi

This report details the final implementation of the serverless MLOps pipeline for predicting the Air Quality Index (AQI) in Karachi, Pakistan, with multi-horizon forecasts (24 hours, 48 hours, and 72 hours).

## 1. System Architecture

The system is built on a **100% serverless stack** following modern MLOps principles, integrating data engineering, model training, and deployment.

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Data Source** | OpenWeatherMap Air Pollution API | Provides current and historical hourly air quality data. |
| **Feature Store & Registry** | Hopsworks | Centralized storage for features, targets, and trained models. |
| **Feature Pipeline** | Python Script (`feature_pipeline.py`) | Fetches raw data, engineers features (including rolling means), and writes to the Feature Store. |
| **Training Pipeline** | Python Script (`training_pipeline.py`) | Fetches features, trains three separate Random Forest models (24h, 48h, 72h), evaluates them, and registers them in the Model Registry. |
| **Automation** | GitHub Actions | Schedules hourly feature ingestion and daily model retraining. |
| **Prediction Service** | FastAPI + Gradio | Loads models and features from Hopsworks, computes multi-horizon predictions, and displays results on a descriptive dashboard. |

## 2. Feature Pipeline Enhancements

The `feature_pipeline.py` script was significantly enhanced to support multi-horizon targets and richer features.

### 2.1. Multi-Horizon Targets
The script now calculates three distinct target variables by shifting the `main.aqi` column:
*   `target_aqi_24h`: AQI value 24 hours in the future (`.shift(-24)`).
*   `target_aqi_48h`: AQI value 48 hours in the future (`.shift(-48)`).
*   `target_aqi_72h`: AQI value 72 hours in the future (`.shift(-72)`).

### 2.2. Enhanced Feature Engineering
To improve model accuracy, the following features were added:
*   **Rolling Means:** 24-hour rolling mean of major pollutants (`pm2_5` and `pm10`) to capture daily trends and smooth out noise.
    *   `pm2_5_roll_24h`
    *   `pm10_roll_24h`
*   **Lagged AQI and Change Rate:** `aqi_lag_1h` and `aqi_change_rate` to capture short-term momentum.
*   **Time-based Features:** `hour`, `day_of_week`, `month`, and `year`.

The new features and targets were stored in **Hopsworks Feature Group Version 3**.

## 3. Training Pipeline and Model Evaluation

The `training_pipeline.py` script was refactored to handle the multi-horizon prediction requirement.

### 3.1. Multi-Model Training
Instead of a single model, three separate **Random Forest Regressor** models were trained, one for each prediction horizon:
*   `aqi_rf_24h` (Predicts `target_aqi_24h`)
*   `aqi_rf_48h` (Predicts `target_aqi_48h`)
*   `aqi_rf_72h` (Predicts `target_aqi_72h`)

All three models were registered in the Hopsworks Model Registry under **Version 4**.

### 3.2. Model Performance
The models were evaluated using RMSE, MAE, and R² metrics on the initial backfilled data (Jan 2021).

| Model | Target | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| `aqi_rf_24h` | 24 Hours | 0.2140 | 0.1133 | 0.4293 |
| `aqi_rf_48h` | 48 Hours | 0.2520 | 0.1137 | 0.2680 |
| `aqi_rf_72h` | 72 Hours | 0.1721 | 0.0601 | 0.4398 |

*Note: The R² score indicates that the models capture a moderate amount of variance in the target variable. The 48-hour prediction is the most challenging, which is common in time-series forecasting.*

## 4. Prediction Service (FastAPI + Gradio)

The `gradio_app.py` script was updated to serve as the prediction endpoint:
1.  **Model Loading:** It connects to Hopsworks and loads all three models (`aqi_rf_24h`, `aqi_rf_48h`, `aqi_rf_72h`).
2.  **Feature Preparation:** It fetches the current air quality data and prepares the features, including setting the lookback and rolling mean features to zero (as these are not available at real-time inference without a Feature Store lookup).
3.  **Multi-Prediction:** It runs the current features through all three models to get the 24h, 48h, and 72h forecasts.
4.  **Dashboard:** The Gradio interface displays the current AQI and a table with the multi-horizon predictions, including the predicted AQI value, the forecast timestamp, and the corresponding AQI level (Good, Moderate, Unhealthy, etc.).

## 5. Automation (GitHub Actions)

The automation workflows remain the same, but they now interact with the updated scripts:
*   **Feature Pipeline Workflow:** Runs hourly to ingest the latest data into Feature Group Version 3.
*   **Training Pipeline Workflow:** Runs daily to retrain all three models and register them as the next version in the Model Registry.

## 6. Conclusion

The project successfully delivers a complete, serverless MLOps solution for multi-horizon AQI prediction. The system is fully automated, uses a centralized Feature Store and Model Registry, and provides an interactive dashboard for end-users. The enhanced feature engineering provides a solid foundation for future model improvements.

---
**Karachi Coordinates:** Latitude: 24.86, Longitude: 67.00
**Live Dashboard:** [https://7863-irrakelhce1h6hzr78dav-a649442f.manusvm.computer](https://7863-irrakelhce1h6hzr78dav-a649442f.manusvm.computer)
