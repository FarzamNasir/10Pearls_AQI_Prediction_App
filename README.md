# AQI Predictor for Karachi

This project implements a 100% serverless Machine Learning Operations (MLOps) pipeline to predict the Air Quality Index (AQI) for the city of Karachi, Pakistan, for the next 3 days.

## Project Components

1.  **Feature Pipeline (`scripts/feature_pipeline.py`):** Fetches historical and real-time air pollution data from the OpenWeatherMap API, computes time-based and derived features (like AQI change rate), and stores them in the **Hopsworks Feature Store**.
2.  **Training Pipeline (`scripts/training_pipeline.py`):** Fetches features and targets from the Feature Store, trains and evaluates various ML models (starting with Scikit-learn's Random Forest), and registers the best model in the **Hopsworks Model Registry**.
3.  **Automation (GitHub Actions):** CI/CD workflows to run the feature pipeline hourly and the training pipeline daily.
4.  **Web Application (`app/main.py`):** A FastAPI application with a Gradio/Streamlit dashboard to load the latest model and features, compute 3-day predictions, and display the results.

## Setup and Configuration

1.  **Dependencies:** Install required packages using `pip install -r requirements.txt`.
2.  **Environment Variables:** Configure API keys and project settings in the `.env` file.

## Data Sources

*   **Air Quality Data:** OpenWeatherMap Air Pollution API.
*   **Feature Store & Model Registry:** Hopsworks.
