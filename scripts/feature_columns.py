# Canonical list of feature columns for the AQI prediction model.
# The order MUST be maintained in both training and inference.

FEATURE_COLUMNS = [
    'hour',
    'day_of_week',
    'month',
    'year',
    'aqi_lag_1h',
    'aqi_change_rate',
    'pm2_5_roll_24h',
    'pm10_roll_24h',
    'co',
    'no',
    'no2',
    'o3',
    'so2',
    'pm2_5',
    'pm10',
    'nh3'
]
