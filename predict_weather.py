import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Function to fetch current weather data
def fetch_current_weather(location):
    api_key = '50324bb67be34dd28a2151821252001'  # Replace with your actual API key
    url = f'http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}'
    response = requests.get(url)
    data = response.json()
    return data['current']['temp_c']

# Function to preprocess the fetched data
def preprocess_data(temp, scaler):
    temp_df = pd.DataFrame({'tavg': [temp]})
    scaled_temp = scaler.transform(temp_df)
    return scaled_temp[0]

# Load model metadata
model_metadata = np.load('/workspaces/climate-predictor/model_metadata.npy', allow_pickle=True).item()

# Load the scaler parameters
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.min_ = np.array([model_metadata['scaler_params']['min']])
scaler.scale_ = np.array([1 / (model_metadata['scaler_params']['max'] - model_metadata['scaler_params']['min'])])

# Correctly set the scaler's scale_ attribute
scaler.scale_ = np.array([1 / (model_metadata['scaler_params']['max'] - model_metadata['scaler_params']['min'])])
scaler.data_min_ = np.array([model_metadata['scaler_params']['min']])
scaler.data_max_ = np.array([model_metadata['scaler_params']['max']])

# Fetch current weather data
location = 'Mumbai'  # Replace with the desired location
current_temp = fetch_current_weather(location)

# Preprocess the fetched data
scaled_temp = preprocess_data(current_temp, scaler)

# Prepare the data for LSTM
time_step = 24
X_input = np.array([scaled_temp for _ in range(time_step)]).reshape(1, time_step, 1)

# Load the trained model
model = load_model('/workspaces/climate-predictor/weather_forecast_model_24h.h5')

# Predict the next 24 hours
predictions = model.predict(X_input)
predictions = scaler.inverse_transform(predictions)

# Print the predictions with timestamps
current_time = datetime.now()
print(f'Predicted temperatures for the next 24 hours in {location}:')
for i, temp in enumerate(predictions.flatten()):
    time = (current_time + timedelta(hours=i)).strftime('%I %p')
    print(f'{time}: {temp:.2f}°C')
