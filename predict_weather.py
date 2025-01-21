import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load your trained model (assuming it's saved as 'weather_model.h5')
model = load_model('weather_model.h5')

# Load the last 24 hours of data for Pune
data = pd.read_csv('Pune_weather_data.csv')

# Convert 'time' to datetime
data['time'] = pd.to_datetime(data['time'])

# Preprocess the data for prediction (use the last 24 hours of data)
last_24_hours = data.tail(24)

# Extract relevant features (time, temperature, precipitation, wind speed)
features = last_24_hours[['temperature_2m', 'precipitation', 'wind_speed_10m']].values

# Normalize the features using the same scaler that was used during training
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Prepare the input data for the model (reshape to 3D for LSTM)
X_input = np.reshape(features_scaled, (1, 24, 3))  # 1 sample, 24 timesteps, 3 features

# Predict the next 24 hours
predictions = model.predict(X_input)

# Rescale the predictions back to original values (if necessary)
predictions_rescaled = scaler.inverse_transform(predictions[0])

# Print the predicted weather for the next 24 hours
print("Predicted Weather for the Next 24 Hours:")
for i, pred in enumerate(predictions_rescaled):
    print(f"Hour {i + 1}: Temperature: {pred[0]:.2f}°C, Precipitation: {pred[1]:.2f}mm, Wind Speed: {pred[2]:.2f} m/s")
