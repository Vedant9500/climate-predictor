import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Your test data (manual entry)
data = """
2023-12-30 00:00:00,12.8,0.0,0.0
2023-12-30 01:00:00,15.8,0.0,1.6944458
2023-12-30 02:00:00,15.9,0.0,1.5000012000000003
2023-12-30 03:00:00,14.8,0.0,0.0
2023-12-30 04:00:00,20.6,0.0,1.1944454
2023-12-30 05:00:00,24.0,0.0,1.6944458
2023-12-30 06:00:00,26.4,0.0,0.5000004
2023-12-30 07:00:00,28.1,0.0,1.805557
2023-12-30 08:00:00,29.0,0.0,1.5000012000000003
2023-12-30 09:00:00,30.6,0.0,1.5000012000000003
2023-12-30 10:00:00,29.4,0.0,2.1111128
2023-12-30 11:00:00,28.8,0.0,2.8055578000000003
2023-12-30 12:00:00,29.2,0.0,0.0
2023-12-30 13:00:00,23.5,0.0,2.3888908
2023-12-30 14:00:00,22.5,0.0,2.5000020000000003
2023-12-30 15:00:00,20.0,0.0,0.0
2023-12-30 16:00:00,21.2,0.0,3.0000024000000005
2023-12-30 17:00:00,21.1,0.0,2.6944466
2023-12-30 18:00:00,16.4,0.0,0.0
2023-12-30 19:00:00,20.2,0.0,2.1944462000000002
2023-12-30 20:00:00,19.7,0.0,1.6111124000000001
2023-12-30 21:00:00,14.6,0.0,0.5000004
2023-12-30 22:00:00,19.0,0.0,2.1944462000000002
2023-12-30 23:00:00,18.0,0.0,2.3055574000000005
"""

# Create a DataFrame from the data
from io import StringIO
df = pd.read_csv(StringIO(data), header=None, names=['time', 'temperature_2m', 'precipitation', 'wind_speed_10m'])

# Convert 'time' to datetime format
df['time'] = pd.to_datetime(df['time'])

# Preprocess the data for prediction (using the last 24 hours)
last_24_hours = df[['temperature_2m', 'precipitation', 'wind_speed_10m']].values

# Normalize the features using MinMaxScaler (fit using the data used during training)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(last_24_hours)

# Reshape data for model input (assuming model expects 3D input with shape (samples, timesteps, features))
X_input = np.reshape(scaled_data, (1, 24, 3))  # 1 sample, 24 timesteps, 3 features

# Load your trained model
model = load_model('unified_weather_lstm_model.h5')

# Predict the next 24 hours
predictions = model.predict(X_input)

# Rescale the predictions back to original values (if necessary)
predictions_rescaled = scaler.inverse_transform(predictions[0])

# Print the predicted weather for the next 24 hours
print("Predicted Weather for the Next 24 Hours:")
for i, pred in enumerate(predictions_rescaled):
    print(f"Hour {i + 1}: Temperature: {pred[0]:.2f}°C, Precipitation: {pred[1]:.2f}mm, Wind Speed: {pred[2]:.2f} m/s")
