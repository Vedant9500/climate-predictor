import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

# Input test data
RAW_DATA = """
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

# Load the test data into a DataFrame
def load_test_data(raw_data):
    df = pd.read_csv(StringIO(raw_data), header=None, names=['time', 'temperature_2m', 'precipitation', 'wind_speed_10m'])
    df['time'] = pd.to_datetime(df['time'])
    return df

def preprocess_test_data(df, scaler):
    # Select the last 24 hours for prediction
    last_24_hours = df[['temperature_2m', 'precipitation', 'wind_speed_10m']].values
    # Scale the data using the scaler from training
    scaled_data = scaler.transform(last_24_hours)
    # Reshape for model input
    X_input = np.reshape(scaled_data, (1, 24, 3))
    return X_input

def predict_weather(model, X_input, scaler):
    # Generate predictions
    predictions = model.predict(X_input)
    # Rescale predictions back to original values
    predictions_rescaled = scaler.inverse_transform(predictions[0])
    return predictions_rescaled

def display_predictions(predictions):
    print("Predicted Weather for the Next 24 Hours:")
    for i, pred in enumerate(predictions):
        print(f"Hour {i + 1}: Temperature: {pred[0]:.2f}°C, Precipitation: {pred[1]:.2f}mm, Wind Speed: {pred[2]:.2f} m/s")

if __name__ == "__main__":
    # Load and preprocess test data
    test_df = load_test_data(RAW_DATA)

    # Load the trained model
    model = load_model('unified_weather_lstm_model_tpu.h5')

    # Define the MinMaxScaler (Ensure feature range matches training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(test_df[['temperature_2m', 'precipitation', 'wind_speed_10m']])

    # Preprocess the test data for prediction
    X_input = preprocess_test_data(test_df, scaler)

    # Predict the next 24 hours of weather
    predictions = predict_weather(model, X_input, scaler)

    # Display the predictions
    display_predictions(predictions)
