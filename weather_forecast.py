import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load the combined data from the unified CSV
def load_combined_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    df['time'] = pd.to_datetime(df['time'])
    return df

# Preprocessing the data (handle missing values and normalize)
def preprocess_data(df):
    # Handle missing values (you can use fillna or dropna depending on the dataset)
    df = df.dropna()

    # Normalize the features (temperature, precipitation, wind speed)
    scaler = MinMaxScaler()
    df[['temperature_2m', 'precipitation', 'wind_speed_10m']] = scaler.fit_transform(df[['temperature_2m', 'precipitation', 'wind_speed_10m']])
    
    return df, scaler

# Create sequences for LSTM (24 hours input to predict 24 hours output)
def create_sequences(df, look_back=24, forecast_horizon=24):
    sequences = []
    targets = []
    
    for i in range(len(df) - look_back - forecast_horizon):
        # Select data for the last 24 hours (input sequence)
        sequence = df.iloc[i:i+look_back][['temperature_2m', 'precipitation', 'wind_speed_10m']].values
        # Select data for the next 24 hours as the target (output sequence)
        target = df.iloc[i+look_back:i+look_back+forecast_horizon][['temperature_2m', 'precipitation', 'wind_speed_10m']].values
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Create and compile the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))  # Ensure this is set to return sequences for 24 time steps
    model.add(Dropout(0.2))
    model.add(Dense(units=3))  # Predict 3 features: temperature, precipitation, wind speed for 24 time steps
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model on the combined data
def train_combined_model(file_path):
    # Load the unified data
    df = load_combined_data(file_path)
    
    # Preprocess the data
    df, scaler = preprocess_data(df)
    
    # Create sequences for the LSTM
    X, y = create_sequences(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Define the model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate and plot the loss curve
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Unified Model Training Loss')
    plt.legend()
    plt.show()

    # Save the model
    model.save('unified_weather_lstm_model.h5')
    
    return model, scaler

# Example usage for combined training across all cities
file_path = 'Unified_data.csv'
model, scaler = train_combined_model(file_path)
