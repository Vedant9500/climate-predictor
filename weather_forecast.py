import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the data and units metadata
preprocessed_data = pd.read_csv('/workspaces/climate-predictor/maharashtra_weather_preprocessed.csv')
raw_data = pd.read_csv('/workspaces/climate-predictor/maharashtra_weather_raw.csv')
units_metadata = pd.read_csv('/workspaces/climate-predictor/maharashtra_weather_units.csv', index_col=0).to_dict()['0']
scaler_params = pd.read_csv('/workspaces/climate-predictor/maharashtra_weather_scaler_params.csv', index_col=0)

print(f"Training model for temperature prediction (units: {units_metadata['tavg']})")

# Assuming the raw data has a 'time' column and a 'tavg' column
raw_data['time'] = pd.to_datetime(raw_data['time'])
raw_data.set_index('time', inplace=True)

# Check for and handle missing values
raw_data['tavg'].fillna(method='ffill', inplace=True)
raw_data['tavg'].fillna(method='bfill', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(raw_data[['tavg']])

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 24  # Change time_step to 24 to predict the next 24 hours
X, Y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))  # Output layer remains the same

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
# Remove the EarlyStopping callback
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=32, verbose=1)

# Save metadata with the model
model_metadata = {
    'units': units_metadata['tavg'],
    'scaler_params': {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0])
    }
}

# Save model and metadata
model.save('/workspaces/climate-predictor/weather_forecast_model_24h.h5')
np.save('/workspaces/climate-predictor/model_metadata.npy', model_metadata)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform the actual values
Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(np.mean(((train_predict - Y_train) ** 2)))
test_rmse = np.sqrt(np.mean(((test_predict - Y_test) ** 2)))

print(f'Train RMSE: {train_rmse} {units_metadata["tavg"]}')
print(f'Test RMSE: {test_rmse} {units_metadata["tavg"]}')