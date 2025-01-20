import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the preprocessed data
preprocessed_data = pd.read_csv('/workspaces/climate-predictor/maharashtra_weather_preprocessed.csv')

# Assuming the preprocessed data has a 'time' column and a 'tavg' column
preprocessed_data['time'] = pd.to_datetime(preprocessed_data['time'])
preprocessed_data.set_index('time', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(preprocessed_data[['tavg']])

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 24  # Change time_step to 24 to match the training
X, Y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Load the updated model
model = load_model('/workspaces/climate-predictor/weather_forecast_model_24h.h5')

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform the actual values
Y_train = scaler.inverse_transform([Y_train])
Y_test = scaler.inverse_transform([Y_test])

# Calculate RMSE
train_rmse = np.sqrt(np.mean(((train_predict - Y_train.T) ** 2)))
test_rmse = np.sqrt(np.mean(((test_predict - Y_test.T) ** 2)))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
