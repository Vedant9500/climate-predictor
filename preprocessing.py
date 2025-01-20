from meteostat import Point, Daily
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Unit conversion functions
def convert_wind_to_kmph(wind_speed_ms):
    return wind_speed_ms * 3.6  # Convert m/s to km/h

def convert_pressure_to_hpa(pressure):
    return pressure / 100  # Convert Pa to hPa

# Define major locations in Maharashtra with (latitude, longitude, elevation)
locations = {
    "Mumbai": Point(19.0760, 72.8777, 14),
    "Pune": Point(18.5204, 73.8567, 560),
    "Nagpur": Point(21.1458, 79.0882, 310),
    "Nashik": Point(19.9975, 73.7898, 700),
    "Aurangabad": Point(19.8762, 75.3433, 568),
    "Kolhapur": Point(16.7050, 74.2433, 570),
    "Solapur": Point(17.6599, 75.9064, 457),
}

# Define date range
start = datetime(2014, 1, 1)
end = datetime(2024, 1, 1)

# Fetch data for all locations
dataframes = []

for city, point in locations.items():
    print(f"Fetching data for {city}...")
    # Fetch daily weather data
    data = Daily(point, start, end)
    df = data.fetch()
    
    # Convert units
    df['wspd'] = df['wspd'].apply(convert_wind_to_kmph)  # Convert wind speed to km/h
    df['wpgt'] = df['wpgt'].apply(convert_wind_to_kmph)  # Convert wind gusts to km/h
    df['pres'] = df['pres'].apply(convert_pressure_to_hpa)  # Convert pressure to hPa
    
    # Add metadata for units
    df['units'] = {
        'tavg': '°C',
        'tmin': '°C',
        'tmax': '°C',
        'prcp': 'mm',
        'snow': 'cm',
        'wspd': 'km/h',
        'wpgt': 'km/h',
        'pres': 'hPa',
        'tsun': 'hours'
    }
    
    # Add a column for the city name
    df['City'] = city
    # Append to the list
    dataframes.append(df)

# Combine all data into a single DataFrame
maharashtra_data = pd.concat(dataframes)

# Reset index and sort by date for clarity
maharashtra_data.reset_index(inplace=True)
maharashtra_data.sort_values(by=['time', 'City'], inplace=True)

# Save the units metadata separately
units_metadata = {
    'tavg': '°C',
    'tmin': '°C',
    'tmax': '°C',
    'prcp': 'mm',
    'snow': 'cm',
    'wspd': 'km/h',
    'wpgt': 'km/h',
    'pres': 'hPa',
    'tsun': 'hours'
}

# Save units metadata
pd.Series(units_metadata).to_csv('maharashtra_weather_units.csv')

# Display the first few rows of the raw data
print(maharashtra_data.head())

# Save raw data to a CSV file (optional)
maharashtra_data.to_csv("maharashtra_weather_raw.csv", index=False)

# --- Preprocessing ---

# Handle missing values
maharashtra_data.fillna(method='ffill', inplace=True)  # Forward fill
maharashtra_data.fillna(method='bfill', inplace=True)  # Backward fill

# Normalize numerical columns (e.g., temperature, precipitation)
scaler = MinMaxScaler()

# Select numerical columns for normalization
numerical_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'wpgt', 'pres', 'tsun']
maharashtra_data[numerical_columns] = scaler.fit_transform(maharashtra_data[numerical_columns])

# Save preprocessed data to a new CSV file
maharashtra_data.to_csv("maharashtra_weather_preprocessed.csv", index=False)

# Save scaler parameters for each column
scaler_params = pd.DataFrame({
    'column': numerical_columns,
    'min': [maharashtra_data[col].min() for col in numerical_columns],
    'max': [maharashtra_data[col].max() for col in numerical_columns]
})
scaler_params.set_index('column', inplace=True)
scaler_params.to_csv('maharashtra_weather_scaler_params.csv')

# Display the first few rows of the preprocessed data
print(maharashtra_data.head())
