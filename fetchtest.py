import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly

# Define location for Pune
pune = Point(18.5204, 73.8567, 560)

# Define time period for the last 20 years
# Define coordinates for 20 major Indian cities (latitude, longitude, elevation)
cities = {
    'Mumbai': Point(19.0760, 72.8777, 14),
    'Delhi': Point(28.6139, 77.2090, 216),
    'Bangalore': Point(12.9716, 77.5946, 920),
    'Hyderabad': Point(17.3850, 78.4867, 542),
    'Chennai': Point(13.0827, 80.2707, 6),
    'Kolkata': Point(22.5726, 88.3639, 9),
    'Pune': Point(18.5204, 73.8567, 560),
    'Ahmedabad': Point(23.0225, 72.5714, 53),
    'Jaipur': Point(26.9124, 75.7873, 431),
    'Lucknow': Point(26.8467, 80.9462, 123),
    'Kanpur': Point(26.4499, 80.3319, 126),
    'Nagpur': Point(21.1458, 79.0882, 310),
    'Indore': Point(22.7196, 75.8577, 553),
    'Thane': Point(19.2183, 72.9781, 10),
    'Bhopal': Point(23.2599, 77.4126, 527),
    'Visakhapatnam': Point(17.6868, 83.2185, 10),
    'Patna': Point(25.5941, 85.1376, 53),
    'Vadodara': Point(22.3072, 73.1812, 39),
    'Ghaziabad': Point(28.6692, 77.4538, 200),
    'Ludhiana': Point(30.9010, 75.8573, 262)
}

# Time period
start = datetime(2003, 1, 1)
end = datetime(2023, 1, 1)

# Create an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Create data for each city
for city, point in cities.items():
    data = Hourly(point, start, end)
    data = data.fetch()
    # Select and rename the required columns
    data = data[['temp', 'prcp', 'wspd']]
    data = data.rename(columns={
        'temp': 'temperature_2m',
        'prcp': 'precipitation',
        'wspd': 'wind_speed_10m'
    })
    # Add a column for the city name
    data['city'] = city
    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data])

# Save the combined data to a single CSV file
combined_data.to_csv('/workspaces/climate-predictor/combined_hourly_data.csv', index=False)
print("Saved combined data for all cities")