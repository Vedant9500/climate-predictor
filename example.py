from meteostat import Point, Daily
import pandas as pd
from datetime import datetime

# Define location (latitude, longitude, and elevation)
location = Point(37.7749, -122.4194)  # Example: San Francisco

# Define date range
start = datetime(2020, 1, 1)
end = datetime(2023, 1, 1)

# Fetch daily weather data
data = Daily(location, start, end)
data = data.fetch()

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())
