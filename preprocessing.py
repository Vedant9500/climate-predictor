import pandas as pd
from datetime import datetime
from meteostat import Stations, Hourly
from concurrent.futures import ThreadPoolExecutor
import time

# Define cities and their coordinates
CITIES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "New Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Surat": {"lat": 21.1702, "lon": 72.8311},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366},
    "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Ranchi": {"lat": 23.3441, "lon": 85.3096},
    "Agra": {"lat": 27.1767, "lon": 78.0081},
    "Indore": {"lat": 22.7196, "lon": 75.8577},
    "Raipur": {"lat": 21.2514, "lon": 81.6296},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362},
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573},
    "Nashik": {"lat": 19.9975, "lon": 73.7898},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
    "Madurai": {"lat": 9.9252, "lon": 78.1198},
    "Hubballi-Dharwad": {"lat": 15.3647, "lon": 75.1239},
    "Mysore": {"lat": 12.2958, "lon": 76.6394},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047},
    "Salem": {"lat": 11.6643, "lon": 78.1460},
    "Thane": {"lat": 19.2183, "lon": 72.9781},
    "Jabalpur": {"lat": 23.1815, "lon": 79.9864},
    "Gwalior": {"lat": 26.2183, "lon": 78.1828},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245},
    "Vijayawada": {"lat": 16.5062, "lon": 80.6480},
    "Amravati": {"lat": 20.9374, "lon": 77.7796},
}

# Define time period
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2024, 1, 1)

def get_nearest_station(lat, lon):
    """Find nearest weather station."""
    stations = Stations()
    stations = stations.nearby(lat, lon)
    station = stations.fetch(1)
    return station.index[0] if not station.empty else None

def fetch_data(city, lat, lon):
    """Fetch weather data for a city."""
    print(f"Fetching data for {city}...")
    
    # Get nearest station
    station_id = get_nearest_station(lat, lon)
    if not station_id:
        print(f"No weather station found near {city}")
        return None
    
    try:
        # Get hourly data
        data = Hourly(station_id, START_DATE, END_DATE)
        df = data.fetch()

        if df.empty:
            print(f"No data found for {city}")
            return None

        # Clean and process data
        df = df.reset_index()
        
        # Rename columns to match the required format
        df = df.rename(columns={
            'time': 'time',
            'temp': 'temperature_2m',  # Already in Celsius
            'prcp': 'precipitation',    # Already in mm
            'wspd': 'wind_speed_10m'    # Convert from km/h to m/s
        })
        
        # Convert wind speed from km/h to m/s
        df['wind_speed_10m'] = df['wind_speed_10m'] * 0.277778
        
        # Keep only needed columns
        needed_columns = ['time', 'temperature_2m', 'precipitation', 'wind_speed_10m']
        df = df[needed_columns]
        
        # Remove missing values
        df = df.dropna()
        
        # Add city column
        df['city'] = city
        
        return df
    
    except Exception as e:
        print(f"Error processing data for {city}: {e}")
        return None

def fetch_all_cities():
    """Fetch weather data for all cities."""
    all_data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for city, coords in CITIES.items():
            futures.append(executor.submit(fetch_data, city, coords["lat"], coords["lon"]))
        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    all_data.append(result)
            except Exception as e:
                print(f"Error in thread execution: {e}")

    # Combine all data into a single DataFrame
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv("all_cities_weather_data.csv", index=False)
        print("All cities data saved to all_cities_weather_data.csv")
    else:
        print("No data fetched for any city.")

if __name__ == "__main__":
    fetch_all_cities()
