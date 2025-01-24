# Required package: pip install python-dotenv
import requests
from datetime import datetime, timedelta
import time
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# WeatherAPI.com configuration
API_KEY = '0d61ea11fb77401ab7a135734252401'  # Direct API key assignment instead of env var
if not API_KEY:
    raise ValueError("API key not found")

CITY = "Pune"
COUNTRY = "IN"

def fetch_historical_weather(start_date, end_date):
    base_url = "http://api.weatherapi.com/v1/history.json"
    
    data_points = []
    current_date = start_date
    
    # Validate API key first
    test_response = requests.get(f"{base_url}?key={API_KEY}&q={CITY}&dt={current_date.strftime('%Y-%m-%d')}")
    if test_response.status_code == 401:
        raise ValueError("Invalid API key. Please check your WeatherAPI.com API key")
    
    while current_date <= end_date:
        params = {
            'key': API_KEY,
            'q': f"{CITY}",
            'dt': current_date.strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data_points.append(response.json())
            else:
                error_message = f"Error {response.status_code}"
                try:
                    error_message += f": {response.json().get('error', {}).get('message', '')}"
                except:
                    pass
                print(f"Error fetching data for {current_date}: {error_message}")
            
            # API has rate limits (1 call per second for free tier)
            time.sleep(1.1)
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        current_date += timedelta(days=1)  # WeatherAPI uses daily intervals for historical data
    
    return data_points

def process_weather_data(data_points):
    """Transform weather data into required format"""
    processed_data = []
    
    for day_data in data_points:
        for hour in day_data['forecast']['forecastday'][0]['hour']:
            processed_data.append({
                'date': hour['time'].split()[0],
                'time': hour['time'].split()[1],
                'temp': hour['temp_c'],
                'precipitation': hour['precip_mm'],
                'windspeed': hour['wind_kph']
            })
    
    return processed_data

def save_to_file(data, filename):
    """Save data in CSV format with specific columns"""
    import pandas as pd
    
    df = pd.DataFrame(data)
    # Ensure correct column order
    df = df[['date', 'time', 'temp', 'precipitation', 'windspeed']]
    df.to_csv(filename, index=False)

def main():
    # Set date range from 2010-01-01 to current date
    end_date = datetime.now()
    start_date = datetime(2010, 1, 1)
    
    print(f"Fetching weather data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    weather_data = fetch_historical_weather(start_date, end_date)
    
    # Process the data into required format
    processed_data = process_weather_data(weather_data)
    
    # Save the data with .csv extension
    filename = f"pune_weather_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    save_to_file(processed_data, filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()
    main()