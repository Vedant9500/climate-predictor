"""
Open-Meteo Historical Weather API client.
Fetches hourly weather data efficiently with rate limiting.
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    OPEN_METEO_BASE_URL,
    HOURLY_VARIABLES,
    LOCATIONS,
    DEFAULT_LOCATION,
    DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDataFetcher:
    """Fetches historical weather data from Open-Meteo API."""
    
    def __init__(
        self,
        location: str = DEFAULT_LOCATION,
        rate_limit_delay: float = 0.1,  # Delay between requests in seconds
    ):
        """
        Initialize the fetcher.
        
        Args:
            location: Location key from LOCATIONS dict
            rate_limit_delay: Seconds to wait between API calls
        """
        if location not in LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(LOCATIONS.keys())}")
        
        self.location = LOCATIONS[location]
        self.rate_limit_delay = rate_limit_delay
        self.base_url = OPEN_METEO_BASE_URL
        
    def fetch_year(self, year: int) -> pd.DataFrame:
        """
        Fetch one year of hourly data.
        
        Args:
            year: Year to fetch (e.g., 2023)
            
        Returns:
            DataFrame with hourly weather data
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        return self.fetch_date_range(start_date, end_date)
    
    def fetch_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch hourly data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with hourly weather data
        """
        params = {
            "latitude": self.location.latitude,
            "longitude": self.location.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
        }
        
        logger.info(f"Fetching {self.location.name} data: {start_date} to {end_date}")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Parse response into DataFrame
            hourly_data = data.get("hourly", {})
            if not hourly_data:
                logger.warning("No hourly data in response")
                return pd.DataFrame()
            
            df = pd.DataFrame(hourly_data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            
            # Add metadata
            df.attrs["latitude"] = data.get("latitude")
            df.attrs["longitude"] = data.get("longitude")
            df.attrs["elevation"] = data.get("elevation")
            
            logger.info(f"Fetched {len(df)} hourly records")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def fetch_years(
        self,
        start_year: int,
        end_year: int,
        save_intermediate: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch multiple years of data.
        
        Args:
            start_year: First year to fetch
            end_year: Last year to fetch (inclusive)
            save_intermediate: Save each year to disk as backup
            
        Returns:
            Combined DataFrame with all years
        """
        all_data = []
        save_dir = Path(DATA_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for year in range(start_year, end_year + 1):
            year_file = save_dir / f"{self.location.name.lower()}_{year}.parquet"
            
            # Check if already downloaded
            if year_file.exists():
                logger.info(f"Loading cached data for {year}")
                df = pd.read_parquet(year_file)
            else:
                df = self.fetch_year(year)
                
                if save_intermediate and not df.empty:
                    df.to_parquet(year_file)
                    logger.info(f"Saved {year} data to {year_file}")
            
            all_data.append(df)
        
        # Combine all years
        combined = pd.concat(all_data, axis=0)
        combined.sort_index(inplace=True)
        
        logger.info(f"Total records: {len(combined)} ({start_year}-{end_year})")
        
        return combined
    
    def fetch_recent(self, days: int = 7) -> pd.DataFrame:
        """
        Fetch recent data for prediction.
        Uses the forecast API for data within last 5 days.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame with recent hourly data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_date_range(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )


if __name__ == "__main__":
    # Example usage
    fetcher = WeatherDataFetcher(location="berlin")
    
    # Fetch recent week for testing
    df = fetcher.fetch_recent(days=7)
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nShape: {df.shape}")
