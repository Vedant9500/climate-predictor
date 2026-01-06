"""
Configuration settings for the Climate Predictor project.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

# =============================================================================
# Location Configuration
# =============================================================================
@dataclass
class Location:
    name: str
    latitude: float
    longitude: float

LOCATIONS = {
    "berlin": Location("Berlin", 52.52, 13.41),
    "munich": Location("Munich", 48.14, 11.58),
    "pune": Location("Pune", 18.52, 73.86),
}

DEFAULT_LOCATION = "berlin"

# =============================================================================
# Open-Meteo API Configuration
# =============================================================================
OPEN_METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly weather variables to fetch
HOURLY_VARIABLES = [
    # Temperature
    "temperature_2m",
    "apparent_temperature",
    "dewpoint_2m",
    
    # Humidity & Pressure
    "relative_humidity_2m",
    "surface_pressure",
    "pressure_msl",
    
    # Wind
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    
    # Precipitation
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    
    # Solar Radiation
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    
    # Cloud Cover
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    
    # Weather Code
    "weather_code",
]


# =============================================================================
# Data Configuration
# =============================================================================
DATA_START_YEAR = 2015
DATA_END_YEAR = 2024
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Sequence length for LSTM (hours of history to use)
# Sequence length for LSTM (hours of history to use)
SEQUENCE_LENGTH = 72  # 3 days of hourly data

# Prediction horizons (hours ahead to predict)
PREDICTION_HORIZONS = [1, 3, 6, 12, 24]

# Target variables to predict
TARGET_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
]

# =============================================================================
# Model Configuration
# =============================================================================
@dataclass
class ModelConfig:
    input_size: int = len(HOURLY_VARIABLES)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    use_attention: bool = True
    output_size: int = len(TARGET_VARIABLES) * len(PREDICTION_HORIZONS)

# =============================================================================
# Training Configuration
# =============================================================================
@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10
    grad_clip: float = 1.0
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
# =============================================================================
# Paths
# =============================================================================
SAVED_MODELS_DIR = "saved_models"
LOGS_DIR = "logs"
