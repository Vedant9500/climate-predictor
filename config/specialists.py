"""
Specialist model configurations.
Each specialist focuses on specific target variables using domain-relevant features.
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SpecialistConfig:
    """Configuration for a specialist model."""
    name: str
    description: str
    targets: List[str]  # Target variables to predict
    features: List[str]  # Input features to use (from HOURLY_VARIABLES)
    # Optional hyperparameter overrides
    dropout: float = 0.25
    weight_decay: float = 0.01
    input_noise: float = 0.01


# =============================================================================
# Specialist Definitions
# =============================================================================

SPECIALISTS: Dict[str, SpecialistConfig] = {
    
    "precipitation": SpecialistConfig(
        name="Precipitation Specialist",
        description="Focuses on rain/snow prediction using moisture and cloud features",
        targets=["precipitation"],
        features=[
            # Moisture indicators (critical for precipitation)
            "relative_humidity_2m",
            "dewpoint_2m",
            # Pressure (low pressure = storms)
            "surface_pressure",
            "pressure_msl",
            # Cloud cover (clouds = potential rain)
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid", 
            "cloud_cover_high",
            # Weather code (encodes current conditions)
            "weather_code",
            # Wind can bring moisture
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        # Rain specialists often benefit from lower regularization
        dropout=0.2,
        weight_decay=0.005,
        input_noise=0.02,
    ),
    
    "temperature": SpecialistConfig(
        name="Temperature Specialist", 
        description="Focuses on temperature prediction using radiation and thermal features",
        targets=["temperature_2m"],
        features=[
            # Solar radiation (main driver of temperature)
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            # Current thermal state
            "apparent_temperature",
            "dewpoint_2m",
            # Cloud cover (modulates radiation)
            "cloud_cover",
            # Humidity affects apparent temperature
            "relative_humidity_2m",
        ],
        # Temperature is more predictable, can use standard regularization
        dropout=0.25,
        weight_decay=0.01,
        input_noise=0.01,
    ),
    
    "wind": SpecialistConfig(
        name="Wind Specialist",
        description="Focuses on wind prediction using pressure and terrain features",
        targets=["wind_speed_10m"],
        features=[
            # Pressure gradients drive wind
            "surface_pressure",
            "pressure_msl",
            # Current wind state
            "wind_direction_10m",
            "wind_gusts_10m",
            # Temperature differences create pressure differences
            "temperature_2m",
            "apparent_temperature",
        ],
        dropout=0.3,
        weight_decay=0.02,
        input_noise=0.01,
    ),
    
    "cloud": SpecialistConfig(
        name="Cloud Cover Specialist",
        description="Focuses on cloud cover prediction using moisture and radiation features",
        targets=["cloud_cover"],
        features=[
            # Moisture leads to clouds
            "relative_humidity_2m",
            "dewpoint_2m",
            # Existing cloud layers
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            # Radiation (inverse relationship with clouds)
            "shortwave_radiation",
            "diffuse_radiation",
            # Pressure patterns
            "surface_pressure",
            "pressure_msl",
        ],
        dropout=0.25,
        weight_decay=0.01,
        input_noise=0.01,
    ),
    
    "humidity": SpecialistConfig(
        name="Humidity Specialist",
        description="Focuses on humidity prediction using temperature and moisture features",
        targets=["relative_humidity_2m"],
        features=[
            # Temperature affects saturation point
            "temperature_2m",
            "apparent_temperature",
            "dewpoint_2m",
            # Precipitation adds moisture
            "precipitation",
            "rain",
            # Pressure systems
            "surface_pressure",
            "pressure_msl",
            # Wind brings/removes moisture
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        dropout=0.25,
        weight_decay=0.01,
        input_noise=0.01,
    ),
}


def get_specialist(name: str) -> SpecialistConfig:
    """Get specialist config by name."""
    if name not in SPECIALISTS:
        available = ", ".join(SPECIALISTS.keys())
        raise ValueError(f"Unknown specialist '{name}'. Available: {available}")
    return SPECIALISTS[name]


def list_specialists() -> List[str]:
    """List all available specialist names."""
    return list(SPECIALISTS.keys())
