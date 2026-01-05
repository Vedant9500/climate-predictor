"""
Weather prediction pipeline.
Fetches recent data, preprocesses, and generates forecasts.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    ModelConfig,
    SAVED_MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_VARIABLES,
    PREDICTION_HORIZONS,
    SEQUENCE_LENGTH,
    DEFAULT_LOCATION,
)
from data.fetcher import WeatherDataFetcher
from data.preprocessor import DataPreprocessor
from models.lstm import WeatherLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherPredictor:
    """End-to-end weather prediction pipeline."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        location: str = DEFAULT_LOCATION,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            location: Location key for fetching data
            device: Device to use for inference
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessor
        self.preprocessor = self._load_preprocessor()
        
        # Data fetcher
        self.fetcher = WeatherDataFetcher(location=location)
        
        logger.info(f"Predictor initialized for {location} on {self.device}")
    
    def _load_model(self, model_path: Optional[str]) -> WeatherLSTM:
        """Load trained model from checkpoint."""
        if model_path is None:
            model_path = Path(SAVED_MODELS_DIR) / "best_model.pt"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint or use default
        config = ModelConfig()
        if 'config' in checkpoint:
            # Update config if stored
            pass
        
        model = WeatherLSTM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded model from {model_path}")
        return model
    
    def _load_preprocessor(self) -> DataPreprocessor:
        """Load preprocessor with fitted scalers."""
        try:
            return DataPreprocessor.load(PROCESSED_DATA_DIR)
        except FileNotFoundError:
            logger.warning("No saved preprocessor found, using default")
            return DataPreprocessor()
    
    def predict(
        self,
        hours_history: int = 96,  # 4 days of history
    ) -> Dict:
        """
        Generate weather predictions.
        
        Args:
            hours_history: Hours of recent data to fetch
            
        Returns:
            Dict with predictions for each horizon and variable
        """
        # Fetch recent data
        days = (hours_history // 24) + 1
        df_raw = self.fetcher.fetch_recent(days=days)
        
        # Preprocess
        df_processed = self.preprocessor.prepare_features(df_raw)
        df_normalized = self.preprocessor.normalize(df_processed, fit=False)
        
        # Get the last sequence
        if len(df_normalized) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Not enough data. Need {SEQUENCE_LENGTH} hours, got {len(df_normalized)}"
            )
        
        sequence = df_normalized.iloc[-SEQUENCE_LENGTH:].values
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(x)
        
        predictions = predictions.cpu().numpy()[0]
        
        # Denormalize
        predictions_denorm = self.preprocessor.denormalize_predictions(
            predictions.reshape(1, -1),
            df_processed,
        )[0]
        
        # Structure output
        result = self._structure_predictions(predictions_denorm, df_raw.index[-1])
        
        return result
    
    def _structure_predictions(
        self,
        predictions: np.ndarray,
        last_timestamp: pd.Timestamp,
    ) -> Dict:
        """Structure flat predictions into readable format."""
        n_targets = len(TARGET_VARIABLES)
        n_horizons = len(PREDICTION_HORIZONS)
        
        result = {
            'generated_at': datetime.now().isoformat(),
            'last_observation': last_timestamp.isoformat(),
            'forecasts': [],
        }
        
        for h_idx, horizon in enumerate(PREDICTION_HORIZONS):
            forecast_time = last_timestamp + pd.Timedelta(hours=horizon)
            forecast = {
                'hours_ahead': horizon,
                'forecast_time': forecast_time.isoformat(),
                'predictions': {},
            }
            
            for t_idx, var_name in enumerate(TARGET_VARIABLES):
                flat_idx = h_idx * n_targets + t_idx
                forecast['predictions'][var_name] = float(predictions[flat_idx])
            
            result['forecasts'].append(forecast)
        
        return result
    
    def predict_range(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Generate predictions for a historical date range.
        Useful for backtesting.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with predictions and actuals
        """
        # Fetch data for the range plus history buffer
        buffer_days = (SEQUENCE_LENGTH // 24) + 2
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
        
        df_raw = self.fetcher.fetch_date_range(
            start_dt.strftime("%Y-%m-%d"),
            end_date,
        )
        
        # Preprocess
        df_processed = self.preprocessor.prepare_features(df_raw)
        df_normalized = self.preprocessor.normalize(df_processed, fit=False)
        
        # Create sequences and predict
        X, _ = self.preprocessor.create_sequences(df_normalized)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        predictions = predictions.cpu().numpy()
        
        # Create DataFrame with results
        # This is simplified - full implementation would align timestamps
        results = pd.DataFrame(predictions)
        results.columns = [
            f"{var}_{h}h" 
            for h in PREDICTION_HORIZONS 
            for var in TARGET_VARIABLES
        ]
        
        return results


def print_forecast(result: Dict):
    """Pretty print forecast results."""
    print("\n" + "="*60)
    print("🌤️ WEATHER FORECAST")
    print("="*60)
    print(f"Generated: {result['generated_at']}")
    print(f"Last observation: {result['last_observation']}")
    print("\n")
    
    for forecast in result['forecasts']:
        print(f"📍 +{forecast['hours_ahead']}h ({forecast['forecast_time'][:16]})")
        for var, value in forecast['predictions'].items():
            unit = _get_unit(var)
            print(f"   {_format_var_name(var)}: {value:.1f} {unit}")
        print()


def _get_unit(var_name: str) -> str:
    """Get unit for variable."""
    units = {
        'temperature_2m': '°C',
        'relative_humidity_2m': '%',
        'precipitation': 'mm',
        'wind_speed_10m': 'km/h',
        'cloud_cover': '%',
    }
    return units.get(var_name, '')


def _format_var_name(var_name: str) -> str:
    """Format variable name for display."""
    names = {
        'temperature_2m': '🌡️ Temperature',
        'relative_humidity_2m': '💧 Humidity',
        'precipitation': '🌧️ Precipitation',
        'wind_speed_10m': '💨 Wind Speed',
        'cloud_cover': '☁️ Cloud Cover',
    }
    return names.get(var_name, var_name)


if __name__ == "__main__":
    # This will fail without a trained model, but shows usage
    try:
        predictor = WeatherPredictor()
        result = predictor.predict()
        print_forecast(result)
    except FileNotFoundError as e:
        print(f"Cannot run prediction: {e}")
        print("Please train a model first with: python train.py")
