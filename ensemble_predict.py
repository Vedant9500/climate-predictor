"""
Ensemble Predictor - Combines specialist and general models for weather prediction.
Supports testing individual specialists or combining them.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    DEFAULT_LOCATION, LOCATIONS, TARGET_VARIABLES, 
    PREDICTION_HORIZONS, SEQUENCE_LENGTH, ModelConfig
)
from config.specialists import SPECIALISTS
from data.fetcher import WeatherDataFetcher
from data.preprocessor import DataPreprocessor
from models.lstm import WeatherLSTM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Combines multiple specialist models for optimal predictions."""
    
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.models = {}
        self.model_configs = {}  # Store config for each model
        self.preprocessor = None
        
    def load_models(self):
        """Load all available models."""
        logger.info("Loading models...")
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor.load()
        
        # Load general model
        general_path = Path("saved_models/best_model.pt")
        if general_path.exists():
            self._load_model('general', general_path)
            logger.info("  ✓ General model loaded")
        
        # Load temperature specialist
        temp_path = Path("saved_models/specialists/temperature/best_model.pt")
        if temp_path.exists():
            self._load_model('temperature_specialist', temp_path, target_var='temperature_2m')
            logger.info("  ✓ Temperature specialist loaded")
        
        # Load rain specialist (from hall of fame)
        rain_paths = [
            Path("hall of fame/1(rain).pt"),
            Path("hall of fame/2(rain).pt"),
        ]
        for path in rain_paths:
            if path.exists():
                self._load_model('rain_specialist', path, target_var='precipitation')
                logger.info(f"  ✓ Rain specialist loaded ({path.name})")
                break
        
        # Load other specialists if available
        for specialist_name in ['humidity', 'wind', 'cloud']:
            path = Path(f"saved_models/specialists/{specialist_name}/best_model.pt")
            if path.exists():
                target_var = {
                    'humidity': 'relative_humidity_2m',
                    'wind': 'wind_speed_10m',
                    'cloud': 'cloud_cover'
                }.get(specialist_name)
                self._load_model(f'{specialist_name}_specialist', path, target_var=target_var)
                logger.info(f"  ✓ {specialist_name.capitalize()} specialist loaded")
        
        logger.info(f"Loaded {len(self.models)} models total")
        
    def _load_model(self, name: str, path: Path, target_var: str = None):
        """Load a single model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint.get('model_config', ModelConfig())
        model = WeatherLSTM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.models[name] = model
        self.model_configs[name] = {
            'config': config,
            'target_var': target_var,
            'input_size': config.input_size,
            'output_size': config.output_size,
        }
    
    def prepare_data(self, df: pd.DataFrame, model_name: str) -> torch.Tensor:
        """Prepare input data for a specific model."""
        df_processed = self.preprocessor.prepare_features(df)
        df_norm = self.preprocessor.normalize(df_processed, fit=False)
        
        model_info = self.model_configs.get(model_name, {})
        expected_features = model_info.get('input_size', 28)
        
        # Remove target variables (no-leakage models expect this)
        feature_cols = [c for c in df_norm.columns if c not in TARGET_VARIABLES]
        
        # Match expected input size
        if len(feature_cols) >= expected_features:
            df_features = df_norm[feature_cols[:expected_features]]
        else:
            df_features = df_norm[feature_cols]
        
        # Get last SEQUENCE_LENGTH hours
        if len(df_features) < SEQUENCE_LENGTH:
            raise ValueError(f"Need at least {SEQUENCE_LENGTH} hours of data")
        
        sequence = df_features.iloc[-SEQUENCE_LENGTH:].values
        return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
    
    def predict_single(self, df: pd.DataFrame, model_name: str) -> tuple:
        """
        Get prediction from a single model.
        Returns (raw_pred, denormalized_pred, target_variable)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        model_info = self.model_configs[model_name]
        
        x = self.prepare_data(df, model_name)
        
        with torch.no_grad():
            raw_pred = model(x).cpu().numpy()[0]
        
        # Denormalize based on model type
        target_var = model_info.get('target_var')
        
        if target_var and target_var in self.preprocessor.scaler_params:
            # Specialist model - denormalize using target variable's stats
            mean = self.preprocessor.scaler_params[target_var]['mean']
            std = self.preprocessor.scaler_params[target_var]['std']
            denorm_pred = raw_pred * std + mean
        else:
            # General model - use full denormalization
            denorm_pred = self.preprocessor.denormalize_predictions(
                raw_pred.reshape(1, -1)
            )[0]
        
        return raw_pred, denorm_pred, target_var


def main():
    parser = argparse.ArgumentParser(description='Ensemble Weather Prediction Backtest')
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION)
    parser.add_argument('--days', type=int, default=5, help='Days of history')
    parser.add_argument('--days-ago', type=int, default=2, help='Days ago to backtest')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to test (e.g., --models temperature general)')
    args = parser.parse_args()
    
    # Initialize
    ensemble = EnsemblePredictor()
    ensemble.load_models()
    
    # Filter models if requested
    if args.models:
        filtered = {}
        filtered_configs = {}
        for name in args.models:
            for model_name in ensemble.models:
                if name in model_name:
                    filtered[model_name] = ensemble.models[model_name]
                    filtered_configs[model_name] = ensemble.model_configs[model_name]
        if filtered:
            ensemble.models = filtered
            ensemble.model_configs = filtered_configs
            logger.info(f"Testing models: {list(filtered.keys())}")
    
    # Fetch data
    logger.info(f"\nFetching data for {args.location}...")
    fetcher = WeatherDataFetcher(location=args.location)
    df = fetcher.fetch_recent(days=args.days + args.days_ago + 2)
    
    # Add static features
    if args.location in LOCATIONS:
        df['latitude'] = LOCATIONS[args.location].latitude
        df['longitude'] = LOCATIONS[args.location].longitude
        df['elevation'] = 0
    
    # Calculate prediction point
    prediction_time = datetime.now() - timedelta(days=args.days_ago)
    prediction_idx = df.index.get_indexer([prediction_time], method='nearest')[0]
    
    if prediction_idx < SEQUENCE_LENGTH:
        logger.error("Not enough history")
        return
    
    df_input = df.iloc[:prediction_idx + 1].copy()
    actual_time = df.index[prediction_idx]
    
    print("\n" + "="*80)
    print("🌤️ MODEL BACKTEST")
    print("="*80)
    print(f"Location: {args.location.upper()}")
    print(f"Prediction time: {actual_time}")
    
    # Test each model
    for model_name in ensemble.models:
        model_info = ensemble.model_configs[model_name]
        target_var = model_info.get('target_var')
        
        print(f"\n📊 {model_name.upper()}")
        print("-"*70)
        
        try:
            raw, denorm, target = ensemble.predict_single(df_input, model_name)
            
            # Determine what variables this model predicts
            if target_var:
                # Specialist - predicts one variable across horizons
                vars_to_show = [target_var]
                n_targets = 1
            else:
                # General - predicts all variables
                vars_to_show = TARGET_VARIABLES
                n_targets = len(TARGET_VARIABLES)
            
            print(f"  Output shape: {raw.shape}")
            print(f"  {'Horizon':<10} {'Predicted':>12} {'Actual':>12} {'Error':>12}")
            print("  " + "-"*50)
            
            for h_idx, horizon in enumerate(PREDICTION_HORIZONS):
                target_time = actual_time + pd.Timedelta(hours=horizon)
                
                for t_idx, var in enumerate(vars_to_show):
                    if target_var:
                        # Specialist: output is (n_horizons,)
                        idx = h_idx
                    else:
                        # General: output is (n_horizons * n_targets,)
                        idx = h_idx * n_targets + t_idx
                    
                    if idx >= len(denorm):
                        continue
                    
                    predicted = denorm[idx]
                    
                    if target_time in df.index and var in df.columns:
                        actual = df.loc[target_time, var]
                        error = predicted - actual
                        
                        status = "✅" if abs(error) < 1 else ("⚠️" if abs(error) < 3 else "❌")
                        
                        label = f"+{horizon}h {var}" if len(vars_to_show) > 1 else f"+{horizon}h"
                        print(f"  {label:<20} {predicted:>12.2f} {actual:>12.2f} {error:>+10.2f} {status}")
                    else:
                        print(f"  +{horizon}h {var:<15} {predicted:>12.2f} {'N/A':>12} {'N/A':>12}")
                        
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
