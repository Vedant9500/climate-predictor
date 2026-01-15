"""
Backtest script - validate model predictions against actual weather data.
Fetches historical data, makes predictions, and compares to actuals.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    DEFAULT_LOCATION,
    TARGET_VARIABLES,
    PREDICTION_HORIZONS,
    SEQUENCE_LENGTH,
)
from data.fetcher import WeatherDataFetcher
from data.preprocessor import DataPreprocessor
from inference.predictor import WeatherPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backtest(
    location: str = DEFAULT_LOCATION,
    days_ago: int = 3,
    model_path: str = None,
):
    """
    Run backtest: predict past weather and compare to actuals.
    
    Args:
        location: Location to test
        days_ago: How many days ago to start the test
        model_path: Path to model checkpoint
    """
    logger.info(f"Running backtest for {location}, {days_ago} days ago")
    
    # Calculate dates
    # We need: prediction_date (when we make prediction) and target_dates (what we predict)
    today = datetime.now()
    prediction_date = today - timedelta(days=days_ago)
    
    logger.info(f"Prediction made on: {prediction_date.strftime('%Y-%m-%d')}")
    
    # Fetch data for prediction (need SEQUENCE_LENGTH hours before prediction_date)
    fetcher = WeatherDataFetcher(location=location)
    
    # Fetch enough history for prediction + actuals for verification
    start_date = prediction_date - timedelta(days=5)  # Extra buffer
    end_date = prediction_date + timedelta(days=2)  # Include target hours
    
    df = fetcher.fetch_date_range(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )
    
    # Add static features (required for multi-location trained models)
    from config.settings import LOCATIONS
    if location in LOCATIONS:
        df['latitude'] = LOCATIONS[location].latitude
        df['longitude'] = LOCATIONS[location].longitude
        df['elevation'] = df.attrs.get('elevation', 0)
    
    logger.info(f"Fetched {len(df)} hours of data")
    
    # Load preprocessor and model
    preprocessor = DataPreprocessor.load()
    
    # Prepare features
    df_processed = preprocessor.prepare_features(df)
    df_normalized = preprocessor.normalize(df_processed, fit=False)
    
    # Find the prediction point (closest hour to prediction_date)
    prediction_idx = df_normalized.index.get_indexer([prediction_date], method='nearest')[0]
    
    if prediction_idx < SEQUENCE_LENGTH:
        raise ValueError("Not enough history before prediction point")
    
    # Remove target variables from features (match no-leakage training)
    feature_cols = [col for col in df_normalized.columns if col not in TARGET_VARIABLES]
    df_features = df_normalized[feature_cols]
    
    # Get the sequence for prediction (without target variables)
    sequence = df_features.iloc[prediction_idx - SEQUENCE_LENGTH:prediction_idx].values
    
    # Load model and predict
    import torch
    from models.lstm import WeatherLSTM
    from config.settings import SAVED_MODELS_DIR, ModelConfig
    
    if model_path is None:
        model_path = Path(SAVED_MODELS_DIR) / "best_model.pt"
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        config = ModelConfig()
    
    model = WeatherLSTM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    x = torch.FloatTensor(sequence).unsqueeze(0)
    
    # Check input shape compatibility
    if x.shape[-1] != config.input_size:
        raise ValueError(
            f"Input feature mismatch! Model expects {config.input_size} features, "
            f"but prepared data has {x.shape[-1]}.\n"
            "This usually happens when the model was trained with a different strategy "
            "(e.g. 'no_leakage' mode) than the current code.\n"
            "Please RETRAIN the model to fix this: python train.py --multi-location"
        )
    
    with torch.no_grad():
        predictions = model(x).numpy()[0]
    
    # Denormalize predictions
    predictions_denorm = preprocessor.denormalize_predictions(
        predictions.reshape(1, -1),
        df_processed,
    )[0]
    
    # Get actual values at each horizon
    prediction_time = df_normalized.index[prediction_idx]
    
    print("\n" + "="*80)
    print("🔬 BACKTEST RESULTS")
    print("="*80)
    print(f"Prediction made at: {prediction_time}")
    print(f"Location: {location.upper()}")
    print()
    
    results = []
    
    for h_idx, horizon in enumerate(PREDICTION_HORIZONS):
        target_time = prediction_time + pd.Timedelta(hours=horizon)
        
        print(f"\n📍 +{horizon}h Forecast (Target: {target_time.strftime('%Y-%m-%d %H:%M')})")
        print("-" * 60)
        
        # Get actual values if available
        if target_time in df.index:
            print(f"  {'Variable':<25} {'Predicted':>12} {'Actual':>12} {'Error':>12}")
            print("  " + "-" * 51)
            
            for t_idx, var in enumerate(TARGET_VARIABLES):
                flat_idx = h_idx * len(TARGET_VARIABLES) + t_idx
                predicted = predictions_denorm[flat_idx]
                
                if var in df.columns:
                    actual = df.loc[target_time, var]
                    error = predicted - actual
                    pct_error = abs(error / (actual + 1e-8)) * 100
                    
                    results.append({
                        'horizon': horizon,
                        'variable': var,
                        'predicted': predicted,
                        'actual': actual,
                        'error': error,
                        'pct_error': pct_error,
                    })
                    
                    # Color coding for error
                    if abs(error) < 1:
                        status = "✅"
                    elif abs(error) < 3:
                        status = "⚠️"
                    else:
                        status = "❌"
                    
                    print(f"  {var:<25} {predicted:>12.2f} {actual:>12.2f} {error:>+12.2f} {status}")
                else:
                    print(f"  {var:<25} {predicted:>12.2f} {'N/A':>12} {'N/A':>12}")
        else:
            print(f"  Target time not in data (future or gap)")
    
    # Summary statistics
    if results:
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        df_results = pd.DataFrame(results)
        
        for var in TARGET_VARIABLES:
            var_data = df_results[df_results['variable'] == var]
            if len(var_data) > 0:
                mae = var_data['error'].abs().mean()
                print(f"  {var:<25} MAE: {mae:.2f}")
        
        print()
        overall_mae = df_results['error'].abs().mean()
        print(f"  Overall MAE: {overall_mae:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Backtest weather predictions')
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION,
                       help='Location to test')
    parser.add_argument('--days-ago', type=int, default=3,
                       help='How many days ago to make the prediction')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    try:
        backtest(
            location=args.location,
            days_ago=args.days_ago,
            model_path=args.model,
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
