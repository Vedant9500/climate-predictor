import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from config.settings import DEFAULT_LOCATION, LOCATIONS, TARGET_VARIABLES, PREDICTION_HORIZONS, ModelConfig
from data.fetcher import WeatherDataFetcher
from data.preprocessor import DataPreprocessor
from models.lstm import WeatherLSTM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_rain_performance(model, x, y_true, preprocessor, df_processed):
    """
    Evaluate model specifically on precipitation with full classification metrics.
    x: Normalized input features
    y_true: Denormalized target values
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(x).numpy()
    
    # Denormalize predictions
    y_pred_denorm = preprocessor.denormalize_predictions(y_pred, df_processed)
    
    # Get precipitation index
    precip_idx = TARGET_VARIABLES.index("precipitation")
    
    metrics = {}
    
    for h_idx, horizon in enumerate(PREDICTION_HORIZONS):
        # Flatten index for this horizon's precipitation
        flat_idx = h_idx * len(TARGET_VARIABLES) + precip_idx
        
        pred_rain = y_pred_denorm[:, flat_idx]
        true_rain = y_true[:, flat_idx]
        
        # --- 1. Masks ---
        # Actual Rain (Ground Truth > 0.1)
        rain_mask = true_rain > 0.1
        # Predicted Rain (Model Output > 0.1)
        pred_wet_mask = pred_rain > 0.1
        
        # --- 2. Regression Metrics (How close was the amount?) ---
        if np.sum(rain_mask) > 0:
            mae_wet = np.mean(np.abs(pred_rain[rain_mask] - true_rain[rain_mask]))
        else:
            mae_wet = 0.0 # Or float('nan')
            
        # --- 3. Classification Metrics (Did it spot the rain?) ---
        # True Positives: Rained, and we predicted rain
        tp = np.sum(pred_wet_mask & rain_mask)
        
        # False Negatives: Rained, but we predicted dry (MISSED RAIN)
        fn = np.sum(~pred_wet_mask & rain_mask)
        
        # False Positives: Didn't rain, but we predicted rain (FALSE ALARM)
        fp = np.sum(pred_wet_mask & ~rain_mask)
        
        # True Negatives: Didn't rain, we predicted dry
        tn = np.sum(~pred_wet_mask & ~rain_mask)
        
        # --- Calculations ---
        # Recall (Sensitivity): How much of the actual rain did we catch?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Precision: When we predicted rain, how often was it real?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # F1-Score: The harmonic balance between Precision and Recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Store metrics
        metrics[f"h{horizon}_mae_wet"] = mae_wet
        metrics[f"h{horizon}_recall"] = recall
        metrics[f"h{horizon}_precision"] = precision
        metrics[f"h{horizon}_f1"] = f1
        metrics[f"h{horizon}_tp"] = tp
        metrics[f"h{horizon}_fp"] = fp
        metrics[f"h{horizon}_count"] = np.sum(rain_mask)
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default='berlin')
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()
    
    # 1. Fetch Data
    fetcher = WeatherDataFetcher(location=args.location)
    df = fetcher.fetch_recent(days=args.days)
    
    # FIX: Elevation attribute
    if args.location in LOCATIONS:
        df['latitude'] = LOCATIONS[args.location].latitude
        df['longitude'] = LOCATIONS[args.location].longitude
        df['elevation'] = df.attrs.get('elevation', 0)

    # 2. Process
    preprocessor = DataPreprocessor.load()
    df_processed = preprocessor.prepare_features(df)
    df_norm = preprocessor.normalize(df_processed, fit=False)
    
    # Create sequences WITH targets (33 features - for old models)
    X_full, y = preprocessor.create_sequences(df_norm, stride=1, noise_level=0.0)
    
    # Create sequences WITHOUT targets (28 features - for no-leakage models)
    feature_cols = [col for col in df_norm.columns if col not in TARGET_VARIABLES]
    df_features_only = df_norm[feature_cols]
    X_no_leak, _ = preprocessor.create_sequences(df_features_only, stride=1, noise_level=0.0)
    
    # Convert 'y' (targets) back to denormalized scale for comparison
    y_denorm = preprocessor.denormalize_predictions(y, df_processed)
    
    X_full_tensor = torch.FloatTensor(X_full)
    X_no_leak_tensor = torch.FloatTensor(X_no_leak)

    # 3. Load Models
    models = {
        "Current": Path("saved_models/best_model.pt"),
        "Rain-Specialist-1": Path("hall of fame/1(rain).pt"),
        "Rain-Specialist-2": Path("hall of fame/2(rain).pt"),
        #"Rain-Specialist-3": Path("hall of fame/3(rain).pt"),
    }
    
    print(f"\n🌧️ RAIN CHECK ANALYSIS ({args.location.upper()}, {args.days} days)")
    print(f"Total Hours: {len(y)}")
    
    precip_idx = TARGET_VARIABLES.index("precipitation")
    # Take H=1 (first horizon) for summary stats
    h1_flat_idx = 0 * len(TARGET_VARIABLES) + precip_idx
    actual_rain_h1 = y_denorm[:, h1_flat_idx]
    rainy_hours = np.sum(actual_rain_h1 > 0.1)
    print(f"Rainy Hours (actual > 0.1mm): {rainy_hours} ({rainy_hours/len(y)*100:.1f}%)")
    print("-" * 60)
    
    print(f"{'Model':<20} | {'MAE(Wet)':<9} | {'Recall':<8} | {'Prec':<8} | {'F1':<6} | {'MAE(All)':<9}")
    print("-" * 80)
    
    for name, path in models.items():
        if not path.exists():
            print(f"{name:<20} | NOT FOUND")
            continue
            
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            config = checkpoint.get('model_config', ModelConfig())
            
            model = WeatherLSTM(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Select appropriate input based on model's expected features
            if config.input_size == X_no_leak_tensor.shape[-1]:
                X_tensor = X_no_leak_tensor  # 28 features (no-leakage model)
            else:
                X_tensor = X_full_tensor  # 33 features (old model)
            
            # 1. Run the detailed evaluation
            metrics = evaluate_rain_performance(model, X_tensor, y_denorm, preprocessor, df_processed)
            
            # 2. Extract specific metrics for Horizon 1 (h1)
            mae_wet = metrics.get('h1_mae_wet', 0)
            recall = metrics.get('h1_recall', 0) * 100
            precision = metrics.get('h1_precision', 0) * 100
            f1 = metrics.get('h1_f1', 0)
            
            # 3. Calculate Global MAE (Error on both rain and dry days)
            model.eval()
            with torch.no_grad():
                full_pred = model(X_tensor).numpy()
            full_pred_denorm = preprocessor.denormalize_predictions(full_pred, df_processed)
            pred_rain_h1 = full_pred_denorm[:, h1_flat_idx]
            mae_all = np.mean(np.abs(pred_rain_h1 - actual_rain_h1))
            
            # 4. Print with the new columns
            print(f"{name:<20} | {mae_wet:9.4f} | {recall:7.1f}% | {precision:7.1f}% | {f1:6.3f} | {mae_all:9.4f}")
            
        except Exception as e:
             # Helpful for debugging if something breaks
             import traceback
             traceback.print_exc()
             print(f"{name:<20} | ERROR: {str(e)[:20]}...")

if __name__ == "__main__":
    main()
