"""
Main training script for the weather prediction model.
"""
import argparse
import numpy as np
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    ModelConfig,
    TrainingConfig,
    DATA_START_YEAR,
    DATA_END_YEAR,
    DEFAULT_LOCATION,
    PROCESSED_DATA_DIR,
    TARGET_VARIABLES,
    HOURLY_VARIABLES,
)
from config.specialists import get_specialist, list_specialists
from data.fetcher import WeatherDataFetcher
from data.preprocessor import DataPreprocessor
from models.lstm import WeatherLSTM
from training.trainer import Trainer
from training.evaluate import Evaluator
from utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train weather prediction model')
    
    # Data arguments
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION,
                       help='Location to train on (berlin, munich, pune)')
    parser.add_argument('--start-year', type=int, default=DATA_START_YEAR,
                       help='Start year for training data')
    parser.add_argument('--end-year', type=int, default=DATA_END_YEAR,
                       help='End year for training data')
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip data fetching (use cached data)')
    parser.add_argument('--multi-location', action='store_true',
                       help='Train on all TRAINING_LOCATIONS instead of single location')
    
    # Model arguments
    parser.add_argument('--hidden-size', type=int, default=48,  # Balanced capacity
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=1,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.25,  # Standard regularization
                       help='Dropout rate')
    parser.add_argument('--no-attention', action='store_true',
                       help='Disable attention mechanism (enabled by default)')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Enable bidirectional LSTM (disabled by default)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,  # Increased for complex patterns
                       help='Early stopping patience')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--input-noise', type=float, default=0.01,
                       help='Input noise level for data augmentation')
    parser.add_argument('--stride', type=int, default=3,
                        help='Stride between sequences (lower = more overlap, more data)')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step'],
                        help='LR scheduler: cosine (smooth decay), plateau (reduce on stall), step (fixed steps)')
    
    # Other
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda)')
    parser.add_argument('--specialist', type=str, default=None,
                       choices=['precipitation', 'temperature', 'wind', 'cloud', 'humidity'],
                       help='Train a specialist model (uses domain-specific features)')
    
    args = parser.parse_args()
    
    # Handle specialist mode
    specialist_config = None
    if args.specialist:
        specialist_config = get_specialist(args.specialist)
        logger.info(f"\n{'='*60}")
        logger.info(f"SPECIALIST MODE: {specialist_config.name}")
        logger.info(f"{'='*60}")
        logger.info(f"Description: {specialist_config.description}")
        logger.info(f"Targets: {specialist_config.targets}")
        logger.info(f"Features: {len(specialist_config.features)} domain-specific")
        # Override hyperparameters from specialist config
        args.dropout = specialist_config.dropout
        args.weight_decay = specialist_config.weight_decay
        args.input_noise = specialist_config.input_noise
    
    # =========================================================================
    # Step 1: Fetch Data (Multi-Location)
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 1: Fetching Multi-Location Data")
    logger.info("="*60)
    
    from config.settings import TRAINING_LOCATIONS, LOCATIONS
    import pandas as pd
    # Note: Path and np already imported at top of file
    
    # Use multiple locations if --multi-location flag is set
    if args.multi_location:
        locations_to_use = TRAINING_LOCATIONS
    else:
        locations_to_use = [args.location]
    logger.info(f"Training locations: {locations_to_use}")
    
    all_location_data = {}
    
    for loc_name in locations_to_use:
        logger.info(f"\n--- Fetching {loc_name} ---")
        fetcher = WeatherDataFetcher(location=loc_name)
        
        if args.skip_fetch:
            # Load cached data
            data_dir = Path("data/raw")
            all_data = []
            for year in range(args.start_year, args.end_year + 1):
                year_file = data_dir / f"{loc_name}_{year}.parquet"
                if year_file.exists():
                    all_data.append(pd.read_parquet(year_file))
            if all_data:
                df_raw = pd.concat(all_data, axis=0)
                df_raw.sort_index(inplace=True)
                # Add static features
                df_raw['latitude'] = LOCATIONS[loc_name].latitude
                df_raw['longitude'] = LOCATIONS[loc_name].longitude
                df_raw['elevation'] = 0  # Default, API provides this
            else:
                logger.warning(f"No cached data for {loc_name}, skipping...")
                continue
        else:
            df_raw = fetcher.fetch_with_static_features(args.start_year, args.end_year)
        
        all_location_data[loc_name] = df_raw
        logger.info(f"{loc_name}: {df_raw.shape}")
    
    # =========================================================================
    # Step 2: Preprocess Each Location Separately
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 2: Preprocessing Multi-Location Data")
    logger.info("="*60)
    
    # Set seeds for reproducibility
    import random
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logger.info(f"Set random seeds to {SEED} for reproducibility")
    
    # Create preprocessor with appropriate target variables
    if specialist_config:
        preprocessor = DataPreprocessor(target_variables=specialist_config.targets)
        logger.info(f"Specialist targets: {specialist_config.targets}")
    else:
        preprocessor = DataPreprocessor()
    
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []
    
    # First pass: collect all training data for scaler fitting
    all_train_dfs = []
    
    for loc_name, df_raw in all_location_data.items():
        logger.info(f"\n--- Processing {loc_name} ---")
        
        df_processed = preprocessor.prepare_features(df_raw)
        
        # FIX Issue 2: Use year-based temporal split to prevent temporal leakage
        # Train: 2015-2021 (7 years), Val: 2022 (1 year), Test: 2023-2024 (2 years)
        TRAIN_YEARS = range(2015, 2022)
        VAL_YEARS = [2022]
        TEST_YEARS = [2023, 2024]
        
        df_train = df_processed[df_processed.index.year.isin(TRAIN_YEARS)]
        df_val = df_processed[df_processed.index.year.isin(VAL_YEARS)]
        df_test = df_processed[df_processed.index.year.isin(TEST_YEARS)]
        
        all_train_dfs.append(df_train)
        
        # Store for later
        all_location_data[loc_name] = {
            'train': df_train,
            'val': df_val,
            'test': df_test,
        }
        logger.info(f"{loc_name} split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    # FIX: Fit scaler on ALL locations' training data combined
    # This ensures proper normalization across all weather ranges
    combined_train = pd.concat(all_train_dfs, axis=0)
    logger.info(f"\nFitting scaler on ALL locations ({len(combined_train)} samples)")
    _ = preprocessor.normalize(combined_train, fit=True)
    del combined_train  # Free memory
    
    # Second pass: normalize and create sequences for each location
    for loc_name, splits in all_location_data.items():
        if not isinstance(splits, dict):
            continue
            
        logger.info(f"\n--- Creating sequences for {loc_name} ---")
        
        # Normalize using fitted scaler
        df_train_norm = preprocessor.normalize(splits['train'], fit=False)
        df_val_norm = preprocessor.normalize(splits['val'], fit=False)
        df_test_norm = preprocessor.normalize(splits['test'], fit=False)
        
        # Specialist mode: Filter to domain-specific features only
        if specialist_config:
            # Get features that exist in both specialist config and data
            # Include time features (hour_sin, etc.) which are always useful
            time_features = [c for c in df_train_norm.columns if c.startswith(('hour_', 'day_', 'month_', 'dow_'))]
            static_features = ['latitude', 'longitude', 'elevation']
            specialist_features = specialist_config.features + time_features + static_features
            available_features = [f for f in specialist_features if f in df_train_norm.columns]
            # Also keep target variables for sequence creation
            available_features += [t for t in specialist_config.targets if t in df_train_norm.columns]
            available_features = list(dict.fromkeys(available_features))  # Remove duplicates
            
            df_train_norm = df_train_norm[available_features]
            df_val_norm = df_val_norm[available_features]
            df_test_norm = df_test_norm[available_features]
            logger.info(f"Specialist features: {len(available_features)} (from {len(splits['train'].columns)})")
        
        # FIX Issue 1: Use create_sequences_no_leakage() to prevent data leakage
        # This separates target variables from input features before sequence creation
        X_train, y_train = preprocessor.create_sequences_no_leakage(df_train_norm, stride=args.stride, noise_level=args.input_noise)
        X_val, y_val = preprocessor.create_sequences_no_leakage(df_val_norm, stride=args.stride, noise_level=0.0)
        X_test, y_test = preprocessor.create_sequences_no_leakage(df_test_norm, stride=args.stride, noise_level=0.0)
        
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_val.append(X_val)
        all_y_val.append(y_val)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
        
        logger.info(f"{loc_name}: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Combine all locations
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    y_val = np.concatenate(all_y_val, axis=0)
    X_test = np.concatenate(all_X_test, axis=0)
    y_test = np.concatenate(all_y_test, axis=0)
    
    # Shuffle training data so batches have mixed cities
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # Save preprocessor
    preprocessor.save()
    
    logger.info(f"\n=== Combined Data ===")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # =========================================================================
    # Step 3: Create Model
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 3: Creating Model")
    logger.info("="*60)
    
    # Calculate output size based on targets and horizons
    from config.settings import PREDICTION_HORIZONS
    if specialist_config:
        n_targets = len(specialist_config.targets)
    else:
        n_targets = len(TARGET_VARIABLES)
    output_size = n_targets * len(PREDICTION_HORIZONS)
    
    model_config = ModelConfig(
        input_size=X_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_attention=not args.no_attention,
        output_size=output_size,
    )
    
    model = WeatherLSTM(model_config)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # =========================================================================
    # Step 4: Train Model
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 4: Training Model")
    logger.info("="*60)
    
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )
    
    # Set save directory (specialists go in separate folder)
    if specialist_config:
        save_dir = f"saved_models/specialists/{args.specialist}"
        logger.info(f"Specialist model will be saved to: {save_dir}/")
    else:
        save_dir = None  # Use default
    
    trainer = Trainer(model, config=train_config, device=args.device, lr_scheduler=args.lr_scheduler, save_dir=save_dir)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # Step 5: Evaluate on Test Set
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 5: Evaluating Model")
    logger.info("="*60)
    
    # Load best model
    trainer.load_best_model()
    
    # Get test predictions in batches to avoid OOM
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    model.eval()
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    all_predictions = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(trainer.device)
            preds = model(batch_x).cpu().numpy()
            all_predictions.append(preds)
    
    test_predictions = np.concatenate(all_predictions, axis=0)
    
    # Evaluate (specialist mode uses different target list)
    if specialist_config:
        eval_targets = specialist_config.targets
    else:
        eval_targets = TARGET_VARIABLES
    evaluator = Evaluator(target_variables=eval_targets)
    results = evaluator.print_report(y_test, test_predictions)
    
    # =========================================================================
    # Step 6: Visualize Results
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 6: Generating Visualizations")
    logger.info("="*60)
    
    viz = Visualizer()
    
    # Training history
    viz.plot_training_history(history, save_name="training_history")
    
    # Error by horizon
    viz.plot_error_by_horizon(results['by_horizon'], save_name="error_by_horizon")
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best model saved to: saved_models/best_model.pt")
    logger.info(f"Run predictions with: python predict.py")


if __name__ == "__main__":
    main()
