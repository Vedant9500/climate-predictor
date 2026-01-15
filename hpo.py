"""
Hyperparameter Optimization (HPO) for weather prediction model.
Uses Optuna to find optimal hyperparameters.

Usage:
    python hpo.py --n-trials 20 --epochs-per-trial 15
"""
import argparse
import optuna
from optuna.trial import Trial
import numpy as np
from pathlib import Path
import logging
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    ModelConfig,
    TrainingConfig,
    DATA_START_YEAR,
    DATA_END_YEAR,
    PROCESSED_DATA_DIR,
)
from data.fetcher import WeatherDataFetcher
from data.preprocessor import DataPreprocessor
from models.lstm import WeatherLSTM
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_data(args) -> tuple:
    """Prepare data once for all trials (saves time)."""
    from config.settings import TRAINING_LOCATIONS, LOCATIONS
    import pandas as pd
    
    locations_to_use = TRAINING_LOCATIONS if args.multi_location else [args.location]
    logger.info(f"Loading data for: {locations_to_use}")
    
    all_location_data = {}
    
    for loc_name in locations_to_use:
        fetcher = WeatherDataFetcher(location=loc_name)
        data_dir = Path("data/raw")
        all_data = []
        
        for year in range(DATA_START_YEAR, DATA_END_YEAR + 1):
            year_file = data_dir / f"{loc_name}_{year}.parquet"
            if year_file.exists():
                all_data.append(pd.read_parquet(year_file))
        
        if all_data:
            df_raw = pd.concat(all_data, axis=0)
            df_raw.sort_index(inplace=True)
            df_raw['latitude'] = LOCATIONS[loc_name].latitude
            df_raw['longitude'] = LOCATIONS[loc_name].longitude
            df_raw['elevation'] = 0
            all_location_data[loc_name] = df_raw
    
    return all_location_data


def create_trial_data(all_location_data: dict, preprocessor: DataPreprocessor, 
                      stride: int, noise_level: float) -> tuple:
    """Create train/val/test data for a trial."""
    TRAIN_YEARS = range(2015, 2022)
    VAL_YEARS = [2022]
    TEST_YEARS = [2023, 2024]
    
    all_train_dfs = []
    processed_data = {}
    
    # First pass: prepare features and split by year
    for loc_name, df_raw in all_location_data.items():
        df_processed = preprocessor.prepare_features(df_raw)
        
        df_train = df_processed[df_processed.index.year.isin(TRAIN_YEARS)]
        df_val = df_processed[df_processed.index.year.isin(VAL_YEARS)]
        df_test = df_processed[df_processed.index.year.isin(TEST_YEARS)]
        
        all_train_dfs.append(df_train)
        processed_data[loc_name] = {'train': df_train, 'val': df_val, 'test': df_test}
    
    # Fit scaler on combined training data
    import pandas as pd
    combined_train = pd.concat(all_train_dfs, axis=0)
    _ = preprocessor.normalize(combined_train, fit=True)
    
    # Second pass: normalize and create sequences
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    
    for loc_name, splits in processed_data.items():
        df_train_norm = preprocessor.normalize(splits['train'], fit=False)
        df_val_norm = preprocessor.normalize(splits['val'], fit=False)
        
        X_train, y_train = preprocessor.create_sequences_no_leakage(
            df_train_norm, stride=stride, noise_level=noise_level
        )
        X_val, y_val = preprocessor.create_sequences_no_leakage(
            df_val_norm, stride=stride, noise_level=0.0
        )
        
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_val.append(X_val)
        all_y_val.append(y_val)
    
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    y_val = np.concatenate(all_y_val, axis=0)
    
    # Shuffle training
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    return X_train, y_train, X_val, y_val


def objective(trial: Trial, args, all_location_data: dict) -> float:
    """Optuna objective function - returns validation loss."""
    
    # Sample hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [32, 48, 64, 96])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    stride = trial.suggest_int('stride', 3, 6)
    input_noise = trial.suggest_float('input_noise', 0.0, 0.05, step=0.01)
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['cosine', 'plateau', 'step'])
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial.number}: hidden={hidden_size}, layers={num_layers}, "
                f"dropout={dropout:.2f}, lr={learning_rate:.2e}")
    
    # Create preprocessor and data for this trial
    preprocessor = DataPreprocessor()
    X_train, y_train, X_val, y_val = create_trial_data(
        all_location_data, preprocessor, stride, input_noise
    )
    
    # Create model
    model_config = ModelConfig(
        input_size=X_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=use_attention,
    )
    model = WeatherLSTM(model_config)
    
    # Create trainer
    train_config = TrainingConfig(
        epochs=args.epochs_per_trial,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=args.patience,
        weight_decay=weight_decay,
    )
    
    trainer = Trainer(model, config=train_config, lr_scheduler=lr_scheduler)
    
    # Train with pruning callback
    try:
        train_loader, val_loader = trainer.create_dataloaders(X_train, y_train, X_val, y_val)
        best_val_loss = float('inf')
        logger.info(f"Trial {trial.number}: Starting training ({len(X_train)} samples, {len(train_loader)} batches/epoch)")
        
        for epoch in range(args.epochs_per_trial):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            logger.info(f"  Epoch {epoch+1}/{args.epochs_per_trial} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            # Update best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            
            # Check if trial should be pruned (hopeless config)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
            
            # Early stopping within trial
            if trainer.early_stopping(val_loss):
                logger.info(f"Trial {trial.number} early stopped at epoch {epoch+1}")
                break
                
    except optuna.TrialPruned:
        raise  # Re-raise for Optuna to handle
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return float('inf')
    
    logger.info(f"Trial {trial.number} complete: val_loss={best_val_loss:.6f}")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    
    # HPO settings
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of Optuna trials')
    parser.add_argument('--epochs-per-trial', type=int, default=15,
                        help='Epochs per trial (keep low for faster search)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience per trial')
    parser.add_argument('--study-name', type=str, default='weather_hpo',
                        help='Optuna study name')
    
    # Data settings
    parser.add_argument('--location', type=str, default='berlin',
                        help='Single location to use')
    parser.add_argument('--multi-location', action='store_true',
                        help='Use all training locations')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)
    logger.info(f"Trials: {args.n_trials}, Epochs/trial: {args.epochs_per_trial}")
    
    # Prepare data once
    logger.info("\nLoading data (one-time)...")
    all_location_data = prepare_data(args)
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, all_location_data),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Results
    logger.info("\n" + "="*60)
    logger.info("HPO COMPLETE")
    logger.info("="*60)
    
    best = study.best_trial
    logger.info(f"\nBest Trial: {best.number}")
    logger.info(f"Best Val Loss: {best.value:.6f}")
    logger.info(f"\nBest Hyperparameters:")
    for key, value in best.params.items():
        logger.info(f"  --{key.replace('_', '-')}: {value}")
    
    # Save results
    results_dir = Path("hpo_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"hpo_{timestamp}.json"
    
    results = {
        'best_trial': best.number,
        'best_val_loss': best.value,
        'best_params': best.params,
        'n_trials': args.n_trials,
        'epochs_per_trial': args.epochs_per_trial,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print command to train with best params
    logger.info("\n" + "="*60)
    logger.info("TRAIN WITH BEST PARAMS:")
    logger.info("="*60)
    cmd = "python train.py"
    
    # Map HPO param names to train.py argument names
    param_map = {
        'learning_rate': 'lr',
        'input_noise': 'input-noise',
        'hidden_size': 'hidden-size',
        'num_layers': 'num-layers',
        'weight_decay': 'weight-decay',
        'batch_size': 'batch-size',
        'lr_scheduler': 'lr-scheduler',
    }
    
    for key, value in best.params.items():
        # Handle use_attention specially (train.py uses --no-attention flag)
        if key == 'use_attention':
            if not value:  # Only add flag if attention is disabled
                cmd += " --no-attention"
            continue
        
        # Map param name to train.py argument name
        arg_name = param_map.get(key, key.replace('_', '-'))
        cmd += f" --{arg_name} {value}"
    
    cmd += " --multi-location --epochs 50"
    logger.info(cmd)


if __name__ == "__main__":
    main()
