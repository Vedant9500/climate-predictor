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
)
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
    
    # Model arguments
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--no-attention', action='store_true',
                       help='Disable attention mechanism')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Other
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda)')
    
    args = parser.parse_args()
    
    # =========================================================================
    # Step 1: Fetch Data
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 1: Fetching Data")
    logger.info("="*60)
    
    fetcher = WeatherDataFetcher(location=args.location)
    
    if args.skip_fetch:
        logger.info("Skipping fetch, loading cached data...")
        # Load from parquet files
        from pathlib import Path
        data_dir = Path("data/raw")
        all_data = []
        for year in range(args.start_year, args.end_year + 1):
            year_file = data_dir / f"{args.location}_{year}.parquet"
            if year_file.exists():
                import pandas as pd
                all_data.append(pd.read_parquet(year_file))
        if all_data:
            import pandas as pd
            df_raw = pd.concat(all_data, axis=0)
            df_raw.sort_index(inplace=True)
        else:
            raise FileNotFoundError("No cached data found. Run without --skip-fetch first.")
    else:
        df_raw = fetcher.fetch_years(args.start_year, args.end_year)
    
    logger.info(f"Raw data shape: {df_raw.shape}")
    logger.info(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")
    
    # =========================================================================
    # Step 2: Preprocess Data
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("="*60)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_features(df_raw)
    df_normalized = preprocessor.normalize(df_processed, fit=True)
    
    # Create sequences
    X, y = preprocessor.create_sequences(df_normalized)
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        preprocessor.train_val_test_split(X, y)
    
    # Save preprocessor
    preprocessor.save()
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # =========================================================================
    # Step 3: Create Model
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 3: Creating Model")
    logger.info("="*60)
    
    model_config = ModelConfig(
        input_size=X_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_attention=not args.no_attention,
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
    )
    
    trainer = Trainer(model, config=train_config, device=args.device)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # Step 5: Evaluate on Test Set
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 5: Evaluating Model")
    logger.info("="*60)
    
    # Load best model
    trainer.load_best_model()
    
    # Get test predictions
    import torch
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
        test_predictions = model(X_test_tensor).cpu().numpy()
    
    # Evaluate
    evaluator = Evaluator()
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
