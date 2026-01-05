"""
V2.0 Training script for CNN-LSTM weather prediction model.
"""
import argparse
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    TrainingConfig,
    DATA_START_YEAR,
    DATA_END_YEAR,
    DEFAULT_LOCATION,
    PROCESSED_DATA_DIR,
    TARGET_VARIABLES,
    PREDICTION_HORIZONS,
)
from data.fetcher import WeatherDataFetcher
from data.preprocessor import DataPreprocessor
from models.cnn_lstm import CNNLSTM
from training.trainer import Trainer
from training.evaluate import Evaluator
from utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train V2.0 CNN-LSTM model')
    
    # Data arguments
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION)
    parser.add_argument('--start-year', type=int, default=DATA_START_YEAR)
    parser.add_argument('--end-year', type=int, default=DATA_END_YEAR)
    parser.add_argument('--skip-fetch', action='store_true')
    
    # Model arguments (V2 specific)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--cnn-filters', type=int, default=64)
    parser.add_argument('--kernel-sizes', type=str, default='3,5',
                       help='Comma-separated kernel sizes')
    parser.add_argument('--pool-size', type=int, default=2)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Parse kernel sizes
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    
    # =========================================================================
    # Step 1: Fetch Data
    # =========================================================================
    logger.info("="*60)
    logger.info("V2.0 CNN-LSTM - STEP 1: Fetching Data")
    logger.info("="*60)
    
    fetcher = WeatherDataFetcher(location=args.location)
    
    if args.skip_fetch:
        logger.info("Loading cached data...")
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
            raise FileNotFoundError("No cached data found.")
    else:
        df_raw = fetcher.fetch_years(args.start_year, args.end_year)
    
    logger.info(f"Raw data shape: {df_raw.shape}")
    
    # =========================================================================
    # Step 2: Preprocess Data
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("="*60)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_features(df_raw)
    df_normalized = preprocessor.normalize(df_processed, fit=True)
    
    X, y = preprocessor.create_sequences(df_normalized)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        preprocessor.train_val_test_split(X, y)
    
    preprocessor.save()
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # =========================================================================
    # Step 3: Create CNN-LSTM Model
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 3: Creating CNN-LSTM Model (V2.0)")
    logger.info("="*60)
    
    output_size = len(TARGET_VARIABLES) * len(PREDICTION_HORIZONS)
    
    model = CNNLSTM(
        input_size=X_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=output_size,
        cnn_filters=args.cnn_filters,
        kernel_sizes=kernel_sizes,
        pool_size=args.pool_size,
        use_attention=True,
        bidirectional=True,
    )
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"CNN filters: {args.cnn_filters}, Kernels: {kernel_sizes}")
    
    # =========================================================================
    # Step 4: Train Model
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 4: Training CNN-LSTM Model")
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
    # Step 5: Evaluate
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 5: Evaluating Model")
    logger.info("="*60)
    
    trainer.load_best_model()
    
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
    
    evaluator = Evaluator()
    results = evaluator.print_report(y_test, test_predictions)
    
    # =========================================================================
    # Step 6: Visualize
    # =========================================================================
    logger.info("="*60)
    logger.info("STEP 6: Generating Visualizations")
    logger.info("="*60)
    
    viz = Visualizer()
    viz.plot_training_history(history, save_name="v2_training_history")
    viz.plot_error_by_horizon(results['by_horizon'], save_name="v2_error_by_horizon")
    
    logger.info("="*60)
    logger.info("V2.0 TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best model saved to: saved_models/best_model.pt")


if __name__ == "__main__":
    main()
