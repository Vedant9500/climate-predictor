"""
Data preprocessing pipeline for LSTM weather prediction.
Handles missing values, feature engineering, normalization, and sequence creation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler
import pickle
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    SEQUENCE_LENGTH,
    PREDICTION_HORIZONS,
    TARGET_VARIABLES,
    HOURLY_VARIABLES,
    PROCESSED_DATA_DIR,
    TrainingConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses weather data for LSTM training."""
    
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        prediction_horizons: List[int] = PREDICTION_HORIZONS,
        target_variables: List[int] = TARGET_VARIABLES,
    ):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Number of hours of history for input
            prediction_horizons: Hours ahead to predict
            target_variables: Variables to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.target_variables = target_variables
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data without data leakage.
        
        Args:
            df: Raw DataFrame from API
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data. Initial shape: {df.shape}")
        
        df = df.copy()
        
        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        for col in df.columns:
            if missing_pct[col] > 0:
                logger.info(f"  {col}: {missing_pct[col]:.2f}% missing")
        
        # Interpolate missing values (linear for time series)
        df = df.interpolate(method='linear', limit_direction='both')
        
        # FIX: Do NOT calculate quantiles on whole dataset (Leakage!)
        # Just fill remaining NaNs with column mean
        df = df.fillna(df.mean())
        
        logger.info(f"Cleaned data shape: {df.shape}")
        
        return df
    
    def remove_target_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove raw target variables from input features to prevent data leakage.
        
        The model should learn from:
        - Auxiliary weather features (pressure, radiation, etc.)
        - Time features (hour, day, month encoding)
        - Lag features of targets (these are properly shifted)
        
        NOT from:
        - Raw target values at current timestep (this causes leakage!)
        """
        df = df.copy()
        
        # Remove raw target columns (keep lag versions)
        cols_to_remove = []
        for col in self.target_variables:
            if col in df.columns:
                cols_to_remove.append(col)
        
        if cols_to_remove:
            # Store target columns separately for later use in y
            self._target_cols_for_y = cols_to_remove.copy()
            df = df.drop(columns=cols_to_remove)
            logger.info(f"Removed {len(cols_to_remove)} raw target columns to prevent leakage: {cols_to_remove}")
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        # Hour of day (cyclical encoding)
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of year (cyclical encoding)
        day_of_year = df.index.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Month (cyclical encoding)
        month = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Day of week (cyclical encoding)
        dow = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        logger.info(f"Added 8 time features")
        
        return df
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        lag_hours: List[int] = [1, 24],  # Reduced: force LSTM to learn patterns
    ) -> pd.DataFrame:
        """
        Add lag features for key variables.
        
        Args:
            df: DataFrame
            lag_hours: Hours to lag
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        # Only add lags for target variables to limit feature explosion
        for col in self.target_variables:
            if col in df.columns:
                for lag in lag_hours:
                    df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        
        # Drop rows with NaN from lagging
        df = df.dropna()
        
        logger.info(f"Added {len(self.target_variables) * len(lag_hours)} lag features")
        
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [6],  # Reduced: force LSTM to learn patterns
    ) -> pd.DataFrame:
        """
        Add rolling statistics.
        
        Args:
            df: DataFrame
            windows: Window sizes in hours
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for col in self.target_variables:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window).mean()
                    # Removed rolling_std - adds noise, not signal
        
        # Drop rows with NaN from rolling
        df = df.dropna()
        
        logger.info(f"Added {len(self.target_variables) * len(windows)} rolling features")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Feature-engineered DataFrame
        """
        df = self.clean_data(df)
        df = self.add_time_features(df)
        # REMOVED: Let LSTM learn patterns from raw sequence
        # df = self.add_lag_features(df)
        # df = self.add_rolling_features(df)
        
        # CRITICAL: Handle static features for multi-location training
        static_cols = ['latitude', 'longitude', 'elevation']
        for col in static_cols:
            if col not in df.columns:
                logger.warning(f"Static feature {col} missing! Filling with 0.")
                df[col] = 0.0
        
        # Normalize static features to ~[-1, 1] range BEFORE StandardScaler
        # This prevents them from dominating loss
        if 'latitude' in df.columns:
            df['latitude'] = (df['latitude'] - 45.0) / 15.0  # Europe: 30-60
        if 'longitude' in df.columns:
            df['longitude'] = (df['longitude'] - 10.0) / 15.0  # Europe: -5 to 25
        if 'elevation' in df.columns:
            df['elevation'] = df['elevation'] / 1000.0  # Scale to km
        
        self.feature_columns = df.columns.tolist()
        
        logger.info(f"Total features: {len(self.feature_columns)}")
        
        return df
    
    def normalize(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            df: DataFrame to normalize
            fit: Whether to fit the scaler (True for training)
            
        Returns:
            Normalized DataFrame
        """
        if fit:
            normalized = self.scaler.fit_transform(df)
        else:
            normalized = self.scaler.transform(df)
        
        return pd.DataFrame(
            normalized,
            index=df.index,
            columns=df.columns,
        )
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        df_targets: pd.DataFrame = None,
        stride: int = 6,
        noise_level: float = 0.0,  # Add Gaussian noise to prevent memorization
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            df: Normalized DataFrame for input features
            df_targets: Optional separate DataFrame containing target columns
                       (use this to prevent data leakage)
            stride: Step size between sequences (default 6 hours)
            noise_level: Std of Gaussian noise to add (0.0 = no noise)
                        Use 0.01-0.05 for training, 0.0 for val/test
            
        Returns:
            Tuple of (X, y) where:
                X: (samples, sequence_length, features)
                y: (samples, horizons * targets)
        """
        data = df.values
        
        # If separate targets df provided, use it; otherwise look in df
        if df_targets is not None:
            target_data = df_targets.values
            target_indices = list(range(len(self.target_variables)))
        else:
            target_data = data
            target_indices = [df.columns.get_loc(col) for col in self.target_variables if col in df.columns]
        
        X, y = [], []
        max_horizon = max(self.prediction_horizons)
        
        # Use stride to reduce sequence overlap
        for i in range(self.sequence_length, len(data) - max_horizon, stride):
            # Get input sequence
            seq = data[i - self.sequence_length:i].copy()
            
            # Add noise to prevent memorization (training only)
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, seq.shape)
                seq = seq + noise
            
            X.append(seq)
            
            # Targets at each horizon (from targets df)
            targets = []
            for h in self.prediction_horizons:
                for idx in target_indices:
                    targets.append(target_data[i + h - 1, idx])
            y.append(targets)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        logger.info(f"Created sequences - X: {X.shape}, y: {y.shape} (stride={stride}, noise={noise_level})")
        
        return X, y
    
    def create_sequences_no_leakage(
        self,
        df_full: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with built-in leakage prevention.
        
        Separates target variables from features before sequence creation.
        """
        # Extract target columns
        target_cols = [col for col in self.target_variables if col in df_full.columns]
        df_targets = df_full[target_cols].copy()
        
        # Remove targets from features
        df_features = df_full.drop(columns=target_cols)
        
        logger.info(f"Split data: {len(df_features.columns)} features, {len(target_cols)} targets")
        
        return self.create_sequences(df_features, df_targets)
    
    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig = TrainingConfig(),
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        Split data temporally for train/val/test.
        
        Args:
            X: Input sequences
            y: Target values
            config: Training configuration
            
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        n = len(X)
        train_end = int(n * config.train_split)
        val_end = int(n * (config.train_split + config.val_split))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save(self, path: str = PROCESSED_DATA_DIR):
        """Save preprocessor state (scalers, columns)."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "target_scaler": self.target_scaler,
                "feature_columns": self.feature_columns,
                "sequence_length": self.sequence_length,
                "prediction_horizons": self.prediction_horizons,
                "target_variables": self.target_variables,
            }, f)
        
        logger.info(f"Saved preprocessor to {save_dir}")
    
    @classmethod
    def load(cls, path: str = PROCESSED_DATA_DIR) -> "DataPreprocessor":
        """Load preprocessor state."""
        with open(Path(path) / "preprocessor.pkl", "rb") as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            sequence_length=state["sequence_length"],
            prediction_horizons=state["prediction_horizons"],
            target_variables=state["target_variables"],
        )
        preprocessor.scaler = state["scaler"]
        preprocessor.target_scaler = state["target_scaler"]
        preprocessor.feature_columns = state["feature_columns"]
        
        return preprocessor
    
    def denormalize_predictions(
        self,
        predictions: np.ndarray,
        df_reference: pd.DataFrame,
    ) -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
            df_reference: Reference DataFrame for column info
            
        Returns:
            Denormalized predictions
        """
        # This is a simplified version - for proper denormalization
        # we need to track the indices of target variables
        # For now, we'll use inverse_transform on a reconstructed array
        
        # Get the mean and std for target variables
        target_indices = [df_reference.columns.get_loc(col) 
                         for col in self.target_variables 
                         if col in df_reference.columns]
        
        means = self.scaler.mean_[target_indices]
        stds = self.scaler.scale_[target_indices]
        
        # Reshape predictions to denormalize
        n_targets = len(self.target_variables)
        n_horizons = len(self.prediction_horizons)
        
        denorm = predictions.copy()
        for i, (h_idx, t_idx) in enumerate(
            [(h, t) for h in range(n_horizons) for t in range(n_targets)]
        ):
            flat_idx = h_idx * n_targets + t_idx
            if flat_idx < denorm.shape[-1]:
                denorm[..., flat_idx] = denorm[..., flat_idx] * stds[t_idx] + means[t_idx]
        
        return denorm


if __name__ == "__main__":
    # Example usage
    from data.fetcher import WeatherDataFetcher
    
    # Fetch sample data
    fetcher = WeatherDataFetcher(location="berlin")
    df = fetcher.fetch_recent(days=30)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_features(df)
    df_normalized = preprocessor.normalize(df_processed)
    
    X, y = preprocessor.create_sequences(df_normalized)
    
    print(f"\nInput shape: {X.shape}")
    print(f"Output shape: {y.shape}")
