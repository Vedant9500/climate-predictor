"""
Visualization utilities for weather prediction project.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import TARGET_VARIABLES, PREDICTION_HORIZONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style
plt.style.use('seaborn-v0_8-whitegrid')


class Visualizer:
    """Visualization tools for weather data and predictions."""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_time_series(
        self,
        df: pd.DataFrame,
        columns: List[str],
        title: str = "Weather Time Series",
        save_name: Optional[str] = None,
    ):
        """
        Plot time series of weather variables.
        
        Args:
            df: DataFrame with datetime index
            columns: Columns to plot
            title: Plot title
            save_name: Filename to save (without extension)
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(14, 3 * n_cols), sharex=True)
        
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, columns):
            if col in df.columns:
                ax.plot(df.index, df[col], linewidth=0.8)
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Time")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {self.save_dir / save_name}.png")
        
        plt.show()
    
    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex,
        variable_idx: int = 0,
        horizon_idx: int = 0,
        save_name: Optional[str] = None,
    ):
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            timestamps: Datetime index for x-axis
            variable_idx: Index of target variable to plot
            horizon_idx: Index of prediction horizon to plot
            save_name: Filename to save
        """
        n_targets = len(TARGET_VARIABLES)
        flat_idx = horizon_idx * n_targets + variable_idx
        
        var_name = TARGET_VARIABLES[variable_idx]
        horizon = PREDICTION_HORIZONS[horizon_idx]
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        ax.plot(timestamps[:len(y_true)], y_true[:, flat_idx], 
                label='Actual', alpha=0.8, linewidth=1)
        ax.plot(timestamps[:len(y_pred)], y_pred[:, flat_idx], 
                label='Predicted', alpha=0.8, linewidth=1)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(var_name)
        ax.set_title(f'{var_name} - {horizon}h Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict[str, list],
        save_name: Optional[str] = None,
    ):
        """
        Plot training and validation loss curves.
        
        Args:
            history: Dict with 'train_loss' and 'val_loss' lists
            save_name: Filename to save
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='o', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'lr' in history:
            ax2.plot(epochs, history['lr'], marker='o', markersize=3, color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_error_by_horizon(
        self,
        metrics_by_horizon: Dict[int, Dict[str, float]],
        metric: str = 'rmse',
        save_name: Optional[str] = None,
    ):
        """
        Plot error metric across prediction horizons.
        
        Args:
            metrics_by_horizon: Dict from evaluator.evaluate_by_horizon()
            metric: Which metric to plot
            save_name: Filename to save
        """
        horizons = list(metrics_by_horizon.keys())
        values = [metrics_by_horizon[h][metric] for h in horizons]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        bars = ax.bar(horizons, values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Forecast Horizon (hours)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Forecast Horizon')
        ax.set_xticks(horizons)
        ax.set_xticklabels([f'{h}h' for h in horizons])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_scatter_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        variable_name: str = "Value",
        save_name: Optional[str] = None,
    ):
        """
        Scatter plot of actual vs predicted values.
        
        Args:
            y_true: Actual values (flattened)
            y_pred: Predicted values (flattened)
            variable_name: Name for axis labels
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {variable_name}')
        ax.set_ylabel(f'Predicted {variable_name}')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Test visualization
    viz = Visualizer()
    
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'temperature_2m': np.random.randn(100).cumsum() + 20,
        'relative_humidity_2m': np.random.rand(100) * 50 + 50,
    }, index=dates)
    
    viz.plot_time_series(df, ['temperature_2m', 'relative_humidity_2m'], 
                        title="Test Weather Data")
