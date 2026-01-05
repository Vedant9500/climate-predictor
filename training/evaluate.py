"""
Model evaluation metrics and analysis.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import TARGET_VARIABLES, PREDICTION_HORIZONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate model predictions with various metrics."""
    
    def __init__(
        self,
        target_variables: List[str] = TARGET_VARIABLES,
        prediction_horizons: List[int] = PREDICTION_HORIZONS,
    ):
        self.target_variables = target_variables
        self.prediction_horizons = prediction_horizons
        
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mape(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared coefficient."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    def evaluate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dict of metric names to values
        """
        return {
            'mae': self.mae(y_true, y_pred),
            'rmse': self.rmse(y_true, y_pred),
            'mape': self.mape(y_true, y_pred),
            'r2': self.r2(y_true, y_pred),
        }
    
    def evaluate_by_horizon(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate metrics for each prediction horizon.
        
        Args:
            y_true: Ground truth (samples, horizons * targets)
            y_pred: Predictions (samples, horizons * targets)
            
        Returns:
            Dict mapping horizon to metrics
        """
        n_targets = len(self.target_variables)
        n_horizons = len(self.prediction_horizons)
        
        results = {}
        for h_idx, horizon in enumerate(self.prediction_horizons):
            start = h_idx * n_targets
            end = start + n_targets
            
            y_true_h = y_true[:, start:end]
            y_pred_h = y_pred[:, start:end]
            
            results[horizon] = self.evaluate_all(y_true_h, y_pred_h)
        
        return results
    
    def evaluate_by_variable(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics for each target variable.
        
        Args:
            y_true: Ground truth (samples, horizons * targets)
            y_pred: Predictions (samples, horizons * targets)
            
        Returns:
            Dict mapping variable name to metrics
        """
        n_targets = len(self.target_variables)
        n_horizons = len(self.prediction_horizons)
        
        results = {}
        for t_idx, var_name in enumerate(self.target_variables):
            # Collect all predictions for this variable across horizons
            indices = [h_idx * n_targets + t_idx for h_idx in range(n_horizons)]
            
            y_true_v = y_true[:, indices]
            y_pred_v = y_pred[:, indices]
            
            results[var_name] = self.evaluate_all(y_true_v, y_pred_v)
        
        return results
    
    def print_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """Print formatted evaluation report."""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        # Overall metrics
        overall = self.evaluate_all(y_true, y_pred)
        print("\n📊 Overall Metrics:")
        print(f"  MAE:  {overall['mae']:.4f}")
        print(f"  RMSE: {overall['rmse']:.4f}")
        print(f"  MAPE: {overall['mape']:.2f}%")
        print(f"  R²:   {overall['r2']:.4f}")
        
        # By horizon
        print("\n⏱️ Metrics by Forecast Horizon:")
        by_horizon = self.evaluate_by_horizon(y_true, y_pred)
        print(f"  {'Horizon':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("  " + "-"*40)
        for horizon, metrics in by_horizon.items():
            print(f"  {str(horizon)+'h':<10} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f}")
        
        # By variable
        print("\n🌡️ Metrics by Variable:")
        by_var = self.evaluate_by_variable(y_true, y_pred)
        print(f"  {'Variable':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("  " + "-"*55)
        for var_name, metrics in by_var.items():
            print(f"  {var_name:<25} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f}")
        
        print("\n" + "="*60)
        
        return {
            'overall': overall,
            'by_horizon': by_horizon,
            'by_variable': by_var,
        }


if __name__ == "__main__":
    # Test evaluator
    evaluator = Evaluator()
    
    # Create dummy data
    n_samples = 100
    n_outputs = len(TARGET_VARIABLES) * len(PREDICTION_HORIZONS)
    
    y_true = np.random.randn(n_samples, n_outputs)
    y_pred = y_true + np.random.randn(n_samples, n_outputs) * 0.1
    
    evaluator.print_report(y_true, y_pred)
