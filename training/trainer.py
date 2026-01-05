"""
Training pipeline for weather LSTM model.
Includes training loop, early stopping, checkpointing, and TensorBoard logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, Callable
import logging
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import TrainingConfig, SAVED_MODELS_DIR, LOGS_DIR
from models.lstm import WeatherLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """Training pipeline for weather prediction model."""
    
    def __init__(
        self,
        model: WeatherLSTM,
        config: TrainingConfig = TrainingConfig(),
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: LSTM model to train
            config: Training configuration
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.patience)
        
        # Logging
        self.log_dir = Path(LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / f"run_{int(time.time())}")
        
        # Checkpointing
        self.save_dir = Path(SAVED_MODELS_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        
    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders from numpy arrays."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")
            logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Dict with training history
        """
        train_loader, val_loader = self.create_dataloaders(
            X_train, y_train, X_val, y_val
        )
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.2e}"
            )
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        self.writer.close()
        logger.info(f"Training complete. Best val_loss: {self.best_val_loss:.6f}")
        
        return history
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        checkpoint_path = self.save_dir / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            logger.warning("No best model checkpoint found")


if __name__ == "__main__":
    # Example usage
    from config.settings import ModelConfig
    
    config = ModelConfig()
    model = WeatherLSTM(config)
    
    trainer = Trainer(model)
    
    # Create dummy data for testing
    n_samples = 1000
    seq_len = 72
    X = np.random.randn(n_samples, seq_len, config.input_size).astype(np.float32)
    y = np.random.randn(n_samples, config.output_size).astype(np.float32)
    
    # Split
    train_size = int(0.8 * n_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    # Train (just 2 epochs for testing)
    trainer.config.epochs = 2
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    print(f"\nTraining complete. Final val_loss: {history['val_loss'][-1]:.6f}")
