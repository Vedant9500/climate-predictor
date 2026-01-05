"""
LSTM model for weather prediction.
Supports bidirectional LSTM with optional attention mechanism.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import ModelConfig


class Attention(nn.Module):
    """
    Simple attention mechanism for LSTM outputs.
    Learns to weight different time steps.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs.
        
        Args:
            lstm_output: (batch, seq_len, hidden_size)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Calculate attention weights
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        return context, weights.squeeze(-1)


class WeatherLSTM(nn.Module):
    """
    LSTM model for multi-horizon weather prediction.
    V1.1: Added layer normalization and output clamping.
    """
    
    # Output clamping ranges for each target variable (in order)
    # temperature, humidity, precipitation, wind, cloud_cover
    OUTPUT_CLAMPS = {
        0: (-50.0, 60.0),   # temperature_2m: -50 to 60°C
        1: (0.0, 100.0),    # relative_humidity_2m: 0-100%
        2: (0.0, 500.0),    # precipitation: 0-500mm (realistic max)
        3: (0.0, 200.0),    # wind_speed_10m: 0-200 km/h
        4: (0.0, 100.0),    # cloud_cover: 0-100%
    }
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        """
        Initialize the LSTM model.
        
        Args:
            config: Model configuration dataclass
        """
        super().__init__()
        self.config = config
        
        # Calculate effective hidden size for bidirectional
        hidden_multiplier = 2 if config.bidirectional else 1
        self.effective_hidden = config.hidden_size * hidden_multiplier
        
        # Input projection
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        
        # V1.1: Layer normalization (better than batch norm for RNNs)
        self.layer_norm = nn.LayerNorm(self.effective_hidden)
        
        # Attention layer (optional)
        self.use_attention = config.use_attention
        if self.use_attention:
            self.attention = Attention(self.effective_hidden)
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(self.effective_hidden, self.effective_hidden // 2)
        self.fc2 = nn.Linear(self.effective_hidden // 2, config.output_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def _clamp_outputs(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        V1.1: Clamp outputs to valid physical ranges.
        
        Args:
            predictions: (batch, output_size) - flattened predictions
                         Structure: [h1_t1, h1_t2, ..., h1_t5, h2_t1, ..., h5_t5]
                         where h=horizon, t=target
        """
        n_targets = len(self.OUTPUT_CLAMPS)
        n_horizons = predictions.size(-1) // n_targets
        
        # Reshape to (batch, horizons, targets)
        reshaped = predictions.view(-1, n_horizons, n_targets)
        
        # Clamp each target variable
        for target_idx, (min_val, max_val) in self.OUTPUT_CLAMPS.items():
            if target_idx < n_targets:
                reshaped[:, :, target_idx] = torch.clamp(
                    reshaped[:, :, target_idx], min=min_val, max=max_val
                )
        
        # Reshape back
        return reshaped.view(-1, predictions.size(-1))
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        clamp_output: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            return_attention: Whether to return attention weights
            clamp_output: Whether to clamp outputs to valid ranges (V1.1)
            
        Returns:
            Predictions (batch, output_size)
            Optional attention weights (batch, seq_len)
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # V1.1: Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Get context vector
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            # Use last hidden state (concatenate directions if bidirectional)
            if self.config.bidirectional:
                context = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                context = h_n[-1]
            attn_weights = None
        
        # Output projection
        out = self.dropout(context)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        predictions = self.fc2(out)
        
        # V1.1: Clamp outputs to valid physical ranges
        if clamp_output:
            predictions = self._clamp_outputs(predictions)
        
        if return_attention:
            return predictions, attn_weights
        return predictions
    
    def predict(
        self,
        x: torch.Tensor,
        n_targets: int = 5,
        n_horizons: int = 5,
    ) -> dict:
        """
        Structured prediction output.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            n_targets: Number of target variables
            n_horizons: Number of prediction horizons
            
        Returns:
            Dict with predictions organized by horizon and variable
        """
        self.eval()
        with torch.no_grad():
            predictions = self(x)
        
        # Reshape: (batch, horizons, targets)
        predictions = predictions.view(-1, n_horizons, n_targets)
        
        return predictions
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    config = ModelConfig()
    model = WeatherLSTM(config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 32
    seq_len = 72  # 3 days
    x = torch.randn(batch_size, seq_len, config.input_size)
    
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with attention
    output, attn = model(x, return_attention=True)
    if attn is not None:
        print(f"Attention shape: {attn.shape}")
