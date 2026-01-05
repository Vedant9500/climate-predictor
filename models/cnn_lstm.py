"""
CNN-LSTM Hybrid model for weather prediction (V2.0).
Uses 1D CNN for local pattern extraction + LSTM for temporal dependencies.
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
    Attention mechanism for LSTM outputs.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights.squeeze(-1)


class CNNLSTM(nn.Module):
    """
    V2.0: CNN-LSTM Hybrid model for weather prediction.
    
    Architecture:
    1. Conv1D layers extract local patterns (3-6 hour trends)
    2. MaxPool reduces sequence length
    3. LSTM captures long-term temporal dependencies
    4. Attention weights important time steps
    5. Output with physical clamping
    """
    
    # Output clamping ranges
    OUTPUT_CLAMPS = {
        0: (-50.0, 60.0),   # temperature_2m
        1: (0.0, 100.0),    # relative_humidity_2m
        2: (0.0, 500.0),    # precipitation
        3: (0.0, 200.0),    # wind_speed_10m
        4: (0.0, 100.0),    # cloud_cover
    }
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        output_size: int = 25,
        cnn_filters: int = 64,
        kernel_sizes: list = [3, 5],
        pool_size: int = 2,
        use_attention: bool = True,
        bidirectional: bool = True,
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of outputs (horizons * targets)
            cnn_filters: Number of CNN filters per kernel size
            kernel_sizes: List of CNN kernel sizes
            pool_size: Max pooling size
            use_attention: Whether to use attention
            bidirectional: Whether LSTM is bidirectional
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # Store config for checkpoint
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'output_size': output_size,
            'cnn_filters': cnn_filters,
            'kernel_sizes': kernel_sizes,
            'pool_size': pool_size,
            'use_attention': use_attention,
            'bidirectional': bidirectional,
        }
        
        # CNN Feature Extraction
        # Multiple kernel sizes for different temporal resolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, cnn_filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
            for k in kernel_sizes
        ])
        
        # Combined CNN output size
        cnn_output_size = cnn_filters * len(kernel_sizes)
        
        # Pooling to reduce sequence length
        self.pool = nn.MaxPool1d(pool_size)
        
        # Layer norm after CNN
        self.cnn_norm = nn.LayerNorm(cnn_output_size)
        
        # LSTM
        hidden_multiplier = 2 if bidirectional else 1
        self.effective_hidden = hidden_size * hidden_multiplier
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(self.effective_hidden)
        
        # Attention
        if use_attention:
            self.attention = Attention(self.effective_hidden)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.effective_hidden, self.effective_hidden)
        self.fc2 = nn.Linear(self.effective_hidden, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def _clamp_outputs(self, predictions: torch.Tensor) -> torch.Tensor:
        """Clamp outputs to valid physical ranges (inference only)."""
        n_targets = len(self.OUTPUT_CLAMPS)
        n_horizons = predictions.size(-1) // n_targets
        
        result = predictions.clone()
        reshaped = result.view(-1, n_horizons, n_targets)
        
        clamped_targets = []
        for target_idx in range(n_targets):
            if target_idx in self.OUTPUT_CLAMPS:
                min_val, max_val = self.OUTPUT_CLAMPS[target_idx]
                clamped_targets.append(torch.clamp(reshaped[:, :, target_idx], min=min_val, max=max_val))
            else:
                clamped_targets.append(reshaped[:, :, target_idx])
        
        clamped = torch.stack(clamped_targets, dim=-1)
        return clamped.view(-1, predictions.size(-1))
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        clamp_output: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            return_attention: Whether to return attention weights
            clamp_output: Whether to clamp outputs
            
        Returns:
            predictions, optional attention weights
        """
        batch_size = x.size(0)
        
        # CNN expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply multiple CNN kernels
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Concatenate along feature dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch, cnn_filters*n_kernels, seq_len)
        
        # Pool to reduce sequence length
        x = self.pool(x)
        
        # Back to (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # Layer norm
        x = self.cnn_norm(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Layer norm
        lstm_out = self.lstm_norm(lstm_out)
        
        # Get context vector
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            if self.bidirectional:
                context = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                context = h_n[-1]
            attn_weights = None
        
        # Output projection
        out = self.dropout(context)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        predictions = self.fc2(out)
        
        # Clamp during inference only
        if clamp_output and not self.training:
            predictions = self._clamp_outputs(predictions)
        
        if return_attention:
            return predictions, attn_weights
        return predictions
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    batch_size = 32
    seq_len = 48
    input_size = 55
    output_size = 25
    
    model = CNNLSTM(
        input_size=input_size,
        hidden_size=64,
        output_size=output_size,
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with attention
    output, attn = model(x, return_attention=True)
    if attn is not None:
        print(f"Attention shape: {attn.shape}")
