# Climate Predictor Project - Complete Summary

## 🎯 GOAL
Build a weather prediction model that forecasts 5 variables at 5 horizons (1h, 3h, 6h, 12h, 24h) using historical weather data from Open-Meteo API.

**Target Variables:**
1. temperature_2m (°C)
2. relative_humidity_2m (%)
3. precipitation (mm)
4. wind_speed_10m (km/h)
5. cloud_cover (%)

---

## 📊 DATA

### Source
- **API:** Open-Meteo Historical Weather API
- **Location:** Berlin (52.52°N, 13.41°E)
- **Period:** 2015-2024 (10 years)
- **Resolution:** Hourly

### Raw Features (22 variables)
```
temperature_2m, apparent_temperature, dewpoint_2m, relative_humidity_2m,
surface_pressure, pressure_msl, wind_speed_10m, wind_direction_10m,
wind_gusts_10m, precipitation, rain, snowfall, snow_depth, cloud_cover,
cloud_cover_low, cloud_cover_mid, cloud_cover_high, shortwave_radiation,
direct_radiation, diffuse_radiation, sunshine_duration, weather_code
```

### Engineered Features (85 total after preprocessing)
- **Time features (8):** hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, dow_sin, dow_cos
- **Lag features (25):** 5 targets × 5 lags (1h, 3h, 6h, 12h, 24h)
- **Rolling features (30):** 5 targets × 3 windows (6h, 12h, 24h) × 2 stats (mean, std)

### Preprocessing Pipeline
1. Interpolate missing values (linear)
2. Fill remaining NaNs with median
3. Add cyclical time encoding
4. Add lag features for target variables
5. Add rolling mean/std statistics
6. Normalize with StandardScaler
7. Create sequences of 72 hours → predict 5 horizons × 5 targets = 25 outputs

### Data Splits
- Training: 70% (~61,270 samples)
- Validation: 15% (~13,129 samples)
- Test: 15% (~13,129 samples)

---

## 🏗️ MODELS TRIED

### V1.0 - Pure Bidirectional LSTM
```python
Architecture:
- Input projection: Linear(85 → hidden_size)
- LSTM: Bidirectional, num_layers, dropout
- Attention: Simple dot-product attention
- Output: FC → ReLU → FC → 25 outputs

Best config: hidden_size=64, num_layers=1, dropout=0.4
Parameters: ~100K
```

**Results:**
- Val loss: **0.51** ✅
- Temperature R²: 93.2%
- Humidity R²: 77.8%
- Precipitation R²: 8.1%
- Wind R²: 53.6%
- Cloud cover R²: 32.8%

**Issue:** Overfitting (train loss << val loss)

---

### V1.1 - LSTM + Layer Normalization + Output Clamping
```python
Changes from V1.0:
- Added LayerNorm after LSTM
- Output clamping to valid physical ranges
- Reduced sequence length: 72h → 48h
- Increased weight decay: 1e-5 → 1e-4
- Reduced features: 85 → 55
```

**Results:**
- Val loss: ~0.52
- Temperature R²: 93.0%
- Humidity R²: 33.1% ❌ (degraded)
- Real-world backtest: Temperature MAE improved 22%!

**Issue:** Underfitting on non-temperature variables

---

### V2.0 - CNN-LSTM Hybrid
```python
Architecture:
- Multi-kernel 1D CNN: Conv1d(k=3) + Conv1d(k=5), 64 filters each
- BatchNorm + ReLU after each conv
- MaxPool1d(pool_size=2) to reduce sequence
- LayerNorm after CNN
- Bidirectional LSTM
- LayerNorm after LSTM
- Attention + FC layers

Config: cnn_filters=64, kernel_sizes=[3,5], pool_size=2, hidden_size=64
Parameters: ~150K
```

**Results:**
- Val loss: **0.72** ❌ (WORSE than pure LSTM!)
- Temperature R²: 92.8%
- All other metrics similar or worse

**Issue:** CNN-LSTM performs WORSE than pure LSTM

---

## ❌ CURRENT PROBLEM

The CNN-LSTM model (V2.0) has **higher validation loss (0.72)** than pure LSTM (0.51), contradicting the expectation that CNN+LSTM should be better.

### Possible Causes
1. **MaxPooling destroys temporal information** - Weather patterns might need full resolution
2. **CNN kernel sizes wrong** - 3h and 5h might not capture meaningful patterns
3. **BatchNorm interfering** - Might disrupt temporal statistics
4. **Feature engineering already captures local patterns** - Lag features already provide what CNN would learn
5. **Data leakage or preprocessing issue** - Something in the pipeline
6. **Wrong CNN placement** - Should CNN be after LSTM instead?

### Observations
- Temperature prediction is consistently good (92-93% R²) across all models
- Precipitation is consistently bad (8% R²) - inherently hard to predict
- Cloud cover is problematic (14-33% R²)
- The validation loss plateau at 0.72 for CNN-LSTM is suspicious

---

## 📁 PROJECT STRUCTURE

```
climate-predictor-main/
├── config/
│   └── settings.py          # All hyperparameters and config
├── data/
│   ├── fetcher.py           # Open-Meteo API client
│   ├── preprocessor.py      # Feature engineering, normalization
│   └── raw/                  # Cached parquet files (10 years)
├── models/
│   ├── lstm.py              # V1.0/V1.1 Pure LSTM
│   └── cnn_lstm.py          # V2.0 CNN-LSTM hybrid
├── training/
│   ├── trainer.py           # Training loop, early stopping
│   └── evaluate.py          # Metrics calculation
├── inference/
│   └── predictor.py         # Prediction pipeline
├── utils/
│   └── visualization.py     # Plotting utilities
├── train_v1-v1.1.py         # Training script for LSTM
├── train_v2.py              # Training script for CNN-LSTM
├── backtest.py              # Real-world validation
└── versions/
    ├── v1.0/                # Baseline LSTM
    └── v1.1/                # LSTM + LayerNorm
```

---

## 🔧 KEY CODE SNIPPETS

### CNN-LSTM Forward Pass (cnn_lstm.py)
```python
def forward(self, x):
    # x: (batch, seq_len=72, features=85)
    
    # CNN expects (batch, features, seq_len)
    x = x.permute(0, 2, 1)
    
    # Multiple CNN kernels
    conv_outputs = [conv(x) for conv in self.convs]  # k=3 and k=5
    x = torch.cat(conv_outputs, dim=1)  # (batch, 128, seq_len)
    
    # Pool to reduce sequence
    x = self.pool(x)  # (batch, 128, seq_len/2)
    
    # Back to (batch, seq_len, features)
    x = x.permute(0, 2, 1)
    x = self.cnn_norm(x)
    
    # LSTM
    lstm_out, _ = self.lstm(x)
    lstm_out = self.lstm_norm(lstm_out)
    
    # Attention + output
    context, _ = self.attention(lstm_out)
    out = self.fc2(relu(self.fc1(dropout(context))))
    return out
```

### Pure LSTM Forward Pass (lstm.py)
```python
def forward(self, x):
    # x: (batch, seq_len=72, features=85)
    
    x = self.input_projection(x)  # Project to hidden_size
    lstm_out, (h_n, c_n) = self.lstm(x)
    lstm_out = self.layer_norm(lstm_out)
    
    context, _ = self.attention(lstm_out)
    out = self.fc2(relu(self.fc1(dropout(context))))
    return out
```

---

## ❓ QUESTIONS FOR INVESTIGATION

1. **Why does CNN+LSTM perform worse?** Is MaxPooling the culprit?
2. **Should we remove pooling entirely?** Just use CNN for feature extraction
3. **Are the CNN kernel sizes appropriate?** 3h and 5h might be too short/long
4. **Is BatchNorm after Conv1d appropriate for time series?**
5. **Should CNN come AFTER LSTM instead of before?**
6. **Is our lag feature engineering redundant with CNN?**
7. **Would different CNN architectures help?** (Dilated conv, causal conv, etc.)

---

## 🎯 SUCCESS CRITERIA

A successful model should achieve:
- Temperature MAE < 1.5°C (currently ~1.5-1.9)
- Validation loss < 0.50
- No overfitting (train_loss ≈ val_loss)
- Better cloud cover prediction (R² > 50%)
