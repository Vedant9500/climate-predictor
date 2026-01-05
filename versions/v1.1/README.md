# Climate Predictor V1.1

**Release Date:** January 2026  
**Status:** Current

## Changes from V1.0
- ✅ Added Layer Normalization after LSTM
- ✅ Output clamping (0-100% for humidity/cloud, etc.)
- ✅ Reduced sequence length: 72h → 48h
- ✅ Increased weight decay: 1e-5 → 1e-4
- ✅ Reduced features: 85 → ~55 (fewer lags/rolling)

## Architecture
- **Model:** Bidirectional LSTM + LayerNorm + Attention
- **Hidden Size:** 64
- **Layers:** 1
- **Dropout:** 0.4
- **Sequence Length:** 48 hours (2 days)

## Test Metrics

| Variable | MAE | RMSE | R² |
|----------|-----|------|-----|
| temperature_2m | 0.20 | 0.26 | **93.0%** |
| relative_humidity_2m | 0.55 | 0.78 | 33.1% |
| precipitation | 0.43 | 1.17 | 8.1% |
| wind_speed_10m | 0.66 | 0.82 | 36.1% |
| cloud_cover | 0.79 | 0.96 | 14.3% |

## Real-World Backtest (Jan 2, 2026)

| Variable | V1.0 | V1.1 | Change |
|----------|------|------|--------|
| temperature_2m | 1.89 | **1.47** | ✅ 22% better |
| relative_humidity | 6.16 | **5.63** | ✅ 9% better |
| precipitation | 0.30 | **0.26** | ✅ 13% better |
| wind_speed_10m | 2.07 | 2.58 | ❌ worse |
| cloud_cover | 7.50 | 10.59 | ❌ worse |
| **Overall** | 3.58 | 4.11 | |

## Key Findings
1. **Temperature improved significantly** (1.89 → 1.47 MAE)
2. Precipitation predictions more accurate
3. No more invalid values (103% cloud cover fixed)
4. Training more stable (less overfitting)
5. Cloud cover degraded - needs specialized model (V2.0)

## Recommendations
- Use V1.1 for temperature/precipitation forecasting
- Consider V2.0 with separate cloud cover model
- Explore CNN-LSTM hybrid for improved accuracy

## Files
- `best_model.pt` - PyTorch checkpoint
