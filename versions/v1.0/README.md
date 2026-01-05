# Climate Predictor V1.0

**Release Date:** January 2026  
**Status:** Baseline

## Architecture
- **Model:** Bidirectional LSTM with Attention
- **Hidden Size:** 64
- **Layers:** 1
- **Dropout:** 0.4
- **Sequence Length:** 72 hours (3 days)

## Features
- 22 weather variables from Open-Meteo API
- 85 engineered features (lags, rolling stats, time encoding)
- 5 target variables × 5 horizons = 25 outputs

## Test Metrics

| Variable | MAE | RMSE | R² |
|----------|-----|------|-----|
| temperature_2m | 0.19 | 0.26 | **93.2%** |
| relative_humidity_2m | 0.35 | 0.45 | 77.8% |
| precipitation | 0.36 | 1.17 | 8.1% |
| wind_speed_10m | 0.53 | 0.70 | 53.6% |
| cloud_cover | 0.70 | 0.85 | 32.8% |

## Real-World Backtest (Jan 2, 2026)

| Variable | MAE | Assessment |
|----------|-----|------------|
| temperature_2m | 1.89°C | Good |
| relative_humidity_2m | 6.16% | Moderate |
| precipitation | 0.30mm | Excellent |
| wind_speed_10m | 2.07 km/h | Good |
| cloud_cover | 7.50% | Poor |
| **Overall** | **3.58** | |

## Known Issues
- Cloud cover predictions sometimes exceed 100%
- Model overfits (train loss << val loss)
- Humidity bias (overpredicts)

## Files
- `best_model.pt` - PyTorch checkpoint
