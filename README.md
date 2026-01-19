# Climate Predictor

LSTM-based weather prediction with multi-location training and specialist models.

## Quick Start

```bash
pip install -r requirements.txt

# Train general model (multi-location)
python train.py --multi-location --epochs 50

# Train a specialist model
python train.py --specialist temperature --multi-location --epochs 50
```

## Features

- **Multi-location training** - Berlin, Frankfurt, Hamburg, Vienna, Munich
- **Specialist models** - Focused models for temperature, precipitation, wind, cloud, humidity
- **Hyperparameter optimization** - Optuna-based HPO with pruning
- **5 forecast horizons** - +1h, +3h, +6h, +12h, +24h
- **No data leakage** - Temporal splits + target separation

## Key Commands

| Task | Command |
|------|---------|
| Train general model | `python train.py --multi-location` |
| Train specialist | `python train.py --specialist temperature --multi-location` |
| Run HPO | `python hpo.py --n-trials 20 --multi-location` |
| Backtest model | `python ensemble_predict.py --days-ago 3` |
| Compare rain models | `python compare_models.py --days 30` |

## Specialists

Models saved to `saved_models/specialists/{name}/`:

| Specialist | Target | Best For |
|------------|--------|----------|
| temperature | temperature_2m | R²=0.95, very accurate |
| precipitation | precipitation | Use rain specialists instead |
| wind | wind_speed_10m | Moderate accuracy |

## Project Structure

```
├── config/          # Settings and specialist configs
├── data/            # Fetching and preprocessing
├── models/          # LSTM architecture
├── training/        # Training and evaluation
├── saved_models/    # Checkpoints
│   └── specialists/ # Specialist model folders
└── hall of fame/    # Best performing models
```

## Data

- **Source**: [Open-Meteo Historical API](https://open-meteo.com)
- **Period**: 2015-2024
- **Locations**: European cities (configurable in `config/settings.py`)