# Climate Predictor

LSTM-based weather prediction model using Open-Meteo historical data.

## Features
- 12-24 hour weather forecasting
- 30+ weather variables (temperature, humidity, wind, precipitation, solar, etc.)
- PyTorch LSTM with optional attention mechanism
- Clean modular architecture

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Fetch Historical Data
```bash
python main.py fetch --location berlin --years 10
```

### 2. Train Model
```bash
python train.py --epochs 50 --batch-size 64
```

### 3. Predict
```bash
python predict.py --hours 24
```

## Project Structure
```
├── config/         # Configuration settings
├── data/           # Data fetching and preprocessing
├── models/         # LSTM model architecture
├── training/       # Training and evaluation
├── inference/      # Prediction pipeline
├── utils/          # Visualization utilities
└── saved_models/   # Model checkpoints
```

## Data Source
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

## Location
Berlin (52.52°N, 13.41°E) - High resolution data available