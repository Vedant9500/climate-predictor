# Overfitting Analysis - Climate Predictor

**Date:** 2026-01-15  
**Analyzed by:** Lime (Clawdbot)

---

## Summary

The model consistently overfits despite regularization attempts. After reviewing the codebase, I identified **3 root causes** related to data leakage and temporal splitting.

---

## Issue 1: Data Leakage (Target Variables in Input Features)

**Location:** `data/preprocessor.py` → `create_sequences()`

**Problem:**  
The `HOURLY_VARIABLES` in `config/settings.py` includes target variables like `temperature_2m`, `relative_humidity_2m`, etc. When sequences are created, the model receives the **current value** of these targets as input features, making it trivial to predict future values by learning small deltas.

```python
# config/settings.py
HOURLY_VARIABLES = [
    "temperature_2m",        # <-- Also a TARGET
    "relative_humidity_2m",  # <-- Also a TARGET
    ...
]

TARGET_VARIABLES = [
    "temperature_2m",        # <-- Duplicated!
    "relative_humidity_2m",  # <-- Duplicated!
    ...
]
```

**Why it causes overfitting:**  
The model "cheats" by copying the current temperature and adding a small learned offset. This results in very low training loss but poor generalization because the model never learns the underlying weather dynamics.

**Fix:**  
Remove raw target variables from input features. Only use their **lagged versions** (e.g., `temperature_2m_lag_1h`, `temperature_2m_lag_24h`).

```python
# In prepare_features() or create_sequences():
# Option A: Explicitly remove targets from features
input_features = [col for col in df.columns if col not in TARGET_VARIABLES]

# Option B: Use create_sequences_no_leakage() which already exists but isn't called
X, y = preprocessor.create_sequences_no_leakage(df_normalized, stride=3, noise_level=0.01)
```

---

## Issue 2: Temporal Leakage in Train/Val/Test Split

**Location:** `train.py` → main loop

**Problem:**  
Data is split per-location (70/15/15), then combined and shuffled. Since all locations share the same time range (2015-2024), overlapping time windows across cities allow the model to "see" patterns from validation/test periods during training.

```python
# Current approach in train.py:
# 1. For each location: split 70/15/15 by index
# 2. Combine all locations
# 3. Shuffle training data

# This means:
# - Berlin's training data includes Jan 2020
# - Munich's validation data includes Jan 2020
# - Similar weather patterns leak across splits!
```

**Fix:**  
Use **strict temporal splitting** across ALL locations:

```python
# Better approach:
TRAIN_YEARS = range(2015, 2022)  # 2015-2021
VAL_YEARS = [2022]               # 2022
TEST_YEARS = [2023, 2024]        # 2023-2024

# Split by year BEFORE combining locations
for loc_name, df_raw in all_location_data.items():
    df_train = df_raw[df_raw.index.year.isin(TRAIN_YEARS)]
    df_val = df_raw[df_raw.index.year.isin(VAL_YEARS)]
    df_test = df_raw[df_raw.index.year.isin(TEST_YEARS)]
```

---

## Issue 3: Look-Ahead Bias in Data Cleaning

**Location:** `data/preprocessor.py` → `clean_data()`

**Problem:**  
Missing values are filled using the mean of the **entire dataframe**, which includes future data relative to early training samples.

```python
# Current code:
df = df.fillna(df.mean())  # <-- df.mean() uses ALL data including future
```

**Fix:**  
Use **forward-fill** or compute statistics only on the training portion:

```python
# Option A: Forward-fill (causal, no future info)
df = df.fillna(method='ffill').fillna(method='bfill')

# Option B: Fill with training mean only (computed separately)
if fit:
    self._fill_values = df_train.mean()
df = df.fillna(self._fill_values)
```

---

## Additional Recommendations

### R1: Reduce Model Capacity

Current config may be too large for the data diversity:

| Parameter | Current | Recommended |
|-----------|---------|-------------|
| `hidden_size` | 48 | 32 |
| `num_layers` | 2 | 1 |
| `dropout` | 0.25 | 0.3 |
| `weight_decay` | 0.01 | 0.05 |

### R2: Add Early Stopping Monitoring

Currently early stopping watches `val_loss`. Consider also monitoring the **gap** between train and val loss:

```python
# If train_loss << val_loss, model is overfitting
overfit_gap = train_loss / val_loss
if overfit_gap < 0.5:  # train is 2x better than val
    logger.warning("Potential overfitting detected!")
```

### R3: Use Proper Cross-Validation

For time series, use **Walk-Forward Validation** instead of random splits:

```
Fold 1: Train 2015-2018, Val 2019
Fold 2: Train 2015-2019, Val 2020
Fold 3: Train 2015-2020, Val 2021
...
```

---

## Priority Order

1. **[CRITICAL]** Fix Issue 1 (Data Leakage) - This is likely the main cause
2. **[HIGH]** Fix Issue 2 (Temporal Splitting)
3. **[MEDIUM]** Fix Issue 3 (Look-Ahead Bias)
4. **[LOW]** Apply R1-R3 recommendations

---

## Files to Modify

| File | Changes |
|------|---------|
| `data/preprocessor.py` | Remove targets from features, fix fillna |
| `train.py` | Implement year-based temporal split |
| `config/settings.py` | Optionally reduce model capacity |

---

## Quick Test

After applying fixes, compare:

```
Before: Train Loss ~0.01, Val Loss ~0.15 (15x gap = severe overfitting)
After:  Train Loss ~0.08, Val Loss ~0.10 (1.25x gap = healthy)
```

If train/val losses are similar, the model is generalizing properly.

---

*Let me know if you want me to apply these fixes directly!*
