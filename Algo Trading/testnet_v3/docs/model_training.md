# Model Training Documentation

## Overview

This document describes the model training process for the algorithmic trading system. The system uses LightGBM models to predict price movements for various cryptocurrencies.

## Training Process

The training process consists of the following steps:

1. **Data Loading**: Load historical price data for all symbols defined in `config.py`.
2. **Data Splitting**: Split the data into training (60%), validation (20%), and test/backtesting (20%) sets.
3. **Feature Engineering**: Generate technical indicators and other features using the `FeatureEngineer` class.
4. **Model Training**: Train a separate LightGBM model for each symbol.
5. **Model Saving**: Save the trained models and metadata for later use.

## Latest Training Script

The latest training script is `scripts/training/retrain_model_v3.py`. This script:

- Uses the symbols defined in `config.py`
- Applies feature engineering based on `ModelConfig.FEATURE_CONFIG`
- Saves test data separately for future backtesting
- Handles missing data and NaN values
- Saves trained models in the `model/trained_models` directory

## Available Models

Models have been trained for the following symbols:
- BTCUSDT
- ETHUSDT
- BNBUSDT
- ADAUSDT
- SOLUSDT
- DOGEUSDT
- LINKUSDT
- AVAXUSDT

## Backtesting Data

Reserved data for backtesting is stored in the `data/backtest` directory. Each symbol has its own CSV file with 20% of the most recent data.

## Next Steps

After training, the next step is to use the trained models for backtesting to evaluate the trading strategy's performance. 