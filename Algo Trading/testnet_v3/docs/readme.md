# Algorithmic Trading System

This project implements an algorithmic trading system for cryptocurrency markets, focusing on futures trading on Binance.

## Project Structure

```
testnet_v3/
├── config.py                # Configuration settings for the entire system
├── data/                    # Market data and backtesting data
│   ├── backtest/            # Reserved data for backtesting
│   └── *_1h.csv             # Historical price data files
├── docs/                    # Documentation files
├── logs/                    # Log files
├── model/                   # Model definition and trained models
│   ├── LGBMmodel.py         # LightGBM model implementation
│   └── trained_models/      # Saved trained models
├── scripts/                 # Organized scripts for different purposes
│   ├── backtesting/         # Scripts for backtesting strategies
│   ├── experimental/        # Experimental and development scripts
│   ├── training/            # Model training scripts
│   └── utils/               # Utility scripts
├── strategy/                # Trading strategy implementations
├── utils/                   # Utility functions and classes
│   ├── feature_engineering.py # Feature engineering implementation
│   └── utils.py             # General utility functions
└── requirements.txt         # Python dependencies
```

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your environment:
   - Update `.env` file with your API credentials
   - Review settings in `config.py`

3. Train the model:
   ```
   python scripts/training/retrain_model_v3.py
   ```

4. Backtest the strategy (after implementing a backtesting script):
   ```
   python scripts/backtesting/backtest.py
   ```

## Key Components

- **Configuration**: All configuration is centralized in `config.py`
- **Data Management**: Data loading, preprocessing, and storage
- **Feature Engineering**: Technical indicators and feature generation
- **Model Training**: LightGBM models for price prediction
- **Strategy**: Trading logic based on model predictions
- **Backtesting**: Evaluation of strategy performance

## Documentation

Additional documentation can be found in the `docs` directory:
- [Model Training](docs/model_training.md)

## Development Practices

When contributing to this project, please follow these practices:
1. Use the appropriate directories for different types of scripts
2. Document your code with docstrings and comments
3. Update README files when adding new functionality
4. Keep experimental code in the experimental directory

## License

This project is for educational purposes only. Use at your own risk.