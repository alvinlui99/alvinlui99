# Algorithmic Trading System

A comprehensive algorithmic trading system using LightGBM for prediction and Binance Futures for execution.

## Project Structure

```
├── config/               # Configuration files and settings
├── core/                 # Core trading system components 
├── data/                 # Historical market data
├── logs/                 # Trading logs
├── model/                # ML model definitions
│   └── LGBMmodel.py      # LightGBM model implementation
├── strategy/             # Trading strategies
│   ├── strategy.py       # Base strategy class
│   └── LGBMstrategy.py   # LightGBM-based strategy 
├── utils/                # Utility functions
├── .env                  # Environment variables (API keys, etc.)
├── download_data.py      # Script to download historical data
├── main_trading.py       # Main trading execution script
├── requirements.txt      # Project dependencies
├── train_model.py        # Script to train and save models
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A Binance account with API keys
- Sufficient funds in your Binance Futures account

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables in the `.env` file:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   ```

## Usage

### 1. Download Historical Data

```bash
python download_data.py --symbols BTCUSDT ETHUSDT --days 60 --interval 1h
```

### 2. Train the Model

```bash
python train_model.py --symbols BTCUSDT ETHUSDT --days 60 --interval 1h --output models/lgbm_model
```

### 3. Run the Trading System

```bash
python main_trading.py
```

## Strategy Overview

The trading strategy uses a LightGBM model to predict future price movements. The model is trained on historical price data and technical indicators. Based on these predictions, the strategy generates buy/sell signals.

### Strategy Components

1. **Model Training (`train_model.py`):**
   - Loads historical price data
   - Calculates technical indicators
   - Trains a LightGBM model for each symbol
   - Saves the models for later use

2. **Signal Generation (`LGBMstrategy.py`):**
   - Loads the trained models
   - Processes live market data
   - Generates trading signals based on price predictions

3. **Trading Execution (`main_trading.py`):**
   - Sets up the trading environment
   - Connects to Binance API
   - Processes signals from the strategy
   - Executes trades accordingly

## Customization

### Adding New Features

To add new technical indicators or features, modify the `_add_features` method in `model/LGBMmodel.py`.

### Creating New Strategies

1. Create a new strategy file in the `strategy/` directory
2. Inherit from the base `Strategy` class
3. Implement the `get_signals` method
4. Update `main_trading.py` to use your new strategy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this system.