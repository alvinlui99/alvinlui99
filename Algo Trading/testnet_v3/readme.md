# Algorithmic Trading System

A comprehensive algorithmic trading system using LightGBM for prediction and Binance Futures for execution.

## System Architecture

This trading system consists of several modular components:

1. **Data Fetcher**: Retrieves market data from Binance Futures API
2. **Model**: Uses LightGBM to predict price movements
3. **Strategy**: Generates trading signals based on model predictions
4. **Portfolio Manager**: Constructs diversified portfolios from signals
5. **Order Executor**: Handles the execution of trades on the exchange
6. **Trading Cycle**: Orchestrates the entire trading process

## Project Structure

```
├── config/                # Configuration files and settings
├── core/                  # Core trading system components
│   ├── data_fetcher.py    # Fetches market data
│   ├── executor.py        # Executes trades
│   ├── portfolio_manager.py # Manages portfolio construction
│   └── trading_cycle.py   # Orchestrates the trading process
├── data/                  # Historical market data
├── logs/                  # Trading logs
├── model/                 # ML model definitions
│   └── LGBMmodel.py       # LightGBM model implementation
├── strategy/              # Trading strategies
│   ├── strategy.py        # Base strategy class
│   └── LGBMstrategy.py    # LightGBM-based strategy
├── utils/                 # Utility functions
│   └── feature_engineering.py # Technical indicator calculations
├── .env                   # Environment variables (API keys, etc.)
├── download_data.py       # Script to download historical data
├── main.py                # Main entry point for the trading system
├── train_model.py         # Script to train and save models
└── README.md              # Project documentation
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
# Run in test mode (no real orders)
python main.py --test --interval 60 --timeframe 1h

# Run in production mode
python main.py --interval 60 --timeframe 1h
```

Command line arguments:
- `--test`: Run in test mode (no real orders placed)
- `--interval`: Trading cycle interval in minutes
- `--timeframe`: Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- `--lookback`: Number of lookback periods for analysis
- `--cycles`: Number of cycles to run (None for infinite)

## System Components

### 1. Data Fetcher
Retrieves market data from Binance Futures API using their official client. It handles data formatting and error recovery.

### 2. Model (LGBMmodel)
Uses LightGBM to predict future price movements based on technical indicators. Features are calculated using the FeatureEngineer class.

### 3. Strategy (LGBMstrategy)
Processes model predictions and converts them into actionable trading signals with confidence scores.

### 4. Portfolio Manager
Constructs a diversified portfolio based on strategy signals, considering position sizing, risk management, and capital allocation.

### 5. Order Executor
Handles the execution of trades, managing order status, and cleanup operations.

### 6. Trading Cycle
Orchestrates the entire process, from data retrieval to execution, in scheduled intervals.

## Customization

### Adding New Features
To add new technical indicators, modify the `FeatureEngineer` class in `utils/feature_engineering.py`.

### Creating New Strategies
1. Create a new strategy file in the `strategy/` directory
2. Inherit from the base `Strategy` class
3. Implement the `get_signals` method
4. Update `main.py` to use your new strategy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this system.