# Algorithmic Trading System

A robust algorithmic trading system that combines technical analysis, statistical arbitrage, and machine learning for cryptocurrency futures trading on Binance.

## Features

- **Data Management**
  - Historical data fetching from Binance Futures
  - Efficient data preprocessing and storage
  - Support for multiple trading pairs and timeframes
  - Real-time data streaming capabilities
  - CSV data storage and retrieval
  - Consistent column naming across API and CSV data

- **Trading Strategies**
  - MACD-based strategy with RSI confirmation
  - Statistical arbitrage with Z-score monitoring
  - Dynamic position sizing based on risk management
  - Configurable risk parameters and indicators

- **Portfolio Management**
  - Position tracking and risk monitoring
  - Dynamic portfolio optimization
  - Support for multiple optimization strategies:
    - Mean-Variance Optimization
    - Risk Parity Optimization
    - Minimum Variance Optimization

- **Backtesting Framework**
  - Comprehensive performance metrics
  - Transaction cost modeling
  - Risk-adjusted return analysis
  - Trade history and equity curve tracking

## Data Format

### API Response and CSV Column Names

The system uses consistent column names across both API responses and CSV files:

```python
[
    'timestamp',      # Unix timestamp in milliseconds
    'open',          # Opening price
    'high',          # Highest price
    'low',           # Lowest price
    'close',         # Closing price
    'volume',        # Trading volume
    'close_time',    # Close timestamp
    'quote_volume',  # Quote asset volume
    'trades',        # Number of trades
    'taker_buy_base', # Taker buy base asset volume
    'taker_buy_quote', # Taker buy quote asset volume
    'ignore'         # Ignore field
]
```

### CSV File Naming Convention

CSV files are stored in the `data` directory with the following naming convention:
```
{symbol}_{timeframe}_{date}.csv
```

Example: `BTCUSDT_15m_2024-07-17.csv`

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── market_data.py      # Market data fetching and preprocessing
│   ├── strategy/
│   │   ├── base_strategy.py    # Base strategy class
│   │   ├── macd_strategy.py    # MACD strategy implementation
│   │   ├── statistical_arbitrage.py  # Statistical arbitrage strategy
│   │   ├── zscore_monitor.py   # Z-score monitoring system
│   │   ├── position_sizer.py   # Position sizing logic
│   │   └── backtest.py         # Backtesting framework
│   └── portfolio/
│       ├── portfolio_manager.py # Position and risk management
│       └── portfolio_optimizer.py # Portfolio optimization
├── docs/
│   └── statistical_arbitrage_strategy.md  # Strategy documentation
├── tests/
│   ├── test_data.py           # Market data tests
│   ├── test_strategy.py       # Strategy tests
│   ├── test_backtest.py       # Backtesting tests
│   └── test_infrastructure.py # Integration tests
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/algo-trading.git
cd algo-trading
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your Binance API credentials:
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## Usage

### Data Management

The system supports both API data fetching and CSV file storage:

```python
from src.data.market_data import MarketData

# Initialize with CSV support
market_data = MarketData(
    client=binance_client,
    use_csv=True,  # Enable CSV file storage/retrieval
    csv_dir='data'  # Directory for CSV files
)

# Fetch historical data (will save to CSV if use_csv=True)
data = market_data.fetch_historical_data(
    symbol='BTCUSDT',
    timeframe='15m',
    start_time='2024-07-01',
    end_time='2024-07-17'
)
```

### Running Tests

The project includes a comprehensive test suite to validate all components:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_infrastructure.py -v

# Run tests with coverage report
pytest --cov=src tests/
```

### Backtesting

To run a backtest:

```bash
# Run MACD strategy backtest
python src/scripts/run_macd_backtest.py

# Run statistical arbitrage backtest
python src/scripts/run_stat_arb_backtest.py
```

## Error Handling and Logging

The system implements comprehensive error handling and logging:

- All API calls and data processing operations are wrapped in try-except blocks
- Detailed error messages include exception class names and tracebacks
- Logging levels:
  - INFO: General system status and important events
  - DEBUG: Detailed operation information
  - WARNING: Potential issues that don't stop execution
  - ERROR: Critical issues that may affect system operation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

# Statistical Arbitrage Trading Strategy

## Overview
This project implements a statistical arbitrage trading strategy for cryptocurrency futures on Binance. The strategy identifies and exploits temporary price inefficiencies between correlated trading pairs.

## Trading Pairs
We focus on the following major cryptocurrency pairs for statistical arbitrage:

### Core Trading Pairs
- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- BNBUSDT (Binance Coin)
- SOLUSDT (Solana)

### Altcoin Pairs
- LINKUSDT (Chainlink)
- NEARUSDT (NEAR Protocol)
- WIFUSDT (WIF)
- AVAXUSDT (Avalanche)
- 1000SHIBUSDT (Shiba Inu)
- DOGEUSDT (Dogecoin)
- 1000PEPEUSDT (Pepe)
- WLDUSDT (Worldcoin)

### Selection Criteria
- High liquidity and trading volume
- Established market presence
- Sufficient historical data
- Strong correlation with major pairs

## Strategy Components
1. **Data Collection**
   - 15-minute candlestick data
   - 3 months of historical data
   - Real-time price updates

2. **Pair Selection**
   - Correlation analysis
   - Cointegration testing
   - Volatility assessment

3. **Signal Generation**
   - Z-score calculation
   - Mean reversion detection
   - Entry/exit signals

4. **Risk Management**
   - Position sizing
   - Stop-loss orders
   - Take-profit targets

## Backtesting
- Training period: First 2 months
- Testing period: Last month
- Performance metrics:
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
  - Profit factor

## Requirements
- Python 3.8+
- Binance API access
- Required Python packages (see requirements.txt)

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see .env.example)
4. Run data collection: `python src/scripts/download_binance_data.py`
5. Run backtest: `python src/scripts/run_stat_arb_backtest.py`

## Documentation
See the `docs` directory for detailed documentation:
- `statistical_arbitrage_strategy.md`: Strategy implementation details
- `API.md`: API integration documentation
- `design_notes.md`: System architecture and design decisions 