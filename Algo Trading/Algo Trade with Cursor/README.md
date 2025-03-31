# Algorithmic Trading System

A robust algorithmic trading system that combines technical analysis, portfolio optimization, and risk management for cryptocurrency futures trading on Binance.

## Features

- **Data Management**
  - Historical data fetching from Binance Futures
  - Efficient data preprocessing and storage
  - Support for multiple trading pairs and timeframes

- **Trading Strategy**
  - MACD-based strategy with RSI confirmation
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

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── market_data.py      # Market data fetching and preprocessing
│   ├── strategy/
│   │   ├── base_strategy.py    # Base strategy class
│   │   ├── macd_strategy.py    # MACD strategy implementation
│   │   └── backtest.py         # Backtesting framework
│   └── portfolio/
│       ├── portfolio_manager.py # Position and risk management
│       └── portfolio_optimizer.py # Portfolio optimization
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

```python
from src.strategy.macd_strategy import MACDStrategy
from src.strategy.backtest import Backtest
from src.data.market_data import MarketData

# Initialize components
strategy = MACDStrategy(
    trading_pairs=['BTCUSDT', 'ETHUSDT'],
    timeframe='1h',
    risk_per_trade=0.02
)
backtest = Backtest(strategy=strategy, initial_capital=10000)

# Run backtest
results = backtest.run(market_data)
```

### Portfolio Optimization

```python
from src.portfolio.portfolio_optimizer import PortfolioOptimizer

# Initialize optimizer
optimizer = PortfolioOptimizer(
    symbols=['BTCUSDT', 'ETHUSDT'],
    returns=returns_data,
    risk_free_rate=0.02
)

# Get optimal weights
weights, return_, risk = optimizer.minimum_variance_optimization()
```

## Risk Management

The system implements several risk management features:

- Position sizing based on account risk per trade
- Maximum position size limits per asset
- Portfolio-level drawdown monitoring
- Dynamic stop-loss and take-profit levels
- Commission and slippage modeling

## Performance Metrics

The backtesting framework calculates various performance metrics:

- Total and annual returns
- Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Average win/loss
- Total commission paid
- Trade duration analysis

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