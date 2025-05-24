# Algorithmic Trading System

A comprehensive algorithmic trading system that integrates technical analysis, portfolio optimization, and risk management for cryptocurrency trading on Binance Futures.

## Current Infrastructure

### 1. Backtesting Engine
- Full event-driven backtesting system
- Multi-asset support
- Commission handling
- Position tracking
- Performance metrics
- Trade history
- Visualization tools

### 2. Portfolio Management
- Position tracking and sizing
- Capital allocation
- Portfolio optimization (mean-variance, risk parity, etc.)
- Risk management
- Performance tracking

### 3. Strategy Framework
- Base strategy class for easy implementation
- Signal generation interface
- Position sizing interface
- Technical indicator support
- Multi-timeframe analysis

### 4. Exchange Integration
- Binance Futures API integration
- Real-time data handling
- Order execution
- Account management
- Position tracking

### 5. Data Management
- Historical data fetching
- Data caching
- Real-time updates
- Multiple timeframe support

## Next Steps

### 1. Strategy Development
- Implement and test various trading strategies
- Optimize strategy parameters
- Combine multiple signals
- Add machine learning models
- Implement portfolio-level strategies

### 2. Risk Management Enhancement
- Dynamic position sizing
- Correlation-based exposure limits
- Volatility-based adjustments
- Drawdown protection
- Portfolio-level stop-loss

### 3. Performance Optimization
- Strategy parameter optimization
- Portfolio weight optimization
- Transaction cost analysis
- Slippage modeling
- Market impact analysis

### 4. Monitoring & Alerts
- Real-time performance tracking
- Risk metrics monitoring
- Alert system for various conditions
- Automated reporting
- System health checks

## Project Structure

```
algo_trading/
├── src/
│   ├── strategy/
│   │   ├── backtest.py          # Backtesting engine
│   │   ├── base_strategy.py     # Strategy interface
│   │   └── macd_strategy.py     # Example strategy
│   ├── portfolio/
│   │   ├── portfolio_manager.py # Position and capital management
│   │   └── portfolio_optimizer.py # Portfolio optimization
│   ├── trading/
│   │   └── trading_bot.py      # Exchange integration
│   └── data/
│       └── market_data.py      # Data management
├── tests/
│   ├── test_backtest.py
│   ├── test_portfolio_manager.py
│   ├── test_portfolio_optimizer.py
│   └── backtest_visualization.py
├── docs/
│   └── README.md
├── config/
│   └── config.yaml
├── data/
│   └── historical_data/
├── plots/
│   └── backtest_results/
├── setup.py
├── requirements.txt
├── environment.yml
├── .env
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/algo_trading.git
cd algo_trading
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Binance API keys in `.env`:
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## Usage

### Running a Backtest

```python
from src.strategy.macd_strategy import MACDStrategy
from src.strategy.backtest import Backtest
from src.data.market_data import MarketData

# Initialize strategy
strategy = MACDStrategy(
    trading_pairs=['BTCUSDT'],
    timeframe='1h',
    rsi_period=14,
    risk_per_trade=0.02
)

# Initialize backtest
backtest = Backtest(
    strategy=strategy,
    initial_capital=10000,
    commission=0.0004
)

# Run backtest
results = backtest.run(historical_data)

# Print results
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
print(f"Win Rate: {results['win_rate']:.2f}%")
```

### Portfolio Optimization

```python
from src.portfolio.portfolio_optimizer import PortfolioOptimizer

# Initialize optimizer
optimizer = PortfolioOptimizer(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    returns=historical_returns,
    risk_free_rate=0.02
)

# Get optimal weights
weights = optimizer.get_optimal_weights(strategy='maximum_sharpe')
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the open-source community for various technical indicators and optimization algorithms
- Special thanks to contributors and maintainers of key dependencies 

## Quick Testing Feature

The system includes a quick testing feature that allows you to run shorter backtests for debugging and development purposes. This is controlled by the `test_duration_hours` parameter in the configuration.

### Usage

```python
config = {
    # ... other config parameters ...
    'test_duration_hours': 1  # Test duration in hours
}
```

### Options

- `1`: Last hour of data (default for quick testing)
- `0.5`: Last 30 minutes
- `2`: Last 2 hours
- `None`: Full backtest (use for production runs)

### Benefits

1. **Faster Development**
   - Quick feedback on strategy changes
   - Reduced waiting time for backtest results
   - Easier debugging of specific issues

2. **Resource Efficiency**
   - Lower memory usage
   - Faster execution
   - Reduced disk I/O

3. **Focused Testing**
   - Test specific market conditions
   - Verify recent strategy behavior
   - Validate capital changes

### Example

```python
# Quick test with 1 hour of data
config = {
    'test_duration_hours': 1,
    'symbols': ['BTCUSDT', 'ETHUSDT'],
    'timeframe': '15m'
}

# Full backtest
config = {
    'test_duration_hours': None,  # Use all available data
    'symbols': ['BTCUSDT', 'ETHUSDT'],
    'timeframe': '15m'
}
```

### Notes

- Use quick testing during development and debugging
- Always run full backtests before deploying to production
- Results from quick tests may not be representative of full backtest performance
- Consider market conditions when interpreting quick test results 

## Latest Backtest Results (20250407_200339)

### Performance Summary
- Strategy shows potential but needs optimization
- All periods show negative total returns due to high commission costs
- High trading frequency (17,000+ trades per period)
- Low win rates (7.80% - 8.13%) but positive profit factors
- Extreme drawdowns and long durations

### Recent Changes
- Changed timeframe from 15-minute to 1-hour intervals
- Expected to reduce trading frequency and commission costs
- Should improve signal quality and risk management

### Improvement Plan
1. **Reduce Trading Frequency**
   - Move to longer timeframe (1h or 4h)
   - Add minimum time between trades
   - Implement trade frequency limits

2. **Improve Risk Management**
   - Reduce max position size
   - Add position correlation filters
   - Implement maximum concurrent position limits
   - Widen stop loss and take profit levels

3. **Optimize Entry/Exit**
   - Add more stringent entry filters
   - Implement minimum holding periods
   - Add trend and volatility filters

4. **Commission Management**
   - Add commission-aware position sizing
   - Implement minimum profit targets
   - Optimize trade frequency

5. **Performance Monitoring**
   - Add detailed commission tracking
   - Implement trade quality metrics
   - Track strategy performance by time of day

For detailed analysis and complete improvement tasks, see [Statistical Arbitrage Strategy Documentation](statistical_arbitrage_strategy.md). 