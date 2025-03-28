# Crypto Trading Bot

## Project Structure
```
├── src/
│   ├── data/           # Market data collection
│   │   ├── market_data.py    # Binance Futures data fetching
│   │   └── data_cache.py     # Data caching mechanism
│   ├── trading/        # Trading bot implementation
│   │   └── trading_bot.py    # Core trading logic
│   └── strategy/       # Trading strategies
│       ├── base_strategy.py  # Base strategy class
│       ├── indicators.py     # Technical indicators
│       ├── macd_strategy.py  # MACD-based strategy
│       └── backtest.py       # Backtesting framework
├── tests/              # Test scripts
│   ├── test_gpu.py     # GPU availability check
│   ├── test_orders.py  # Order placement testing
│   ├── test_backtest.py # Basic strategy backtesting
│   └── backtest_visualization.py # Strategy backtesting with visualization
├── docs/              # Documentation
├── data/              # Cached market data
└── config/            # Configuration files
```

## Current Status
- [x] Basic project structure
- [x] Market data collection from Binance Futures
- [x] Market data caching for faster backtesting
- [x] Testnet integration
- [x] Basic order placement with leverage
- [x] Position tracking
- [x] Balance monitoring
- [x] Technical indicators implementation
- [x] Base strategy framework
- [x] MACD strategy implementation
- [x] Backtesting framework
- [x] Backtest visualization
- [ ] Portfolio optimization
- [ ] Machine learning integration
- [ ] Risk management
- [ ] Performance monitoring

## Development Roadmap

### Phase 1: Core Trading Infrastructure (Current)
- [x] Market data collection
- [x] Basic order placement
- [x] Position management
- [x] Balance tracking
- [x] Leverage management
- [ ] Order history
- [ ] Position sizing
- [ ] Risk limits

### Phase 2: Strategy Development (Current)
- [x] Technical indicators
- [x] Base strategy framework
- [x] MACD strategy
- [x] Backtesting framework
- [x] Backtest visualization
- [ ] Strategy optimization
- [ ] Multiple timeframe analysis
- [ ] Market regime detection
- [ ] Strategy performance metrics

### Phase 3: Portfolio Management
- [ ] Correlation analysis between assets
- [ ] Position sizing optimization
- [ ] Portfolio rebalancing
- [ ] Risk allocation
- [ ] Drawdown protection

### Phase 4: Machine Learning Integration
- [ ] Feature engineering
- [ ] Price prediction models
- [ ] Market regime detection
- [ ] Sentiment analysis
- [ ] Model performance tracking

### Phase 5: Risk Management
- [ ] Stop-loss automation
- [ ] Take-profit strategies
- [ ] Position sizing rules
- [ ] Leverage management
- [ ] Risk metrics calculation

### Phase 6: Performance Optimization
- [ ] Performance metrics
- [ ] Strategy optimization
- [ ] Real-time monitoring
- [ ] Alert system

## Using the Data Cache

The bot includes a data caching mechanism to avoid repeated API calls when running backtests. This is especially useful during strategy development when you need to run multiple backtests on the same data.

Key features:
- Automatically caches market data to the `data/` directory
- Reuses cached data when available instead of making API calls
- Cache files are named with symbol, timeframe, and date range for easy identification
- Includes metadata about when the data was cached

To use the cache:
```python
# Create market data object with cache enabled
market_data = MarketData(
    client=client,
    symbols=['BTCUSDT'],
    start_date='2023-01-01',
    end_date='2023-03-01',
    timeframe='1h',
    use_cache=True  # Enable caching
)

# This will use cache if available, otherwise fetch from API
historical_data = market_data.fetch_historical_data()
``` 