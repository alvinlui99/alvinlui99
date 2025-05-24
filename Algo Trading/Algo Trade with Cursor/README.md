# Binance Crypto Trading Bot

This project implements an automated trading system for cryptocurrencies on the Binance exchange.

## Project Structure

- `docs/` - Documentation files
  - `setup.md` - Setup and configuration guide
  - `strategy.md` - Trading strategy documentation
  - `api.md` - Binance API integration details
  - `risk.md` - Risk management guidelines
- `src/` - Source code directory
- `tests/` - Test files
- `config/` - Configuration files

## Getting Started

1. Read through the documentation in the `docs/` directory
2. Set up your Binance API credentials
3. Configure your trading parameters
4. Run the trading bot

## Important Notes

- This is a work in progress
- Always test with small amounts first
- Never share your API keys
- Monitor the bot regularly

## Road Map

First Phase - Data Collection & Basic Processing
Start with src/data/collector.py to fetch data from Binance
Then src/data/processor.py to calculate return variance and z score
This gives us the foundation to validate our data and calculations

Second Phase - Pair Analysis
Implement src/utils/indicators.py for z-score and correlation calculations
Then src/strategy/pair_selector.py to identify trading pairs
This lets us validate our pair selection logic

Third Phase - Basic Backtesting
Create src/backtesting/backtest.py for simple strategy testing
Add src/backtesting/performance.py for basic metrics
This allows us to test our strategy without real money

Fourth Phase - Trading Logic
Implement src/strategy/signals.py for entry/exit signals
Then src/trading/risk.py for position sizing
Finally src/trading/executor.py for order execution