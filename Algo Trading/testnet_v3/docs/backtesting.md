# Backtesting Documentation

## Overview

The backtesting system evaluates trading strategies using historical data before deploying them in real-world trading. This helps assess strategy viability and performance.

## Components

### 1. Backtest Engine (`scripts/backtesting/backtest_engine.py`)
- Processes historical data
- Executes trades based on strategy signals
- Tracks positions and equity
- Calculates performance metrics

### 2. Main Script (`scripts/backtesting/run_backtest.py`)
- Loads backtest data and trained models
- Initializes the strategy
- Runs the backtest
- Saves detailed results

## Data Source

The backtesting system uses reserved test data stored in `data/backtest/`. This data was separated during the model training process and represents 20% of the most recent data.

## How to Run

1. Ensure you have:
   - Trained models in `model/trained_models/`
   - Backtest data in `data/backtest/`
   - Required dependencies installed

2. Run the backtest:
   ```bash
   python scripts/backtesting/run_backtest.py
   ```

3. Results will be saved in `data/backtest_results/`:
   - `trades.csv`: Individual trades
   - `equity_curve.csv`: Portfolio value over time
   - `metrics.csv`: Performance metrics

## Performance Metrics

The backtesting system calculates and reports:
- Total return
- Annualized return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades
- Final equity

## Best Practices

1. Always backtest with sufficient historical data
2. Use realistic initial capital amounts
3. Consider transaction costs and slippage
4. Monitor position sizing and risk management
5. Review detailed trade logs for strategy improvement 