# Market Neutral Crypto Trading Strategy

A machine learning-based market neutral trading strategy for cryptocurrency pairs trading.

## Overview

This project implements a market neutral trading strategy using LSTM neural networks to predict optimal portfolio weights. The strategy aims to maintain market neutrality while capturing relative value opportunities between different cryptocurrency pairs.

## Features

- LSTM-based weight prediction
- Portfolio optimization with risk constraints
- Automated backtesting framework
- Real-time trading capabilities
- Performance visualization and analysis

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure settings in `config.py`
2. Run backtesting:
```bash
python main.py
```

## Results

The strategy performance metrics include:
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

## Potential Improvements

### 1. Market Regime Detection
The model could be enhanced by incorporating market regime detection:
- Use rolling window analysis of returns and volatility
- Identify different market states:
  * Bullish (high returns, low volatility)
  * Bearish (negative returns, high volatility)
  * Ranging (low returns, low volatility)
  * Crisis (negative returns, extreme volatility)
- Adjust strategy parameters based on regime:
  * Position sizing
  * Rebalancing frequency
  * Risk limits
- Indicators for regime detection:
  * Trend strength (ADX)
  * Volatility regimes (ATR)
  * Market sentiment (RSI, MACD)
  * Volume analysis

### 2. Cross-Asset Signals
Improve prediction accuracy by considering relationships between assets:
- Correlation-based features:
  * Rolling correlations between pairs
  * Correlation regime changes
  * Lead-lag relationships
- Market structure analysis:
  * Common factor exposures
  * Sector relationships
  * Market dominance metrics
- Volume relationships:
  * Relative volume changes
  * Cross-pair volume signals
  * Order book dynamics

### 3. Future Enhancements
Other areas for improvement:
- Dynamic feature selection
- Adaptive hyperparameter tuning
- Alternative neural network architectures
- Advanced risk management techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 