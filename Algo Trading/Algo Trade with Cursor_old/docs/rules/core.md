# Statistical Arbitrage Strategy Core Rules

## Overview
The statistical arbitrage strategy is designed to identify and exploit temporary price divergences between correlated assets. The strategy uses a combination of statistical measures and dynamic thresholds to determine entry and exit points.

## Key Components

### Spread Calculation
- The spread is calculated as the normalized deviation between two asset prices
- A multiplier (1000x) is applied to increase sensitivity to small price changes
- Spreads are normalized to percentage terms for consistent comparison

### Entry Conditions
- Minimum correlation: 0.7 between asset pairs
- Minimum spread standard deviation: 1% (normalized)
- Minimum spread value: 1% (normalized)
- Dynamic stop-loss and take-profit levels based on volatility

### Position Management
- Maximum position size: 10% of capital per trade
- Dynamic position sizing based on volatility
- Risk-adjusted stop-loss and take-profit levels

### Risk Management
- Base stop-loss: 2% of position value
- Base take-profit: 3% of position value
- Volatility-adjusted thresholds
- Maximum drawdown limits

## Performance Metrics
- Sharpe ratio
- Maximum drawdown
- Win rate
- Average profit per trade
- Risk-adjusted returns

## Market Conditions
- Works best in markets with:
  - High correlation between assets
  - Moderate volatility
  - Regular mean reversion patterns
  - Sufficient liquidity 