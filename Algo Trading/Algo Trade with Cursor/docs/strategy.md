# Pair Trading Strategy

## Overview
This is a statistical arbitrage strategy based on pairs trading, using cointegration and mean reversion principles. The strategy identifies trading opportunities when two correlated assets temporarily diverge from their historical relationship.

## Signal Generation Logic

### Key Parameters
- `window`: 240 periods (default) for spread calculation
- `zscore_threshold`: 3.0 (default) for entry signals
- `adf_entry_threshold`: 0.05 for cointegration testing
- `adf_exit_threshold`: 0.10 for position exit
- `hedge_window`: 240 periods for hedge ratio calculation
- `long_term_window`: 720 periods for volatility calculation
- `short_term_window`: 240 periods for volatility calculation

### Signal Generation Process

1. **Hedge Ratio Calculation**
   - Uses rolling linear regression on log prices
   - Calculated over `hedge_window` periods
   - Updates continuously as new data arrives

2. **Spread Calculation**
   - Spread = log(price1) - hedge_ratio * log(price2)
   - Maintains a rolling window of 800 periods

3. **Dynamic Threshold Adjustment**
   - Base threshold: 3.0
   - Adjusted based on volatility ratio (long-term vs short-term)
   - Bounded between 2.5 and 4.0
   - Higher volatility → higher threshold
   - Lower volatility → lower threshold

4. **Entry Conditions**
   - Must be out of position
   - ADF p-value ≤ 0.05 (cointegration test)
   - Z-score crosses dynamic threshold:
     - Long: Z-score ≤ -threshold
     - Short: Z-score ≥ threshold

5. **Exit Conditions**
   - Z-score crosses mean (0)
   - ADF p-value ≥ 0.10
   - Position type:
     - Long spread: Exit when Z-score ≥ 0
     - Short spread: Exit when Z-score ≤ 0

### Position Management
- Signal values:
  - 1: Long spread position
  - -1: Short spread position
  - 0: No position

### Risk Management
- Dynamic threshold adjustment based on market volatility
- Cointegration testing for pair selection
- Position exit on mean reversion or deteriorating cointegration

## Performance Metrics
The strategy tracks:
- Total signals generated
- Long vs short signal distribution
- Average ADF p-values
- Average Z-scores
- Signal clustering analysis
