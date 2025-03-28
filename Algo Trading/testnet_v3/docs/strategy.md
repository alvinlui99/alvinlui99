# Trading Strategy Documentation

## Overview

The trading system implements a LightGBM-based strategy that uses machine learning predictions to generate trading signals. The strategy is designed for cryptocurrency futures trading on Binance, supporting multiple trading pairs including BTC, ETH, BNB, ADA, SOL, DOGE, LINK, and AVAX.

## Strategy Components

### 1. Signal Generation

The strategy uses trained LightGBM models to predict price movements and generate trading signals based on the following logic:

- **Buy Signal**: Generated when predicted price change > threshold
- **Sell Signal**: Generated when predicted price change < -threshold (if shorts are allowed)
- **No Signal**: Generated when predicted price change is within threshold bounds

### 2. Position Sizing

Position sizing is managed through two methods:
- **Fixed Position Size**: Uses a predefined quantity for all trades
- **Percentage-based**: Calculates position size based on available capital and risk parameters

### 3. Risk Management

The strategy includes several risk management features:
- Configurable signal threshold
- Maximum allocation per trade
- Optional short position restrictions
- Minimum position size enforcement

## Configuration

Key strategy parameters can be configured in `config.py`:

```python
TradingConfig = {
    'SIGNAL_THRESHOLD': 0.02,        # Minimum prediction threshold for signals
    'MAX_ALLOCATION': 0.1,           # Maximum capital allocation per trade
    'ALLOW_SHORTS': True,            # Whether to allow short positions
    'DEFAULT_ORDER_TYPE': 'MARKET',  # Default order type
    'USE_FIXED_POSITION_SIZE': False,# Whether to use fixed position size
    'FIXED_POSITION_SIZE': 0.01      # Fixed position size if enabled
}
```

## Usage

### 1. Initialization

```python
from strategy.LGBMstrategy import LGBMstrategy

# Initialize strategy with a trained model
strategy = LGBMstrategy(
    model=trained_model,
    threshold=0.02,  # Optional: override default threshold
    logger=logger    # Optional: provide custom logger
)
```

### 2. Generating Signals

```python
# Generate trading signals from market data
signals = strategy.get_signals(klines_data)

# Example signal output
{
    'BTCUSDT': {
        'side': 'BUY',
        'type': 'MARKET',
        'quantity': 0.01,
        'price': 50000.0,
        'confidence': 0.025
    }
}
```

### 3. Parameter Adjustment

Strategy parameters can be adjusted dynamically:

```python
strategy.adjust_parameters(
    threshold=0.03,           # New signal threshold
    position_size_pct=0.15,   # New position size percentage
    allow_shorts=False        # Disable short positions
)
```

## Performance Considerations

1. **Signal Quality**:
   - Higher thresholds reduce false signals but may miss opportunities
   - Lower thresholds increase trading frequency but may lead to more false signals

2. **Position Sizing**:
   - Fixed position size is simpler but less flexible
   - Percentage-based sizing adapts to account balance but requires careful risk management

3. **Risk Management**:
   - Short positions increase trading opportunities but add complexity
   - Maximum allocation limits help prevent overexposure to single positions

## Best Practices

1. **Backtesting**:
   - Always backtest strategy changes before live trading
   - Use different market conditions to validate performance

2. **Monitoring**:
   - Track signal confidence levels
   - Monitor prediction accuracy
   - Watch for model drift

3. **Risk Management**:
   - Start with conservative position sizes
   - Implement stop-loss orders
   - Monitor overall portfolio exposure

## Next Steps

1. Implement backtesting framework
2. Add performance metrics tracking
3. Develop position management system
4. Create risk monitoring dashboard
5. Implement automated model retraining 