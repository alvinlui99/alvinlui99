## TODO
- Recalculate target risk on rolling window basis (in model.py: calculate_target_risk method)

## Development Roadmap

### Trading Cycle Implementation

1. **Leverage Regime** (Priority)
  - add more indicators
  - _calculate_signals
    currently the signals are too simple, need to add more complex logic
    instead of using int, the signals can be a float, which allows for more nuanced decision making
  - becareful of collinearity

2. **Housekeeping**
  - Organize the config files

## REFERENCES
- https://developers.binance.com/docs/derivatives/Introduction