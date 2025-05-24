# Design Notes and Lessons Learned

## Position Management

### Leverage Management
- Leverage should be managed through the `set_leverage()` method rather than being stored in the positions dictionary
- This is because:
  1. Leverage is a trading parameter that should be explicitly set before placing orders
  2. The leverage value from position data might not reflect the current desired leverage for new orders
  3. It's better to have a single source of truth for leverage settings
  4. The `set_leverage()` method provides proper error handling and logging

### Margin Type Management
- Margin type should be managed through dedicated `set_margin_type()` and `get_margin_type()` methods
- Two types of margin available:
  1. ISOLATED: Each position has its own margin requirement
  2. CROSS: All positions share the same margin pool
- Margin type should be set before placing orders
- Default to ISOLATED margin for safety
- Margin type is a trading parameter and should not be stored in positions dictionary

### Position Tracking
- The positions dictionary should only track actual position data:
  - Position size
  - Entry price
  - Unrealized PnL
- Trading parameters like leverage and margin type should be managed separately through dedicated methods

## Data Handling Conventions

### Column Naming Conventions
- Binance Futures API returns data with capitalized column names: 'Open', 'High', 'Low', 'Close', 'Volume'
- These differ from common pandas conventions which often use lowercase ('open', 'high', 'low', 'close', 'volume')
- Our code should handle both conventions by:
  1. Consistently using the same convention within our code (prefer the Binance convention)
  2. Adding conversion logic where needed when interfacing with external libraries
  3. Being careful with case sensitivity in column references
- Always check the actual column names in the data frame before processing

## Best Practices
1. Always use dedicated methods for changing trading parameters (leverage, margin type, etc.)
2. Keep position data focused on actual position information
3. Log all parameter changes for better debugging and monitoring
4. Implement proper error handling for parameter changes
5. Default to safer options (e.g., ISOLATED margin) when in doubt
6. Validate actual column names in data frames before processing