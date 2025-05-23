# Long Term Diversified Portfolio

A Python-based portfolio management system designed for long-term, diversified investing. The system implements a strategic asset allocation approach with sector-specific optimizations and comprehensive backtesting capabilities.

## System Workflow

### 1. Portfolio Optimization
- Uses historical data to optimize sector weights
- Selects stocks based on scoring criteria
- Determines initial portfolio composition
- Outputs optimized holdings

### 2. Portfolio Initialization
- Gets exact prices as of start date
- Calculates position sizes
- Sets up initial portfolio
- Records starting positions

### 3. Backtesting Process
- Tracks daily position values
- Updates portfolio metrics
- Monitors performance
- Records trade history

## System Components

### Portfolio Management
- Asset allocation optimization
- Stock selection and scoring
- Portfolio construction
- Performance calculation
- Risk metrics computation

### Backtesting Framework
- Portfolio initialization with exact prices
- Daily position value tracking
- Performance metrics calculation
- Sector performance analysis
- Comprehensive reporting

## Performance Metrics

### Return Metrics
- Total Return
- Annualized Return
- Monthly Returns
- Sector Returns

### Risk Metrics
- Sharpe Ratio
- Maximum Drawdown
- Volatility
- Sector Correlations

## Output Reports

### Position Reports
- Starting positions with exact prices
- Ending positions with performance
- Sector allocation breakdown

### Performance Reports
- Portfolio value vs S&P 500
- Monthly returns heatmap
- Drawdown analysis
- Returns distribution
- Sector performance
- Individual stock performance

## Future Enhancements

1. Semi-annual rebalancing
2. Tax-loss harvesting
3. Factor analysis
4. Risk parity optimization
5. Dynamic asset allocation

## Implementation Notes

- Uses yfinance for market data
- All prices are adjusted for splits and dividends
- Commission and slippage are not modeled
- Assumes perfect execution of trades