# Long Term Diversified Portfolio

A Python-based portfolio management system that helps create and analyze diversified long-term investment portfolios.

## Project Structure

```
├── src/
│   ├── core/               # Core portfolio management logic
│   ├── data/              # Data handling and processing
│   ├── utils/             # Utility functions
│   ├── visualization/     # Data visualization modules
│   ├── strategies/        # Investment strategy implementations
│   └── config/            # Configuration files
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── output/               # Generated reports and outputs
├── data/                 # Data storage
└── requirements.txt      # Project dependencies
```

## Market Data Integration

This project uses Yahoo Finance (yfinance) for market data retrieval. The integration provides:

1. Historical price data
2. Company financial information
3. Dividend data
4. Market statistics

### Data Retrieval

The `Portfolio` class handles all interactions with yfinance. It provides methods for:

- Fetching historical data for individual symbols
- Retrieving data for multiple symbols simultaneously
- Converting API responses to pandas DataFrames for analysis
- Calculating returns and volatility metrics

## Portfolio Management

The portfolio management system provides:

1. Asset allocation across:
   - Equities (65%)
   - Fixed Income (30%)
   - Cash (5%)

2. Sector diversification within equities:
   - Technology (28%)
   - Healthcare (13%)
   - Financials (12%)
   - Consumer Discretionary (10%)
   - And more...

3. Risk metrics calculation:
   - Expected returns
   - Volatility
   - Sharpe ratio
   - Maximum drawdown

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the portfolio analysis:
   ```python
   from src.core.portfolio import Portfolio
   
   portfolio = Portfolio(initial_capital=1_000_000)
   holdings = portfolio.get_target_holdings()
   metrics = portfolio.calculate_portfolio_metrics(holdings)
   ```

3. Export results:
   ```python
   portfolio.export_to_csv(holdings, metrics)
   ```