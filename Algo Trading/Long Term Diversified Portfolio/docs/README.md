# Long Term Diversified Portfolio

This repository contains the implementation of a long-term diversified portfolio management system with algorithmic trading capabilities through Interactive Brokers API.

## Project Structure

```
├── docs/                      # Documentation
│   ├── IPS.md                # Investment Policy Statement
│   └── README.md             # Project documentation (this file)
│
├── src/                      # Source code
│   ├── config/              # Configuration files
│   │   ├── portfolio_config.py    # Portfolio allocation settings
│   │   └── api_config.py          # IBKR API configuration
│   │
│   ├── core/                # Core functionality
│   │   ├── portfolio.py     # Portfolio management logic
│   │   ├── rebalancer.py    # Rebalancing strategy implementation
│   │   └── risk_manager.py  # Risk management functions
│   │
│   ├── data/               # Data handling
│   │   ├── fetcher.py      # Market data retrieval
│   │   ├── processor.py    # Data processing utilities
│   │   └── storage.py      # Data storage management
│   │
│   ├── strategies/         # Trading strategies
│   │   ├── equity.py       # Equity allocation logic
│   │   ├── fixed_income.py # Fixed income management
│   │   └── cash.py         # Cash management
│   │
│   ├── visualization/      # Visualization modules
│   │   ├── performance.py  # Performance visualization
│   │   ├── allocation.py   # Asset allocation charts
│   │   └── risk.py         # Risk metrics visualization
│   │
│   └── utils/              # Utility functions
│       ├── logger.py       # Logging utilities
│       └── helpers.py      # Helper functions
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
│
├── notebooks/             # Jupyter notebooks for analysis
│   ├── portfolio_analysis.ipynb
│   └── backtesting.ipynb
│
├── data/                  # Data storage
│   ├── raw/              # Raw data from fetcher
│   │   ├── market_data/  # Market price data
│   │   ├── fundamental/  # Fundamental data
│   │   └── economic/     # Economic indicators
│   │
│   └── processed/        # Processed data ready for analysis
│       ├── daily/        # Daily processed data
│       └── monthly/      # Monthly aggregated data
│
├── output/               # Output storage
│   ├── reports/         # Generated reports
│   │   ├── monthly/     # Monthly performance reports
│   │   └── quarterly/   # Quarterly analysis reports
│   │
│   ├── visualizations/  # Generated charts and graphs
│   │   ├── performance/ # Performance charts
│   │   ├── allocation/  # Asset allocation charts
│   │   └── risk/       # Risk analysis charts
│   │
│   └── data/           # Processed data files
│       ├── csv/        # CSV exports
│       └── json/       # JSON data files
│
└── requirements.txt       # Python dependencies
```

## Key Features

- **Portfolio Management**: Implements the 65/30/5 asset allocation strategy as specified in the IPS
- **Automated Rebalancing**: Semi-annual rebalancing with tactical adjustments
- **Performance Tracking**: Comprehensive performance monitoring and visualization
- **Risk Management**: Continuous risk monitoring and management
- **Data Analysis**: Historical data analysis and backtesting capabilities
- **Visualization**: Rich set of visualization tools for portfolio analysis

## Getting Started

### Environment Setup

This project uses Conda for environment management. The environment is named `capital_mgmt` and uses Python 3.11.

1. **Activate the Conda environment**:
   ```bash
   conda activate capital_mgmt
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Development

- The project uses Python 3.11
- All dependencies are managed through Conda and pip
- Jupyter notebooks are configured to use the `capital_mgmt` kernel

## Data Management

The system maintains three levels of data:

1. **Raw Data** (`data/raw/`):
   - Market price data from various sources
   - Fundamental data for securities
   - Economic indicators and market data
   - Raw data is preserved in its original format

2. **Processed Data** (`data/processed/`):
   - Cleaned and normalized data
   - Daily and monthly aggregations
   - Calculated metrics and indicators
   - Ready for analysis and visualization

3. **Output Data** (`output/data/`):
   - Final processed data for reports
   - Exported data in various formats
   - Performance metrics and analysis results

## Output and Visualization

The system generates various outputs stored in the `output/` directory:

1. **Reports**:
   - Monthly performance reports
   - Quarterly analysis reports
   - Rebalancing reports

2. **Visualizations**:
   - Performance charts (returns, drawdowns)
   - Asset allocation charts
   - Risk metrics visualization
   - Correlation matrices
   - Factor analysis charts

3. **Data Exports**:
   - CSV files for portfolio holdings
   - JSON files for configuration and state
   - Performance metrics exports 