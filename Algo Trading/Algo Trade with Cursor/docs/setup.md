# Setup Guide

## Prerequisites

1. Python 3.11
2. Binance account with API access
3. Basic understanding of pair trading

## Environment Setup

### 1. Python Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install python-binance pandas numpy scipy
```

### 2. Binance Account Setup
1. Create API keys with spot trading permissions
2. Enable 2FA
3. Set IP restrictions (recommended)

### 3. Project Configuration
1. Create `.env` file for API keys:
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## Project Structure

```
src/
  ├── data/
  │   ├── collector.py      # Data collection
  │   └── processor.py      # Data processing
  ├── strategy/
  │   ├── pair_selector.py  # Pair selection logic
  │   └── signals.py        # Trading signals
  ├── trading/
  │   ├── executor.py       # Order execution
  │   └── risk.py          # Risk management
  └── utils/
      ├── indicators.py     # Technical indicators
      └── logger.py         # Logging utilities
```

## Testing

1. **Backtesting**
   - Use historical data
   - Test z-score thresholds
   - Validate correlation analysis

2. **Paper Trading**
   - Test execution speed
   - Monitor slippage
   - Validate risk management

## Next Steps

1. Set up Python environment
2. Create Binance API keys
3. Set up project structure
4. Start with data collection 