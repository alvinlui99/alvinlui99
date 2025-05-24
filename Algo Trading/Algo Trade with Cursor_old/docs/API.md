# API Documentation

## Market Data Class
```python
class MarketData:
    def __init__(self, client: UMFutures, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None)
    def fetch_historical_data(self) -> dict
    def get_latest_price(self, symbol: str) -> float
    def get_volume(self, symbol: str) -> float
```

## Trading Bot Class
```python
class TradingBot:
    def __init__(self, trading_pairs: List[str], initial_capital: float)
    def get_account_balance(self) -> float
    def place_order(self, symbol: str, side: str, quantity: float) -> Dict
    def run(self) -> None
```

## Configuration
Trading parameters and settings are defined in `config/trading_config.py`:
- Trading pairs
- Position sizes
- Risk management parameters
- ML model parameters

## Environment Variables
Required environment variables in `.env`:
- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret 