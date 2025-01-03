from typing import TypedDict, Dict, List, NamedTuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pandas import DataFrame, Series
from decimal import Decimal
from datetime import datetime

class TradeRecord(TypedDict):
    """Expected format for individual trade records"""
    timestamp: pd.Timestamp
    symbol: str
    size: float        # Position change amount
    price: float       # Execution price
    commission: float  # Trading fee
    trade_cost: float  # Total cost of trade
    type: str          # 'execution' or 'placement'

class EquityPoint(TypedDict):
    """Represents a single point in the equity curve tracking portfolio value over time.
    
    This type is used to structure the data points that make up the equity curve
    in the backtesting system. The equity curve is typically stored as a pandas DataFrame
    with a DatetimeIndex and a single 'equity' column.
    
    Attributes:
        timestamp (pd.Timestamp): The point in time for this equity measurement
        equity (float): The total portfolio value at this timestamp
        
    Example DataFrame Structure:
                            equity
    2024-01-01 00:00:00    10000.0
    2024-01-01 00:01:00    10002.5
    2024-01-01 00:02:00    10001.8
    """
    timestamp: pd.Timestamp
    equity: float

class BacktestResults(NamedTuple):
    """Container for backtest results including the equity curve.
    
    Attributes:
        stats (Dict[str, float]): Performance metrics like returns, Sharpe ratio, etc.
        equity_curve (pd.DataFrame): Portfolio value over time, structured as:
            - Index: pd.DatetimeIndex (timestamps)
            - Column: 'equity' (float) representing total portfolio value
        trade_history (List[TradeRecord]): List of all executed trades
    """
    stats: Dict[str, float]
    equity_curve: pd.DataFrame
    trade_history: List['TradeRecord']

class RegimeStats(TypedDict):
    """Expected format for regime statistics"""
    regime: str           # 'Bull', 'Neutral', or 'Bear'
    pct_time: float      # Percentage of time in regime
    avg_duration: float   # Average duration in periods
    transitions: Dict[str, float]  # Transition probabilities to other regimes 

class HistoricalData(TypedDict):
    """Expected format for historical price data DataFrame
    
    DataFrame with MultiIndex columns:
        Level 0: Symbol (e.g. 'BTCUSDT', 'ETHUSDT')
        Level 1: Feature (e.g. 'price', 'volume', 'return')
        
    Index: pd.DatetimeIndex
        Timestamp of each data point
        
    Required columns per symbol:
        {symbol}_price: float    # Close/Mark price
        {symbol}_volume: float   # Trading volume
        {symbol}_return: float   # Price returns
        
    Example:
        timestamp | BTCUSDT_price | BTCUSDT_volume | BTCUSDT_return | ETHUSDT_price ...
        2020-01-01|    9000.0    |    1000.0      |     0.02       |    200.0    ...
        2020-01-02|    9100.0    |    1100.0      |     0.011      |    205.0    ...
    """
    index: pd.DatetimeIndex
    columns: List[str]  # Format: "{symbol}_{feature}"
    values: np.ndarray  # 2D array of float values

@dataclass
class MarketData:
    """Container for different timeframes of market data"""
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    full_data: pd.DataFrame

class PriceBar(TypedDict):
    """Individual price bar data structure"""
    datetime: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    vwap: float 

class Asset:
    """Type stub for Asset class."""
    pass

class PortfolioRow(TypedDict):
    """Represents a row in the portfolio DataFrame."""
    size: Union[float, Decimal]  # Position size of the asset
    asset: Asset                 # Asset object containing price history and returns
    price: float                 # Current market price
    weight: float                # Portfolio allocation weight (0.0 to 1.0)

class PortfolioDataFrame:
    """
    Type hints for the portfolio DataFrame structure.
    
    DataFrame Structure:
        Index: Trading symbols (str) - e.g., 'BTC/USD', 'ETH/USD'
        Columns:
            - size (float): Current position size
            - asset (Asset): Asset object tracking prices and returns
            - price (float): Current market price
            - weight (float): Portfolio allocation weight
            
    Example:
              size    asset    price    weight
    BTC/USD   0.5   <Asset>   50000     0.6
    ETH/USD   2.0   <Asset>   3000      0.4
    """
    def __init__(self):
        self.df: DataFrame[PortfolioRow] = DataFrame()
        
    @property
    def size(self) -> Series:
        """Position sizes for all assets."""
        return self.df['size']
        
    @property
    def price(self) -> Series:
        """Current prices for all assets."""
        return self.df['price']
        
    @property
    def weight(self) -> Series:
        """Portfolio weights for all assets."""
        return self.df['weight'] 

class MarketDataFeed(TypedDict):
    """Represents a single market data update for a trading instrument"""
    symbol: str
    markPrice: str           # Current market price with 8 decimal precision
    indexPrice: str          # Index price with 8 decimal precision
    estimatedSettlePrice: str  # Estimated settlement price with 8 decimal precision
    lastFundingRate: str     # Last funding rate, e.g., '0.00010000'
    interestRate: str        # Interest rate, e.g., '0.00010000'
    nextFundingTime: int     # Unix timestamp in milliseconds for next funding
    time: int               # Current unix timestamp in milliseconds

# Type hints for nested dictionary structure
TimeSeriesMarketData = Dict[datetime, Dict[str, MarketDataFeed]]

class MarketDataStructure:
    """Documents the structure of market data feed used in backtesting
    
    The market data is organized in two main formats:
    1. DataFrame format (price_history):
       - Columns are named as '{symbol}_price' (e.g., 'BTC_price', 'ETH_price')
       - Index is timestamp
       - Values are float prices
    
    2. Dictionary format (market_data):
       - First level key: timestamp (datetime)
       - Second level key: symbol (str)
       - Value: MarketDataFeed containing price and funding information
    
    Example:
    ```
    # DataFrame format (self.price_history):
    timestamp            | BTC_price | ETH_price | ...
    2023-01-01 00:00:00 | 50000.00  | 3000.00   | ...
    2023-01-01 00:01:00 | 50050.00  | 3010.00   | ...
    
    # Dictionary format (market_data):
    {
        datetime(2023, 1, 1, 0, 0): {
            'BTC': {
                'symbol': 'BTC',
                'markPrice': '50000.00000000',
                'indexPrice': '50000.00000000',
                'estimatedSettlePrice': '50000.00000000',
                'lastFundingRate': '0.00010000',
                'interestRate': '0.00010000',
                'nextFundingTime': 1672531200000,
                'time': 1672527600000
            },
            'ETH': {
                # Similar structure for ETH
            }
        },
        # More timestamps...
    }
    ```
    """
    pass 