from typing import TypedDict, Dict, List, NamedTuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pandas import DataFrame, Series
from decimal import Decimal

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
    """Expected format for equity curve data points"""
    timestamp: pd.Timestamp
    equity: float     # Total portfolio value

class BacktestResults(NamedTuple):
    """Container for backtest results"""
    stats: Dict[str, float]        # Performance metrics
    equity_curve: pd.DataFrame     # Portfolio value over time
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