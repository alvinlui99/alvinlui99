import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import deque
from typing import Optional, List, Dict

class Asset:
    """Represents a single trading asset with price and return tracking."""
    
    def __init__(self, symbol: str, maxlen: int = 250):
        """
        Initialize asset with symbol and rolling window size.
        
        Args:
            symbol: Trading pair symbol
            maxlen: Maximum length of rolling window for returns calculation (default: 250)
            
        Raises:
            ValueError: If symbol is empty or maxlen is less than 2
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if maxlen < 2:
            raise ValueError("maxlen must be at least 2")
            
        self.symbol = symbol
        self.prices = deque(maxlen=maxlen)
        self.ret_pct = deque(maxlen=maxlen)

    def update_price(self, price: float) -> None:
        """
        Update asset price and calculate returns if enough prices available.
        
        Args:
            price: New price value to add
            
        Raises:
            ValueError: If price is invalid (non-numeric or <= 0)
        """
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(f"Invalid price value: {price}")
        self.prices.append(float(price))
        if len(self.prices) >= 2:
            self._calculate_return()

    def _calculate_return(self) -> None:
        """Calculate and store return percentage if conditions are met."""
        if len(self.prices) < 2:
            return
        
        ret_pct = (self.prices[-1] - self.prices[0]) / self.prices[0] * 100
        if not self.ret_pct or self.ret_pct[-1] != ret_pct:
            self.ret_pct.append(ret_pct)

    def get_return(self) -> deque:
        """Return the deque of return percentages."""
        return self.ret_pct

    def get_latest_price(self) -> float:
        """Return most recent price or 0.0 if no prices available."""
        return self.prices[-1] if self.prices else 0.0

    def get_price_history(self) -> List[float]:
        """
        Get the price history as a list of float values.
        
        Returns:
            List of historical prices
        """
        return list(self.prices)

class Portfolio:
    """Manages a collection of trading assets with portfolio optimization."""
    
    def __init__(self, symbols: List[str], initial_capital: float):
        """
        Initialize portfolio with trading assets and initial capital.
        
        Args:
            symbols: List of trading symbols
            initial_capital: Starting cash amount
        """
        self.cash = initial_capital
        self.assets = {symbol: Asset(symbol) for symbol in symbols}
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.trade_history = []  # List to store trade records
        
    def get_position(self, symbol: str) -> float:
        """Safely get position for a symbol"""
        return self.positions.get(symbol, 0.0)
    
    def update_position(self, symbol: str, new_position: float) -> None:
        """
        Update position for a symbol and record the trade
        
        Args:
            symbol: Trading symbol
            new_position: New position size
        """
        old_position = self.positions.get(symbol, 0.0)
        self.positions[symbol] = float(new_position)
        
        # Record the trade
        trade_record = {
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'old_position': old_position,
            'new_position': new_position,
            'price': self.assets[symbol].get_latest_price(),
            'portfolio_value': self.get_total_value()
        }
        self.trade_history.append(trade_record)
    
    def update_price(self, symbol: str, price: float) -> None:
        """
        Update price for a symbol and associated Asset instance
        
        Args:
            symbol: Trading symbol
            price: Current price value
        """
        self.assets[symbol].update_price(float(price))
    
    def update_cash(self, amount: float) -> None:
        """
        Update portfolio cash balance
        
        Args:
            amount: Amount to add (positive) or subtract (negative)
        """
        self.cash += amount
    
    def get_total_value(self, current_prices: Optional[Dict] = None) -> float:
        """
        Calculate total portfolio value
        
        Args:
            current_prices: Optional dictionary of current prices, where each price can be
                          either a float or a dict with price information
            
        Returns:
            Total portfolio value including cash
        """
        total = self.cash
        
        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                price = float(current_prices[symbol])
            else:
                price = self.assets[symbol].get_latest_price()
            total += position * price
            
        return total
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get historical portfolio data as a DataFrame
        
        Returns:
            DataFrame containing trade history and portfolio values
        """
        return pd.DataFrame(self.trade_history)
    
    def get_asset_returns(self) -> Dict[str, List[float]]:
        """
        Get historical returns for all assets
        
        Returns:
            Dictionary mapping symbols to their return histories
        """
        return {symbol: list(asset.get_return()) 
                for symbol, asset in self.assets.items()}