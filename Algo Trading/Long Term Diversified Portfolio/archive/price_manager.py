"""
Price management module for handling asset prices.
"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd


class PriceManager:
    """
    Manages current and historical prices for all assets.
    Can be extended to fetch real-time prices from various sources.
    """
    
    def __init__(self):
        self._prices: Dict[str, float] = {}  # Current prices
        self._price_history: Dict[str, pd.Series] = {}  # Historical prices
        self._last_update: Dict[str, datetime] = {}  # Last price update timestamp
    
    def update_price(self, symbol: str, price: float) -> None:
        """
        Update the current price for an asset.
        
        Args:
            symbol (str): Asset symbol
            price (float): Current price
        """
        self._prices[symbol] = price
        self._last_update[symbol] = datetime.now()
        
        # Update price history
        if symbol not in self._price_history:
            self._price_history[symbol] = pd.Series()
        self._price_history[symbol][datetime.now()] = price
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for an asset.
        
        Args:
            symbol (str): Asset symbol
            
        Returns:
            Optional[float]: Current price if available, None otherwise
        """
        return self._prices.get(symbol)
    
    def get_price_history(self, symbol: str) -> Optional[pd.Series]:
        """
        Get historical prices for an asset.
        
        Args:
            symbol (str): Asset symbol
            
        Returns:
            Optional[pd.Series]: Historical prices if available, None otherwise
        """
        return self._price_history.get(symbol)
    
    def get_last_update(self, symbol: str) -> Optional[datetime]:
        """
        Get the last update timestamp for an asset's price.
        
        Args:
            symbol (str): Asset symbol
            
        Returns:
            Optional[datetime]: Last update timestamp if available, None otherwise
        """
        return self._last_update.get(symbol) 