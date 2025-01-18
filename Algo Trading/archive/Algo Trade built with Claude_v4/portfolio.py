import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import deque
from typing import Optional, List, Dict

class Asset:
    """
    Represents an individual tradable asset in the portfolio.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.leverage = 0.0
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.used_margin = 0.0
        self.borrowed_margin = 0.0
        self.entry_price = 0.0

    def open_position(self, price: float, quantity: float, leverage: float) -> None:
        """
        Add a new position for the asset.
        """
        self.leverage = leverage
        self.position = quantity
        self.unrealized_pnl = 0.0
        self.borrowed_margin = quantity * price * (leverage - 1)
        self.used_margin = quantity * price
        self.entry_price = price

    def close_positions(self, current_price: float) -> float:
        """
        Close all positions and calculate realized P&L.
        """
        self.update_pnl(current_price)
        realized_pnl = self.unrealized_pnl
        self.leverage = 0.0
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.borrowed_margin = 0.0
        self.used_margin = 0.0
        self.entry_price = 0.0
        return realized_pnl

    def update_pnl(self, current_price: float) -> None:
        """
        Update the unrealized P&L based on the current price.
        """
        self.unrealized_pnl = self.position * (current_price - self.entry_price)
        self.used_margin = self.position * current_price

    def __str__(self):
        """
        String representation of the asset.
        """
        return f"Asset: {self.symbol}, Unrealized P&L: ${self.unrealized_pnl:.2f}, Positions: {self.position}"

class Portfolio:
    """Manages a collection of trading assets with portfolio optimization."""
    
    def __init__(self, symbols: List[str], initial_capital: float):
        """
        Initialize portfolio with trading assets and initial capital.
        
        Args:
            symbols: List of trading symbols
            initial_capital: Starting cash amount
        """
        self.assets = {symbol: Asset(symbol) for symbol in symbols}
        self.balance = initial_capital
        self.margin_balance = initial_capital
        self.borrowed_margin = 0.0
        self.used_margin = 0.0
        self.unrealized_pnl = 0.0
        self.leverage = 1.0
    
    def open_position(self, symbol: str, price: float, quantity: float, leverage: float) -> None:
        """
        Open a new position for a symbol
        
        Args:
            symbol: Trading symbol
            price: Current price value
            quantity: New position size
            leverage: Leverage for the position
        """
        self.leverage = leverage
        self.assets[symbol].open_position(price, quantity, leverage)
        self.update_portfolio_pnl({symbol: price})

    def close_position(self, symbol: str, price: float) -> None:
        """
        Close an existing position for a symbol
        """
        self.balance += self.assets[symbol].close_positions(price)
        self.update_portfolio_pnl({symbol: price})
        
    def update_position(self, symbol: str, price: float, quantity: float, leverage: float, close_position: bool = False) -> None:
        """
        Update the position for a symbol
        """
        if self.assets[symbol].position != 0:
            self.close_position(symbol, price)
        if not close_position:
            self.open_position(symbol, price, quantity, leverage)

    def update_portfolio_pnl(self, current_price: dict[str, float]) -> None:
        """
        Update the unrealized P&L based on the current price.
        """
        self.unrealized_pnl = 0.0
        self.borrowed_margin = 0.0
        self.margin_balance = self.balance
        self.used_margin = 0.0

        for symbol, price in current_price.items():
            self.assets[symbol].update_pnl(price)
        for symbol, asset in self.assets.items():
            self.unrealized_pnl += asset.unrealized_pnl
            self.borrowed_margin += asset.borrowed_margin
            self.margin_balance += asset.unrealized_pnl - asset.used_margin
            self.used_margin += asset.used_margin

    def get_total_value(self, current_price: dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            current_prices: Optional dictionary of current prices, where each price can be
                          either a float or a dict with price information
            
        Returns:
            
        """
        self.update_portfolio_pnl(current_price)
        return self.margin_balance + self.used_margin
    
    def get_composition(self, current_price: dict[str, float]) -> dict[str, dict[str, float]]:
        composition = {}
        for symbol, asset in self.assets.items():
            composition[symbol] = {
                'position': asset.position,
                'leverage': asset.leverage,
                'value': asset.position * current_price[symbol]
            }
        return composition