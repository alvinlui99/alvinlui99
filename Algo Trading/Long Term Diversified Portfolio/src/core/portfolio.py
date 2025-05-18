"""
Portfolio management module for handling the long-term diversified portfolio.
Implements the 65/30/5 asset allocation strategy as specified in the IPS.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from .price_manager import PriceManager


@dataclass
class AssetAllocation:
    """Represents the target asset allocation for the portfolio."""
    us_equities: float = 0.65  # 65% US Equities
    fixed_income: float = 0.30  # 30% Fixed Income
    cash: float = 0.05  # 5% Cash


class Asset:
    """Base class for all assets"""
    def __init__(self, symbol: str, name: str, quantity: float = 0.0):
        self.symbol = symbol
        self.name = name
        self.quantity = quantity
    
    def get_market_value(self, price_manager: PriceManager) -> float:
        """
        Get current market value using the price manager.
        
        Args:
            price_manager (PriceManager): Price manager instance
            
        Returns:
            float: Current market value
        """
        current_price = price_manager.get_price(self.symbol)
        if current_price is None:
            raise ValueError(f"No price available for {self.symbol}")
        return current_price * self.quantity
    
    def get_risk_metrics(self, price_manager: PriceManager) -> Dict[str, float]:
        """Calculate risk metrics specific to this asset type"""
        raise NotImplementedError


class Equity(Asset):
    """Equity asset class"""
    def __init__(self, symbol: str, name: str, sector: str, market_cap: float, 
                 dividend_yield: float = 0.0, quantity: float = 0.0):
        super().__init__(symbol, name, quantity)
        self.sector = sector
        self.market_cap = market_cap
        self.dividend_yield = dividend_yield
    
    def get_risk_metrics(self, price_manager: PriceManager) -> Dict[str, float]:
        price_history = price_manager.get_price_history(self.symbol)
        if price_history is None or len(price_history) < 2:
            return {
                'beta': 0.0,
                'volatility': 0.0,
                'dividend_yield': self.dividend_yield
            }
        
        returns = price_history.pct_change().dropna()
        return {
            'beta': 0.0,  # To be implemented with market data
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'dividend_yield': self.dividend_yield
        }


class FixedIncome(Asset):
    """Fixed income asset class"""
    def __init__(self, symbol: str, name: str, coupon_rate: float, 
                 maturity_date: str, credit_rating: str, quantity: float = 0.0):
        super().__init__(symbol, name, quantity)
        self.coupon_rate = coupon_rate
        self.maturity_date = maturity_date
        self.credit_rating = credit_rating
    
    def get_risk_metrics(self, price_manager: PriceManager) -> Dict[str, float]:
        return {
            'duration': 0.0,  # To be implemented
            'yield_to_maturity': 0.0,  # To be implemented
            'credit_spread': 0.0  # To be implemented
        }


class Portfolio:
    """
    Main portfolio management class that handles:
    - Current portfolio state
    - Asset allocation tracking
    - Performance monitoring
    - Rebalancing calculations
    """

    def __init__(self, initial_capital: float):
        """
        Initialize the portfolio with initial capital.

        Args:
            initial_capital (float): Initial investment amount
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.target_allocation = AssetAllocation()
        self.price_manager = PriceManager()
        
        # Initialize portfolio holdings with specific asset types
        self.holdings: Dict[str, List[Asset]] = {
            'us_equities': [],
            'fixed_income': [],
            'cash': initial_capital
        }
        
        self.performance_history: List[Dict] = []

    def get_current_allocation(self) -> Dict[str, float]:
        """
        Calculate current asset allocation percentages.

        Returns:
            Dict[str, float]: Current allocation percentages for each asset class
        """
        total_value = self.get_total_value()
        if total_value == 0:
            return {k: 0.0 for k in self.holdings.keys()}
        
        allocations = {}
        for asset_class, assets in self.holdings.items():
            if asset_class == 'cash':
                allocations[asset_class] = assets / total_value
            else:
                allocations[asset_class] = sum(
                    asset.get_market_value(self.price_manager) 
                    for asset in assets
                ) / total_value
        
        return allocations

    def get_total_value(self) -> float:
        """
        Calculate total portfolio value.

        Returns:
            float: Total portfolio value
        """
        total = self.holdings['cash']
        for asset_class, assets in self.holdings.items():
            if asset_class != 'cash':
                total += sum(
                    asset.get_market_value(self.price_manager) 
                    for asset in assets
                )
        return total

    def calculate_rebalancing_needs(self) -> Dict[str, float]:
        """
        Calculate the amount needed to rebalance to target allocation.

        Returns:
            Dict[str, float]: Amount to add (positive) or subtract (negative) for each asset class
        """
        current_allocation = self.get_current_allocation()
        total_value = self.get_total_value()
        
        rebalancing_needs = {}
        for asset_class in self.holdings.keys():
            target_pct = getattr(self.target_allocation, asset_class)
            current_pct = current_allocation[asset_class]
            rebalancing_needs[asset_class] = total_value * (target_pct - current_pct)
            
        return rebalancing_needs

    def update_holdings(self, new_holdings: Dict[str, List[Asset]]) -> None:
        """
        Update portfolio holdings with new values.

        Args:
            new_holdings (Dict[str, List[Asset]]): New holdings for each asset class
        """
        for asset_class, assets in new_holdings.items():
            if asset_class == 'cash':
                self.holdings[asset_class] = assets
            else:
                self.holdings[asset_class] = assets
        
        self.current_capital = self.get_total_value()
        
        # Record performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'total_value': self.current_capital,
            'allocation': self.get_current_allocation()
        })

    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get historical performance data as a DataFrame.

        Returns:
            pd.DataFrame: Historical performance data
        """
        return pd.DataFrame(self.performance_history)


if __name__ == "__main__":
    # Example usage
    portfolio = Portfolio(initial_capital=100000.0)
    
    # Create some example assets
    aapl = Equity("AAPL", "Apple Inc.", "Technology", 2000000000000.0, 0.005)
    tlt = FixedIncome("TLT", "iShares 20+ Year Treasury Bond ETF", 0.02, "2025-12-31", "AAA")
    
    # Update prices
    portfolio.price_manager.update_price("AAPL", 150.0)
    portfolio.price_manager.update_price("TLT", 100.0)
    
    # Add assets to portfolio
    portfolio.update_holdings({
        'us_equities': [aapl],
        'fixed_income': [tlt],
        'cash': 5000.0
    })
    
    print("Current allocation:", portfolio.get_current_allocation())
    print("Rebalancing needs:", portfolio.calculate_rebalancing_needs()) 