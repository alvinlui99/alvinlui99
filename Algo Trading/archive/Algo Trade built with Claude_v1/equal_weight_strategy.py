from typing import Dict
from portfolio import Portfolio
import pandas as pd
from strategy import Strategy

class EqualWeightStrategy(Strategy):
    """Simple equal weight strategy"""
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, float], 
                 test_data: Dict[str, pd.DataFrame], timestep: int = 0) -> Dict[str, float]:
        """
        Execute equal weight strategy
        
        Args:
            portfolio: Portfolio object containing current positions
            current_prices: Dictionary mapping symbols to their current prices
            test_data: Full test dataset (not used in equal weight strategy)
            timestep: Current timestep in the simulation
            
        Returns:
            Dictionary mapping symbols to target positions
        """
        # Calculate equal target weights
        n_assets = len(portfolio.assets)
        target_weight = 1.0 / n_assets
        
        # Calculate current portfolio value and adjust for commissions
        total_equity = portfolio.get_total_value(current_prices)
        adjusted_equity = self.get_commission_adjusted_equity(total_equity)
        
        # Calculate target positions based on equal weights
        target_positions = {}
        for symbol in portfolio.assets.keys():
            price = current_prices[symbol]
            target_positions[symbol] = (target_weight * adjusted_equity) / price
                
        return target_positions 