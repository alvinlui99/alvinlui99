from typing import Dict
from portfolio import Portfolio
import pandas as pd
from strategy import Strategy

class EqualWeightStrategy(Strategy):
    """Simple equal weight strategy"""
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], timestep: int = 0) -> Dict[str, float]:
        """Execute equal weight strategy"""
        n_assets = len(portfolio.symbols)
        target_weight = 1.0 / n_assets
        
        # Calculate current portfolio value and weights
        current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)
        
        # Calculate target weights
        target_weights = {symbol: target_weight for symbol in portfolio.symbols}
        
        # Calculate and return target positions
        signals = self.calculate_positions(portfolio, current_prices, target_weights, current_equity)
        
        # Track portfolio state
        self.track_portfolio_state(portfolio, current_equity, timestep)
        
        return signals 