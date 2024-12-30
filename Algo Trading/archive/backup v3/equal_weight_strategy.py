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
        print("\n=== Equal Weight Strategy Calculation ===")
        n_assets = len(portfolio.symbols)
        target_weight = 1.0 / n_assets
        print(f"Number of assets: {n_assets}")
        print(f"Target weight per asset: {target_weight:.4f}")
        
        # Calculate current portfolio value and weights
        current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)
        
        print("\nCurrent Portfolio State:")
        print(f"Initial Cash: ${portfolio.cash:.2f}")
        
        # Print current positions and weights
        for symbol, weight in current_weights.items():
            position = portfolio.portfolio_df.loc[symbol, 'position']
            if isinstance(position, pd.Series):
                position = position.iloc[0]
            price = float(current_prices[symbol]['markPrice'])
            position_value = float(position) * price
            
            print(f"{symbol}:")
            print(f"  Position: {position:.6f}")
            print(f"  Price: ${price:.2f}")
            print(f"  Value: ${position_value:.2f}")
            print(f"  Current Weight: {weight:.4f}")
        
        print(f"\nTotal Portfolio Value: ${current_equity:.2f}")
        
        # Calculate target weights
        target_weights = {symbol: target_weight for symbol in portfolio.symbols}
        
        # Calculate and return target positions
        signals = self.calculate_positions(portfolio, current_prices, target_weights, current_equity)
        
        # Track portfolio state
        self.track_portfolio_state(portfolio, current_equity, timestep)
        
        print("=== End Strategy Calculation ===\n")
        return signals 