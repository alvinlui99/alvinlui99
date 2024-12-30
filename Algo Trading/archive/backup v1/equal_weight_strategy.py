from typing import Dict
from portfolio import Portfolio
import pandas as pd

def equal_weight_strategy(portfolio: Portfolio, current_prices: Dict[str, dict], timestep: int = 0) -> Dict[str, float]:
    """Simple equal weight strategy"""
    print("\n=== Equal Weight Strategy Calculation ===")
    n_assets = len(portfolio.symbols)
    target_weight = 1.0 / n_assets
    print(f"Number of assets: {n_assets}")
    print(f"Target weight per asset: {target_weight:.4f}")
    
    # Calculate current portfolio value
    current_equity = portfolio.cash
    current_weights = {}
    
    print("\nCurrent Portfolio State:")
    print(f"Initial Cash: ${portfolio.cash:.2f}")
    
    for symbol, row in portfolio.portfolio_df.iterrows():
        position = row['position']
        if isinstance(position, pd.Series):
            position = position.iloc[0]
        
        # Convert position and price to float
        position = float(position)
        price = float(current_prices[symbol]['markPrice'])
        position_value = position * price
        current_equity += position_value
        current_weights[symbol] = position_value / current_equity if current_equity > 0 else 0
        
        print(f"{symbol}:")
        print(f"  Position: {position:.6f}")
        print(f"  Price: ${price:.2f}")
        print(f"  Value: ${position_value:.2f}")
        print(f"  Current Weight: {current_weights[symbol]:.4f}")
    
    print(f"\nTotal Portfolio Value: ${current_equity:.2f}")
    
    # Calculate target positions
    print("\nCalculating Target Positions:")
    signals = {}
    for symbol in portfolio.symbols:
        price = float(current_prices[symbol]['markPrice'])
        target_value = target_weight * current_equity
        signals[symbol] = target_value / price
        print(f"{symbol}:")
        print(f"  Target Value: ${target_value:.2f}")
        print(f"  Price: ${price:.2f}")
        print(f"  Target Position: {signals[symbol]:.6f}")
    
    print("=== End Strategy Calculation ===\n")
    return signals 