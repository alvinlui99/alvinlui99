from typing import Dict
from portfolio import Portfolio
import numpy as np
from config import (REBALANCE_INTERVAL, REBALANCE_THRESHOLD,
                   MAX_POSITION_PCT, MAX_POSITION_CHANGE)

def min_variance_strategy(portfolio: Portfolio, current_prices: Dict[str, dict], 
                         timestep: int = 0,
                         rebalance_threshold: float = REBALANCE_THRESHOLD,
                         rebalance_interval: int = REBALANCE_INTERVAL
                         ) -> Dict[str, float]:
    """
    Minimum variance strategy with both threshold and periodic rebalancing
    """
    signals = {}
    current_equity = portfolio.cash
    current_weights = {}
    
    # Calculate current equity and weights
    for symbol, row in portfolio.portfolio_df.iterrows():
        position_value = row['position'] * current_prices[symbol]['markPrice'] 
        current_equity += position_value
        current_weights[symbol] = position_value / current_equity if current_equity > 0 else 0
    
    # Check if rebalancing is needed based on threshold or interval
    should_rebalance = False
    if rebalance_threshold > 0:
        target_weight = 1.0 / len(portfolio.symbols)
        max_deviation = max(abs(weight - target_weight) for weight in current_weights.values())
        should_rebalance = max_deviation > rebalance_threshold
    
    if rebalance_interval > 0 and timestep % rebalance_interval == 0:
        should_rebalance = True
    
    if should_rebalance:
        result = portfolio.get_optim_weights()
        
        if result is not None:
            optimal_weights = result.x
            
            for i, symbol in enumerate(portfolio.symbols):
                price = current_prices[symbol]['markPrice']
                limited_weight = min(optimal_weights[i], MAX_POSITION_PCT)
                target_position = (limited_weight * current_equity) / price
                current_position = portfolio.portfolio_df.loc[symbol, 'position']
                
                # Gradual position changes
                position_change = abs(target_position - current_position)
                max_change = abs(current_position * MAX_POSITION_CHANGE)
                
                if position_change > max_change:
                    if target_position > current_position:
                        target_position = current_position + max_change
                    else:
                        target_position = current_position - max_change
                
                signals[symbol] = target_position
        else:
            # Fallback to equal weights
            weight = 1.0 / len(portfolio.symbols)
            for symbol in portfolio.symbols:
                price = current_prices[symbol]['markPrice']
                signals[symbol] = (weight * current_equity) / price
    else:
        # Maintain current positions
        for symbol in portfolio.symbols:
            signals[symbol] = portfolio.portfolio_df.loc[symbol, 'position']
    
    return signals