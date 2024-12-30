from typing import Dict
from portfolio import Portfolio
import numpy as np

def threshold_rebalance_strategy(portfolio: Portfolio, current_prices: Dict[str, dict], 
                               timestep: int = 0,  # Used for initial allocation
                               rebalance_threshold: float = 0.1  # 10% threshold
                               ) -> Dict[str, float]:
    """
    Minimum variance strategy that rebalances when weights deviate beyond threshold
    """
    signals = {}
    is_initial_allocation = (timestep == 0)
    
    # Calculate current equity and weights
    current_equity = portfolio.cash
    current_weights = {}
    
    for symbol, row in portfolio.portfolio_df.iterrows():
        position_value = row['position'] * current_prices[symbol]['markPrice']
        current_equity += position_value
        current_weights[symbol] = position_value / current_equity if current_equity > 0 else 0
    
    # Check if rebalancing is needed
    should_rebalance = is_initial_allocation
    
    if not is_initial_allocation and rebalance_threshold > 0:
        target_weight = 1.0 / len(portfolio.symbols)
        max_deviation = max(abs(weight - target_weight) for weight in current_weights.values())
        should_rebalance = max_deviation > rebalance_threshold
    
    if should_rebalance:
        result = portfolio.get_optim_weights()
        
        if result is not None:
            optimal_weights = result.x
            max_position_pct = 0.3  # Maximum 30% of portfolio in any single asset
            
            for i, symbol in enumerate(portfolio.symbols):
                price = current_prices[symbol]['markPrice']
                limited_weight = min(optimal_weights[i], max_position_pct)
                target_position = (limited_weight * current_equity) / price
                current_position = portfolio.portfolio_df.loc[symbol, 'position']
                
                if not is_initial_allocation:
                    position_change = abs(target_position - current_position)
                    max_change = abs(current_position * 0.20)
                    
                    if position_change > max_change:
                        if target_position > current_position:
                            target_position = current_position + max_change
                        else:
                            target_position = current_position - max_change
                
                signals[symbol] = target_position
        else:
            # Fallback to equal weights if optimization fails
            weight = 1.0 / len(portfolio.symbols)
            for symbol in portfolio.symbols:
                price = current_prices[symbol]['markPrice']
                signals[symbol] = (weight * current_equity) / price
    else:
        # Maintain current positions if no rebalancing is needed
        for symbol in portfolio.symbols:
            signals[symbol] = portfolio.portfolio_df.loc[symbol, 'position']
    
    return signals 