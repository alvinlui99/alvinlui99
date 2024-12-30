from typing import Dict
from portfolio import Portfolio
import numpy as np
from config import REBALANCE_INTERVAL, MAX_POSITION_PCT, MAX_POSITION_CHANGE

def periodic_rebalance_strategy(portfolio: Portfolio, current_prices: Dict[str, dict], 
                              timestep: int = 0,
                              rebalance_interval: int = REBALANCE_INTERVAL
                              ) -> Dict[str, float]:
    """
    Minimum variance strategy that rebalances at fixed intervals
    """
    signals = {}
    
    # Check if it's time for periodic rebalancing
    should_rebalance = (timestep % rebalance_interval == 0)
    
    if should_rebalance:
        current_equity = portfolio.cash
        for symbol, row in portfolio.portfolio_df.iterrows():
            current_equity += row['position'] * current_prices[symbol]['markPrice']
        
        result = portfolio.get_optim_weights()
        
        if result is not None:
            optimal_weights = result.x
            
            for i, symbol in enumerate(portfolio.symbols):
                price = current_prices[symbol]['markPrice']
                limited_weight = min(optimal_weights[i], MAX_POSITION_PCT)
                target_position = (limited_weight * current_equity) / price
                current_position = portfolio.portfolio_df.loc[symbol, 'position']
                
                position_change = abs(target_position - current_position)
                max_change = abs(current_position * MAX_POSITION_CHANGE)
                
                if position_change > max_change:
                    if target_position > current_position:
                        target_position = current_position + max_change
                    else:
                        target_position = current_position - max_change
                
                signals[symbol] = target_position
        else:
            weight = 1.0 / len(portfolio.symbols)
            for symbol in portfolio.symbols:
                price = current_prices[symbol]['markPrice']
                signals[symbol] = (weight * current_equity) / price
    else:
        for symbol in portfolio.symbols:
            signals[symbol] = portfolio.portfolio_df.loc[symbol, 'position']
    
    return signals