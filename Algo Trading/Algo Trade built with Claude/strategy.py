from typing import Dict, Tuple
from portfolio import Portfolio
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.portfolio_values = []
        self.cash_values = []
        self.timestamps = []
    
    def calculate_current_weights(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> Tuple[float, Dict[str, float]]:
        """Calculate current portfolio equity and weights"""
        # First calculate total equity
        current_equity = portfolio.cash
        position_values = {}
        
        # Calculate all position values first
        for symbol, row in portfolio.portfolio_df.iterrows():
            position = row['position']
            if isinstance(position, pd.Series):
                position = position.iloc[0]
            
            position = float(position)
            price = float(current_prices[symbol]['markPrice'])
            position_value = position * price
            current_equity += position_value
            position_values[symbol] = position_value
        
        # Then calculate weights using final equity value
        current_weights = {
            symbol: value / current_equity if current_equity > 0 else 0
            for symbol, value in position_values.items()
        }
        
        return current_equity, current_weights
    
    def calculate_positions(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                          weights_dict: Dict[str, float], current_equity: float) -> Dict[str, float]:
        """Calculate target positions while respecting cash constraints"""
        available_cash = portfolio.cash
        positions = {}
        total_target_value = current_equity  # Total portfolio target value
        
        # First pass: calculate all desired positions and cash requirements
        total_buy_value = 0
        total_sell_value = 0
        
        for symbol in portfolio.symbols:
            if symbol in weights_dict:
                price = float(current_prices[symbol]['markPrice'])
                
                # Handle position access properly
                position_value = portfolio.portfolio_df.loc[symbol, 'position']
                if isinstance(position_value, pd.Series):
                    current_pos = float(position_value.iloc[0])
                else:
                    current_pos = float(position_value)
                
                # Calculate target value for this asset
                target_value = float(weights_dict[symbol] * total_target_value)
                current_value = current_pos * price
                value_diff = target_value - current_value
                
                positions[symbol] = {
                    'price': price,
                    'current_pos': current_pos,
                    'target_value': target_value,
                    'value_diff': value_diff
                }
                
                if value_diff > 0:
                    total_buy_value += value_diff
                else:
                    total_sell_value += abs(value_diff)
        
        # Calculate scaling factor if we need more cash than available
        available_cash_with_sells = available_cash + total_sell_value
        if total_buy_value > available_cash_with_sells:
            buy_scale_factor = available_cash_with_sells / total_buy_value
        else:
            buy_scale_factor = 1.0
            
        # Second pass: calculate final positions with scaling
        final_positions = {}
        total_final_value = 0
        
        for symbol, pos_info in positions.items():
            price = pos_info['price']
            current_pos = pos_info['current_pos']
            value_diff = pos_info['value_diff']
            
            if value_diff > 0:
                # Scale down buys if necessary
                scaled_value_diff = value_diff * buy_scale_factor
            else:
                # Keep sells as is
                scaled_value_diff = value_diff
            
            # Calculate final position
            target_value = current_pos * price + scaled_value_diff
            final_position = target_value / price
            final_positions[symbol] = final_position
            
            total_final_value += final_position * price
            
        return final_positions
    
    def track_portfolio_state(self, portfolio: Portfolio, current_equity: float, timestep: int = 0):
        """Track portfolio and cash values over time"""
        self.portfolio_values.append(current_equity)
        self.cash_values.append(portfolio.cash)
        self.timestamps.append(pd.Timestamp.now() if timestep == 0 else pd.Timestamp.now() + pd.Timedelta(hours=timestep))
    
    @abstractmethod
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], timestep: int = 0) -> Dict[str, float]:
        """
        Execute strategy and return target positions
        
        Args:
            portfolio: Portfolio object containing current positions
            current_prices: Dictionary of current prices for each symbol
            timestep: Current timestep in the simulation
            
        Returns:
            Dictionary mapping symbols to target positions
        """
        pass 