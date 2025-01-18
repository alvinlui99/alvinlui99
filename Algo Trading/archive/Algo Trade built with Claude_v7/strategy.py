from typing import Dict, Tuple
from portfolio import Portfolio
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from config import TradingConfig

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.portfolio_values = []
        self.cash_values = []
        self.timestamps = []
    
    def calculate_current_weights(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate current portfolio equity and weights
        
        Args:
            portfolio: Portfolio object containing current positions
            current_prices: Dictionary of current prices for each symbol
            
        Returns:
            Tuple[float, Dict[str, float]]: (total_equity, weights_dictionary)
        """
        if not current_prices:
            raise ValueError("current_prices dictionary cannot be empty")
        
        current_equity = portfolio.cash
        position_values = {}
        
        # Calculate total equity and position values
        for symbol, position in portfolio.positions.items():
            if symbol not in current_prices:
                raise KeyError(f"Price data missing for symbol {symbol}")
            
            price = float(current_prices[symbol]['markPrice'])
            position_value = position * price
            position_values[symbol] = position_value
            current_equity += position_value
        
        # Calculate weights based on total equity
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
        total_target_value = current_equity
        
        # First pass: calculate all desired positions and cash requirements
        total_buy_value = 0
        total_sell_value = 0
        
        # Cache price conversions at the start
        prices = {symbol: float(current_prices[symbol]['markPrice']) for symbol in portfolio.assets.keys()}
        
        # Use cached prices throughout the method
        for symbol in portfolio.assets.keys():
            if symbol in weights_dict:
                price = prices[symbol]
                current_pos = portfolio.get_position(symbol)
                
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
        """
        Track portfolio and cash values over time.
        
        Args:
            portfolio (Portfolio): Portfolio object containing current positions
            current_equity (float): Current total portfolio value including cash
            timestep (int): Number of hours to add to current timestamp (for backtesting)
            
        Note:
            This method maintains historical records of:
            - Total portfolio value
            - Cash balance
            - Timestamps for each update
        """
        self.portfolio_values.append(current_equity)
        self.cash_values.append(portfolio.cash)
        self.timestamps.append(pd.Timestamp.now() if timestep == 0 else pd.Timestamp.now() + pd.Timedelta(hours=timestep))
    
    @abstractmethod
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], equity_curve: pd.Series) -> Dict[str, Dict[str, float]]:
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
    
    def get_commission_adjusted_equity(self, total_equity: float) -> float:
        """
        Returns the equity available for position sizing after reserving for commissions
        
        Args:
            total_equity: Total portfolio equity value
            
        Returns:
            Adjusted equity value after commission buffer
        """
        return total_equity * (1 - TradingConfig.COMMISSION_BUFFER_PERCENTAGE) 