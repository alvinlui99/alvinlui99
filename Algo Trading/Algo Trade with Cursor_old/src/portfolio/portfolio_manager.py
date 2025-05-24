import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 10000,
                 commission: float = 0.0004):
        """
        Initialize the portfolio manager.
        
        Args:
            symbols (List[str]): List of trading pairs
            initial_capital (float): Initial capital in USDT
            commission (float): Commission rate per trade (default: 0.04%)
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Portfolio state
        self.positions: Dict[str, Dict] = {
            symbol: {
                'size': 0,
                'entry_price': 0,
                'current_price': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0
            } for symbol in symbols
        }
        
        # Portfolio metrics
        self.total_capital = initial_capital
        self.available_capital = initial_capital
        self.equity_curve: List[float] = [initial_capital]
        self.trades: List[Dict] = []
        
        # Risk management parameters
        self.max_position_size: Dict[str, float] = {
            symbol: initial_capital * 0.2 for symbol in symbols  # Max 20% per asset
        }
        self.max_drawdown = 0.1  # 10% maximum drawdown
        self.current_drawdown = 0
        self.peak_value = initial_capital
        
        # Performance tracking
        self.daily_returns: List[float] = []
        
        # Initialize portfolio weights with equal weights
        weight = 1.0 / len(symbols)
        self.portfolio_weights: Dict[str, float] = {
            symbol: weight for symbol in symbols
        }
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for all assets.
        
        Args:
            prices (Dict[str, float]): Current prices for each symbol
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position['current_price'] = price
                if position['size'] != 0:  # Only update PnL if position exists
                    self._update_position_pnl(symbol)
        
        self._update_portfolio_metrics()
    
    def _update_position_pnl(self, symbol: str) -> None:
        """Update unrealized PnL for a position."""
        position = self.positions[symbol]
        if position['size'] != 0 and position['entry_price'] != 0:
            if position['size'] > 0:  # Long position
                position['unrealized_pnl'] = (position['current_price'] - position['entry_price']) * position['size']
            else:  # Short position
                position['unrealized_pnl'] = (position['entry_price'] - position['current_price']) * abs(position['size'])
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio-level metrics."""
        # Calculate total portfolio value
        total_value = self.available_capital
        total_position_value = 0
        
        # Calculate unrealized PnL and position values
        for symbol, position in self.positions.items():
            if position['size'] != 0:
                # Update unrealized PnL first
                self._update_position_pnl(symbol)
                total_value += position['unrealized_pnl']
                total_position_value += abs(position['size'] * position['current_price'])
        
        # Update equity curve
        self.equity_curve.append(total_value)
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            daily_return = (total_value / self.equity_curve[-2] - 1)
            self.daily_returns.append(daily_return)
        
        # Update peak value and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - total_value) / self.peak_value
        else:
            self.current_drawdown = 0
        
        # Update portfolio weights
        if total_position_value > 0:
            for symbol, position in self.positions.items():
                position_value = abs(position['size'] * position['current_price'])
                self.portfolio_weights[symbol] = position_value / total_position_value
        else:
            # If no positions, distribute weights equally
            weight = 1.0 / len(self.symbols)
            for symbol in self.symbols:
                self.portfolio_weights[symbol] = weight
    
    def can_open_position(self, symbol: str, size: float, price: float) -> bool:
        """
        Check if a position can be opened.
        
        Args:
            symbol (str): Trading pair
            size (float): Position size
            price (float): Current price
            
        Returns:
            bool: True if position can be opened
        """
        # Check if symbol is in portfolio
        if symbol not in self.positions:
            return False
        
        # Calculate required capital including commission
        required_capital = abs(size * price * (1 + self.commission))
        
        # Check if we have enough capital
        if required_capital > self.available_capital:
            return False
        
        # Check position size limits
        position_value = abs(size * price)
        if position_value > self.max_position_size[symbol]:
            return False
        
        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown:
            return False
        
        return True
    
    def open_position(self, symbol: str, size: float, price: float, timestamp: datetime) -> bool:
        """
        Open a new position.
        
        Args:
            symbol (str): Trading pair
            size (float): Position size
            price (float): Entry price
            timestamp (datetime): Trade timestamp
            
        Returns:
            bool: True if position was opened successfully
        """
        if not self.can_open_position(symbol, size, price):
            return False
        
        # Calculate cost including commission
        cost = abs(size * price * (1 + self.commission))
        
        # Check if we have enough capital again (double check)
        if cost > self.available_capital:
            return False
        
        # Update position
        self.positions[symbol].update({
            'size': size,
            'entry_price': price,
            'current_price': price,
            'unrealized_pnl': 0
        })
        
        # Update available capital
        self.available_capital -= cost
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'BUY' if size > 0 else 'SELL',
            'size': size,
            'price': price,
            'cost': cost,
            'pnl': 0
        })
        
        # Update metrics after position change
        self._update_portfolio_metrics()
        
        return True
    
    def close_position(self, symbol: str, price: float, timestamp: datetime) -> bool:
        """
        Close an existing position.
        
        Args:
            symbol (str): Trading pair
            price (float): Exit price
            timestamp (datetime): Trade timestamp
            
        Returns:
            bool: True if position was closed successfully
        """
        position = self.positions[symbol]
        if position['size'] == 0:
            return False
        
        old_size = position['size']  # Store for trade record
        entry_price = position['entry_price']  # Store entry price before resetting
        
        # Calculate PnL
        if position['size'] > 0:  # Long position
            pnl = (price - entry_price) * position['size']
        else:  # Short position
            pnl = (entry_price - price) * abs(position['size'])
        
        # Account for commission (both entry and exit)
        entry_commission = abs(position['size'] * entry_price * self.commission)
        exit_commission = abs(position['size'] * price * self.commission)
        total_commission = entry_commission + exit_commission
        net_pnl = pnl - total_commission
        
        # Calculate position value
        position_value = abs(old_size * entry_price)
        
        # Update position
        position['realized_pnl'] += net_pnl
        position['size'] = 0
        position['entry_price'] = 0
        position['unrealized_pnl'] = 0
        position['current_price'] = price
        
        # Update available capital
        # Return the original position value plus net PnL
        self.available_capital += position_value + net_pnl
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'CLOSE',
            'size': -old_size,  # Opposite of current position
            'price': price,
            'cost': exit_commission,  # Only record exit commission
            'pnl': net_pnl
        })
        
        # Update metrics after position change
        self._update_portfolio_metrics()
        
        return True
    
    def get_portfolio_metrics(self) -> Dict:
        """
        Get current portfolio metrics.
        
        Returns:
            Dict: Portfolio metrics
        """
        total_value = self.available_capital
        total_pnl = 0
        total_commission = 0
        
        for symbol, position in self.positions.items():
            total_value += position['unrealized_pnl']
            total_pnl += position['realized_pnl'] + position['unrealized_pnl']
            total_commission += abs(position['size'] * position['current_price'] * self.commission)
        
        return {
            'total_value': total_value,
            'available_capital': self.available_capital,
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'current_drawdown': self.current_drawdown,
            'portfolio_weights': self.portfolio_weights.copy(),
            'positions': self.positions.copy()
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as a DataFrame.
        
        Returns:
            pd.DataFrame: Trade history
        """
        return pd.DataFrame(self.trades)
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as a Series.
        
        Returns:
            pd.Series: Equity curve
        """
        return pd.Series(self.equity_curve) 