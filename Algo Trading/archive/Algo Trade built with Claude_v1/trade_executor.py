from typing import Dict, List
import pandas as pd
import numpy as np
from portfolio import Portfolio
from config import TradingConfig

class TradeExecutor:
    """Handles trade execution and position updates"""
    
    def __init__(self, commission: float = TradingConfig.COMMISSION_RATE):
        """
        Initialize trade executor
        
        Args:
            commission: Commission rate as decimal (e.g., 0.001 for 0.1%)
        """
        self.commission = commission
    
    def execute_trades(
        self, 
        portfolio: Portfolio, 
        signals: Dict[str, float], 
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> List[Dict]:
        """
        Execute trades based on strategy signals
        
        Args:
            portfolio: Portfolio instance to execute trades on
            signals: Dictionary mapping symbols to target positions
            current_prices: Dictionary mapping symbols to current prices
            timestamp: Current timestamp
            
        Returns:
            List of executed trade records
            
        Raises:
            ValueError: If insufficient cash for trades or invalid signals
        """
        executed_trades = []
        
        # Validate signals
        if not isinstance(signals, dict):
            raise ValueError("Signals must be a dictionary")
            
        # Calculate net cash impact of all trades
        net_cash_impact = 0
        trade_plan = {}
        
        for symbol, target_position in signals.items():
            if symbol not in portfolio.assets:
                raise ValueError(f"Unknown symbol in signals: {symbol}")
                
            current_position = portfolio.get_position(symbol)
            if np.isclose(target_position, current_position, rtol=1e-5):
                continue
                
            price = current_prices[symbol]
            trade_size = target_position - current_position
            trade_value = abs(trade_size * price)
            commission = trade_value * self.commission
            
            # Calculate cash impact (negative for buys, positive for sells)
            cash_impact = -trade_value * np.sign(trade_size) - commission
            net_cash_impact += cash_impact
                
            trade_plan[symbol] = {
                'size': trade_size,
                'value': trade_value,
                'commission': commission,
                'price': price
            }
        
        # Check if enough cash available after all trades
        if portfolio.cash + net_cash_impact < 0:
            raise ValueError(
                f"Insufficient cash for trades. Net cash impact: {net_cash_impact:.2f}, "
                f"Available: {portfolio.cash:.2f}"
            )
        
        # Execute trades
        for symbol, trade in trade_plan.items():
            try:
                # Get current position
                current_position = portfolio.get_position(symbol)
                target_position = signals[symbol]
                
                # Record trade details
                trade_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': trade['price'],
                    'size': trade['size'],
                    'value': trade['value'],
                    'commission': trade['commission'],
                    'type': 'buy' if trade['size'] > 0 else 'sell',
                    'old_position': current_position,
                    'new_position': target_position
                }
                
                # Update position
                portfolio.update_position(symbol, target_position)
                
                # Update cash
                portfolio.update_cash(
                    -(trade['value'] * np.sign(trade['size']) + trade['commission'])
                )
                
                executed_trades.append(trade_record)
                
            except Exception as e:
                raise RuntimeError(f"Error executing trade for {symbol}: {str(e)}")
        
        return executed_trades
    
    def validate_trade(
        self, 
        portfolio: Portfolio,
        symbol: str, 
        target_position: float,
        current_price: float
    ) -> bool:
        """
        Validate if a trade is possible given current portfolio state
        
        Args:
            portfolio: Portfolio instance
            symbol: Trading symbol
            target_position: Desired position size
            current_price: Current price of the asset
            
        Returns:
            True if trade is valid, False otherwise
        """
        if symbol not in portfolio.assets:
            return False
            
        current_position = portfolio.get_position(symbol)
        trade_size = target_position - current_position
        
        if trade_size == 0:
            return False
            
        trade_value = abs(trade_size * current_price)
        commission = trade_value * self.commission
        
        if trade_size > 0:  # Buy trade
            return trade_value + commission <= portfolio.cash
            
        return True  # Sell trades always valid if position exists 