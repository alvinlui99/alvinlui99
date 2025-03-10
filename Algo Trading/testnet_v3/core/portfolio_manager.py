"""
Portfolio manager for constructing diversified portfolios based on strategy signals.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

class PortfolioManager:
    """
    Handles portfolio construction and risk management.
    """
    
    def __init__(self, client, symbols: List[str], total_capital: float = 1000.0, 
                risk_per_trade: float = 0.02, max_allocation: float = 0.25,
                logger=None):
        """
        Initialize the portfolio manager.
        
        Args:
            client: Binance Futures client
            symbols: List of trading symbols
            total_capital: Total capital to allocate
            risk_per_trade: Maximum risk per trade as a fraction of capital
            max_allocation: Maximum allocation to a single asset
            logger: Optional logger instance
        """
        self.client = client
        self.symbols = symbols
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade
        self.max_allocation = max_allocation
        self.logger = logger or logging.getLogger(__name__)
        
        # Current portfolio state
        self.positions = {}
        self.available_capital = total_capital
        
    def update_account_info(self):
        """
        Update account information from Binance.
        """
        try:
            account_info = self.client.account()
            
            # Update capital
            self.total_capital = float(account_info['totalWalletBalance'])
            self.available_capital = float(account_info['availableBalance'])
            
            # Update positions
            positions = {}
            for position in account_info.get('positions', []):
                symbol = position['symbol']
                position_amt = float(position['positionAmt'])
                
                if abs(position_amt) > 0:
                    positions[symbol] = {
                        'quantity': position_amt,
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'pnl': float(position['unrealizedProfit'])
                    }
            
            self.positions = positions
            self.logger.info(f"Account updated: Capital=${self.total_capital:.2f}, Positions: {len(self.positions)}")
            
        except Exception as e:
            self.logger.error(f"Error updating account info: {str(e)}")
            
    def construct_portfolio(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Construct a diversified portfolio based on strategy signals.
        
        Args:
            signals: Dictionary of signals from the strategy
                Format: {
                    symbol: {
                        "side": "BUY"/"SELL"/"NONE",
                        "confidence": float,
                        "price": float
                    }
                }
                
        Returns:
            Dictionary of trade decisions:
                Format: {
                    symbol: {
                        "side": "BUY"/"SELL"/"CLOSE",
                        "quantity": float,
                        "price": float,
                        "type": "MARKET"/"LIMIT"
                    }
                }
        """
        self.logger.info("Constructing portfolio from signals")
        self.update_account_info()  # Get latest account information
        
        # Filter for only actionable signals
        actionable_signals = {
            symbol: signal for symbol, signal in signals.items()
            if signal['side'] in ['BUY', 'SELL'] and symbol in self.symbols
        }
        
        # Rank signals by confidence
        sorted_signals = sorted(
            actionable_signals.items(), 
            key=lambda x: x[1]['confidence'], 
            reverse=True
        )
        
        # Calculate allocation for each signal
        # Start with simple equal weighting for now
        decisions = {}
        
        if not sorted_signals:
            self.logger.info("No actionable signals")
            return decisions
            
        # Get available capital for new positions
        available_for_new = self.available_capital
        
        # Determine allocation per position, capped by max_allocation
        # This is a simplified approach - in production you'd want more sophisticated allocation
        max_positions = min(len(sorted_signals), int(1 / self.max_allocation))
        allocation_per_position = min(
            self.max_allocation * self.total_capital,
            available_for_new / max_positions
        )
        
        # Construct trade decisions
        for symbol, signal in sorted_signals[:max_positions]:
            # If we already have a position in this symbol
            if symbol in self.positions:
                current_pos = self.positions[symbol]
                current_side = "LONG" if current_pos['quantity'] > 0 else "SHORT"
                
                # If signal is in opposite direction, close existing position
                if (signal['side'] == 'BUY' and current_side == 'SHORT') or \
                   (signal['side'] == 'SELL' and current_side == 'LONG'):
                    decisions[symbol] = {
                        "side": "CLOSE",
                        "quantity": abs(current_pos['quantity']),
                        "price": signal['price'],
                        "type": "MARKET"
                    }
                    # Signal to open new position in opposite direction
                    quantity = self._calculate_position_size(
                        symbol, signal['price'], allocation_per_position, signal['side']
                    )
                    if quantity > 0:
                        decisions[symbol] = {
                            "side": signal['side'],
                            "quantity": quantity,
                            "price": signal['price'],
                            "type": "MARKET"
                        }
                
                # If signal is in same direction, may adjust position size
                # (For simplicity, we're not implementing position sizing adjustments here)
                        
            else:
                # New position
                quantity = self._calculate_position_size(
                    symbol, signal['price'], allocation_per_position, signal['side']
                )
                if quantity > 0:
                    decisions[symbol] = {
                        "side": signal['side'],
                        "quantity": quantity,
                        "price": signal['price'],
                        "type": "MARKET"
                    }
        
        self.logger.info(f"Generated {len(decisions)} trade decisions")
        return decisions
    
    def _calculate_position_size(self, symbol: str, price: float, 
                               allocation: float, side: str) -> float:
        """
        Calculate position size based on allocation and risk management.
        
        Args:
            symbol: Trading symbol
            price: Current price
            allocation: Capital to allocate
            side: Trade side ('BUY' or 'SELL')
            
        Returns:
            Float: Quantity to trade
        """
        # This is a simplified implementation
        # In production, you'd want to incorporate stop loss levels, volatility, etc.
        
        try:
            # Get symbol info
            # In a real implementation, you'd get precision requirements from the exchange
            min_qty = 0.001  # Minimum quantity
            
            # Calculate raw quantity based on allocation
            raw_quantity = allocation / price
            
            # Round to minimum quantity precision
            quantity = max(min_qty, raw_quantity)
            
            # Apply any other constraints
            # For example, account for leverage, margin requirements, etc.
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 0.0 