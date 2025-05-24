"""
Position sizer for statistical arbitrage strategy.

This module provides functionality for calculating position sizes based on:
- Account balance
- Risk parameters
- Market conditions
- Signal confidence
"""

from typing import Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PositionSizer:
    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.2,
                 max_leverage: float = 2.0,
                 min_confidence: float = 0.6,
                 volatility_threshold: float = 0.02):
        """
        Initialize position sizer.
        
        Args:
            initial_capital: Initial capital for trading
            max_position_size: Maximum position size as fraction of capital
            max_leverage: Maximum leverage allowed
            min_confidence: Minimum confidence required for trade
            volatility_threshold: Maximum allowed volatility
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.min_confidence = min_confidence
        self.volatility_threshold = volatility_threshold
        
    def calculate_position_size(self,
                              symbol1: str,
                              symbol2: str,
                              confidence: float,
                              zscore: float) -> float:
        """
        Calculate position size based on various factors.
        
        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            confidence: Signal confidence
            zscore: Current z-score
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Base position size as percentage of capital
            base_size = self.initial_capital * self.max_position_size
            
            # Adjust for confidence
            confidence_factor = min(confidence, 1.0)
            position_size = base_size * confidence_factor
            
            # Adjust for z-score magnitude
            zscore_factor = min(1.0, abs(zscore) / 3.0)  # Cap at 1.0
            position_size *= zscore_factor
            
            # Apply leverage limit
            max_leveraged_size = self.initial_capital * self.max_leverage
            position_size = min(position_size, max_leveraged_size)
            
            logger.info(f"Position size calculation for {symbol1}-{symbol2}:")
            logger.info(f"Base size: ${base_size:.2f}")
            logger.info(f"Confidence factor: {confidence_factor:.2f}")
            logger.info(f"Z-score factor: {zscore_factor:.2f}")
            logger.info(f"Final size: ${position_size:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0 