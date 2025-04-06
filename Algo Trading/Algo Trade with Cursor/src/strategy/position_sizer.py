import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PositionSizer:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_size: float = 0.2,
        max_leverage: float = 2.0,
        min_confidence: float = 0.6,
        volatility_threshold: float = 0.02
    ):
        """
        Initialize position sizer.
        
        Args:
            initial_capital (float): Initial capital for position sizing
            max_position_size (float): Maximum position size as fraction of capital
            max_leverage (float): Maximum leverage allowed
            min_confidence (float): Minimum confidence required for trade
            volatility_threshold (float): Maximum allowed volatility
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.min_confidence = min_confidence
        self.volatility_threshold = volatility_threshold
        
    def calculate_position_size(
        self,
        symbol1: str,
        symbol2: str,
        confidence: float,
        zscore: float,
        price1: Optional[float] = None,
        price2: Optional[float] = None
    ) -> float:
        """
        Calculate position size for a pair of symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            confidence: Trade confidence (0-1)
            zscore: Current Z-score
            price1: Current price of first symbol
            price2: Current price of second symbol
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Check minimum confidence
            if confidence < self.min_confidence:
                logger.warning(f"Confidence {confidence:.4f} below minimum {self.min_confidence}")
                return 0.0
                
            # Calculate base position size
            base_size = self.initial_capital * self.max_position_size
            
            # Adjust for confidence
            confidence_adjusted = base_size * confidence
            
            # Adjust for Z-score magnitude
            zscore_adjusted = confidence_adjusted * (1 - abs(zscore) / self.max_leverage)
            
            # Apply leverage limit
            final_size = min(zscore_adjusted, self.initial_capital * self.max_leverage)
            
            # If prices are provided, convert to base currency
            if price1 is not None and price2 is not None:
                # Use the lower price to be conservative
                min_price = min(price1, price2)
                final_size = final_size / min_price
                
            logger.debug(f"Position size calculation for {symbol1}-{symbol2}:")
            logger.debug(f"Base size: {base_size:.2f}")
            logger.debug(f"Confidence adjusted: {confidence_adjusted:.2f}")
            logger.debug(f"Z-score adjusted: {zscore_adjusted:.2f}")
            logger.debug(f"Final size: {final_size:.2f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0 