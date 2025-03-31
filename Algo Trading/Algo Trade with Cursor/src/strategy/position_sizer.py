import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from src.strategy.zscore_monitor import ZScoreSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PositionSize:
    """Data class to hold position sizing information."""
    symbol1: str
    symbol2: str
    size1: float  # Position size for first symbol
    size2: float  # Position size for second symbol
    leverage: float
    confidence: float  # Statistical confidence (0-1)
    volatility: float  # Current volatility
    timestamp: str

class PositionSizer:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_size: float = 0.2,  # Maximum 20% of portfolio per asset
        max_leverage: float = 2.0,
        min_confidence: float = 0.6,  # Minimum confidence required to enter
        volatility_threshold: float = 0.02  # 2% volatility threshold
    ):
        """
        Initialize the position sizing system.
        
        Args:
            initial_capital: Initial trading capital
            max_position_size: Maximum position size as fraction of portfolio
            max_leverage: Maximum allowed leverage
            min_confidence: Minimum confidence required to enter a position
            volatility_threshold: Maximum allowed volatility
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.min_confidence = min_confidence
        self.volatility_threshold = volatility_threshold
        
        # Track open positions
        self.open_positions: Dict[Tuple[str, str], PositionSize] = {}
    
    def calculate_position_size(
        self,
        signal: ZScoreSignal,
        price1: float,
        price2: float
    ) -> Optional[PositionSize]:
        """
        Calculate position size based on Z-score signal and current prices.
        
        Args:
            signal: Z-score signal containing spread statistics
            price1: Current price of first symbol
            price2: Current price of second symbol
            
        Returns:
            PositionSize object if position should be taken, None otherwise
        """
        # Skip if no entry signal
        if signal.signal_type not in ['ENTRY_LONG', 'ENTRY_SHORT']:
            return None
        
        # Calculate statistical confidence based on Z-score magnitude
        zscore_magnitude = abs(signal.zscore)
        confidence = min(1.0, zscore_magnitude / signal.std)
        
        # Skip if confidence is too low
        if confidence < self.min_confidence:
            logger.info(f"Confidence {confidence:.2f} below minimum threshold")
            return None
        
        # Calculate volatility (normalized standard deviation)
        volatility = signal.std / signal.mean
        
        # Skip if volatility is too high
        if volatility > self.volatility_threshold:
            logger.info(f"Volatility {volatility:.2f} above threshold")
            return None
        
        # Calculate base position size based on confidence
        base_size = self.current_capital * self.max_position_size * confidence
        
        # Adjust for volatility
        volatility_factor = 1 - (volatility / self.volatility_threshold)
        adjusted_size = base_size * volatility_factor
        
        # Calculate leverage based on confidence and volatility
        leverage = min(
            self.max_leverage,
            1 + (confidence * (self.max_leverage - 1))
        )
        
        # Calculate actual position sizes
        if signal.signal_type == 'ENTRY_LONG':
            size1 = adjusted_size * leverage
            size2 = -adjusted_size * leverage
        else:  # ENTRY_SHORT
            size1 = -adjusted_size * leverage
            size2 = adjusted_size * leverage
        
        # Create position size object
        position = PositionSize(
            symbol1=signal.pair[0],
            symbol2=signal.pair[1],
            size1=size1,
            size2=size2,
            leverage=leverage,
            confidence=confidence,
            volatility=volatility,
            timestamp=signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        logger.info(f"Calculated position size for {signal.pair[0]}-{signal.pair[1]}:")
        logger.info(f"Size1: {size1:.2f}, Size2: {size2:.2f}")
        logger.info(f"Leverage: {leverage:.2f}, Confidence: {confidence:.2f}")
        logger.info(f"Volatility: {volatility:.2f}")
        
        return position
    
    def update_position(
        self,
        pair: Tuple[str, str],
        pnl: float
    ) -> None:
        """
        Update position and capital after PnL realization.
        
        Args:
            pair: Trading pair
            pnl: Realized profit/loss
        """
        self.current_capital += pnl
        
        if pair in self.open_positions:
            del self.open_positions[pair]
    
    def get_position(self, pair: Tuple[str, str]) -> Optional[PositionSize]:
        """Get current position for a pair."""
        return self.open_positions.get(pair)
    
    def get_all_positions(self) -> Dict[Tuple[str, str], PositionSize]:
        """Get all current positions."""
        return self.open_positions.copy()
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.current_capital 