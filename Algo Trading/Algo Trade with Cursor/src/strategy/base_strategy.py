from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, trading_pairs: List[str], timeframe: str = '1h'):
        """
        Initialize the base strategy.
        
        Args:
            trading_pairs (List[str]): List of trading pairs to trade
            timeframe (str): Timeframe for analysis (e.g., '1h', '4h', '1d')
        """
        self.trading_pairs = trading_pairs
        self.timeframe = timeframe
        self.positions: Dict[str, Dict] = {}
        self.signals: Dict[str, Dict] = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate trading signals based on the strategy logic.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            Dict[str, Dict]: Trading signals for each trading pair
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, 
                              symbol: str, 
                              signal: Dict,
                              account_balance: float) -> float:
        """
        Calculate the position size based on the signal and account balance.
        
        Args:
            symbol (str): Trading pair
            signal (Dict): Trading signal
            account_balance (float): Current account balance
            
        Returns:
            float: Position size in base currency
        """
        pass
    
    def update_positions(self, positions: Dict[str, Dict]) -> None:
        """
        Update current positions.
        
        Args:
            positions (Dict[str, Dict]): Current positions
        """
        self.positions = positions
    
    def get_signals(self) -> Dict[str, Dict]:
        """
        Get current trading signals.
        
        Returns:
            Dict[str, Dict]: Current trading signals
        """
        return self.signals
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators used by the strategy.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: Data with calculated indicators
        """
        return data
    
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate if a signal meets the strategy's criteria.
        
        Args:
            signal (Dict): Trading signal
            
        Returns:
            bool: True if signal is valid, False otherwise
        """
        return True 