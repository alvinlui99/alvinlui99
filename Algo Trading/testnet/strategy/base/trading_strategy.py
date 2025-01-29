from typing import Dict
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class TradingStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.is_configured = False

    def get_signals(self) -> Dict[str, float]:
        """
        Get target weights for each asset
        """
        pass