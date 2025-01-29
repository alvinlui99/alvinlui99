from typing import Dict
import pandas as pd
from abc import ABC, abstractmethod

class IndicatorCalculator(ABC):
    @abstractmethod
    def calculate_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        pass