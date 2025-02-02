from typing import Dict
import pandas as pd
from abc import ABC, abstractmethod

class SignalCalculator(ABC):
    @abstractmethod
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        pass