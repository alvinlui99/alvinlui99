from itertools import combinations
import pandas as pd
from typing import List, Dict, Optional

from statsmodels.tsa.stattools import coint

from collector import BinanceDataCollector
from config import Config

class Selector:
    def __init__(self, coins=None):
        self.config = Config()
        self.coins = self.config.coins if coins is None else coins
        self.collector = BinanceDataCollector()
        self.data = None

    def cointegration_filter(self, data: Dict[str, pd.DataFrame], pvalue_threshold: float = None):
        if pvalue_threshold is None:
            pvalue_threshold = Config().coint_pvalue_threshold
        selected_pairs = []
        pairs = list(combinations(self.coins, 2))
        for c1, c2 in pairs:
            s1 = data[c1]['close']
            s2 = data[c2]['close']
            # Align on timestamp
            df = pd.DataFrame({c1: s1.values, c2: s2.values})
            df = df.dropna()
            if len(df) < 2:
                continue
            result = coint(df[c1], df[c2])
            if result[1] < pvalue_threshold:
                selected_pairs.append((c1, c2))
        return selected_pairs

    def run(self, data: Dict[str, pd.DataFrame], pvalue_threshold: float = None):
        if pvalue_threshold is None:
            pvalue_threshold = Config().coint_pvalue_threshold
        selected_pairs = self.cointegration_filter(data)
        return selected_pairs
    
if __name__ == "__main__":
    selector = Selector()
    selector.run()