from statsmodels.tsa.stattools import coint
from itertools import combinations
import pandas as pd

from collector import BinanceDataCollector
from config import Config

class Selector:
    def __init__(self, coins=None):
        if coins is None:
            self.coins = [
                "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
                "DOTUSDT","LINKUSDT","MATICUSDT","AVAXUSDT","ATOMUSDT",
                "LTCUSDT","XRPUSDT","UNIUSDT","AAVEUSDT","DOGEUSDT"
            ]
        else:
            self.coins = coins
        self.collector = BinanceDataCollector()
        self.data = None

    def collect_data(self, interval: str = None, days_back: int = None):
        if interval is None:
            interval = Config().interval
        if days_back is None:
            days_back = Config().lookback_days
        self.data = self.collector.get_multiple_symbols_data(self.coins, interval=interval, days_back=days_back)

    def cointegration_filter(self, pvalue_threshold: float = None):
        if pvalue_threshold is None:
            pvalue_threshold = Config().coint_pvalue_threshold
        if self.data is None:
            raise ValueError("Data not collected. Call collect_data() first.")
        selected_pairs = []
        pairs = list(combinations(self.data.keys(), 2))
        for c1, c2 in pairs:
            s1 = self.data[c1]['close']
            s2 = self.data[c2]['close']
            # Align on timestamp
            df = pd.DataFrame({c1: s1.values, c2: s2.values})
            df = df.dropna()
            if len(df) < 2:
                continue
            result = coint(df[c1], df[c2])
            if result[1] < pvalue_threshold:
                selected_pairs.append((c1, c2, result[1]))
        return selected_pairs

    def run(self, interval: str = None, days_back: int = None, pvalue_threshold: float = None):
        if interval is None:
            interval = Config().interval
        if days_back is None:
            days_back = Config().lookback_days
        if pvalue_threshold is None:
            pvalue_threshold = Config().coint_pvalue_threshold

        self.collect_data(interval=interval, days_back=days_back)
        selected_pairs = self.cointegration_filter(pvalue_threshold=pvalue_threshold)
        print(f"Selected cointegrated pairs (p < {pvalue_threshold}):")
        for pair in selected_pairs:
            print(f"{pair[0]}-{pair[1]}: p-value={pair[2]:.4f}")
        return selected_pairs
    
if __name__ == "__main__":
    selector = Selector()
    selector.run()