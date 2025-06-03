from statsmodels.tsa.stattools import coint
from itertools import combinations
import pandas as pd

from collector import BinanceDataCollector

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

    def collect_data(self, interval='1h', days_back=10):
        self.data = self.collector.get_multiple_symbols_data(self.coins, interval=interval, days_back=days_back)

    def cointegration_filter(self, pvalue_threshold=0.05):
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

    def run(self, interval='1h', days_back=10, pvalue_threshold=0.05):
        self.collect_data(interval=interval, days_back=days_back)
        selected_pairs = self.cointegration_filter(pvalue_threshold=pvalue_threshold)
        print("Selected cointegrated pairs (p < 0.05):")
        for pair in selected_pairs:
            print(f"{pair[0]}-{pair[1]}: p-value={pair[2]:.4f}")
        return selected_pairs
    
if __name__ == "__main__":
    selector = Selector()
    selector.run(interval='1h', days_back=10)