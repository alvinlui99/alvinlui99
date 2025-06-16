import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt

from typing import Dict

class MarginalFitter:
    def __init__(self):
        self.models = {}
        self.returns = {}

    def evaluate(self, asset: str) -> float:
        result = stats.kstest(self.returns[asset], 't', args=self.models[asset])
        return result.pvalue

    def fit_assets(self, data: Dict[str, pd.DataFrame], asset_names: list, returns_col: str = 'close') -> Dict[str, Dict]:
        summary = {}
        marginal_params = []
        for asset in asset_names:
            df = data[asset]
            self.returns[asset] = df[returns_col].pct_change().dropna().values
            self.models[asset] = stats.t.fit(self.returns[asset])
            p_value = self.evaluate(asset)
            uniform = stats.t.cdf(self.returns[asset], *self.models[asset])
            summary[asset] = {
                'returns': self.returns[asset],
                'uniform': uniform,
                'params': self.models[asset],
                'p_value': p_value
            }
            marginal_params.append({
                'asset': asset,
                'df': self.models[asset][0],
                'loc': self.models[asset][1],
                'scale': self.models[asset][2]
            })
        pd.DataFrame(marginal_params).to_csv('model_params/marginal_params.csv', index=False)
        return summary
    
    def qq_plot(self, asset: str):
        qqplot(self.returns[asset], stats.t, fit=True, line='45')
        plt.title(f'QQ Plot for {asset}')
        plt.show()

if __name__ == "__main__":
    from collector import BinanceDataCollector
    from config import Config
    coins = Config().coins
    collector = BinanceDataCollector()
    data = collector.get_multiple_symbols_data(symbols=coins, start_str='2024-01-01 00:00:00', end_str='2025-06-15 23:59:59')
    marginal_fitter = MarginalFitter()
    summary = marginal_fitter.fit_assets(data, coins)