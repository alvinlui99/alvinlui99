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
        self.p_values = []
        self.marginal_params = []

    def evaluate(self, asset: str) -> float:
        result = stats.kstest(self.returns[asset], 't', args=self.models[asset])
        return result.pvalue

    def fit_assets(self, data: Dict[str, pd.DataFrame], asset_names: list, returns_col: str = 'close') -> Dict[str, Dict]:
        for asset in asset_names:
            df = data[asset]
            self.returns[asset] = df[returns_col].pct_change().dropna().values
            self.models[asset] = stats.t.fit(self.returns[asset])
            self.p_values.append({
                'asset': asset,
                'p_value': self.evaluate(asset)
            })
            self.marginal_params.append({
                'asset': asset,
                'df': self.models[asset][0],
                'loc': self.models[asset][1],
                'scale': self.models[asset][2]
            })

    def save_marginal_params(self, file_extension: str = ''):
        pd.DataFrame(self.marginal_params).to_csv(f'model_params/marginal_params{file_extension}.csv', index=False)
        pd.DataFrame(self.p_values).to_csv(f'model_params/marginal_p_values{file_extension}.csv', index=False)

if __name__ == "__main__":
    from collector import BybitDataCollector, BinanceDataCollector
    from config import Config
    coins = Config().coins
    # coins = ['ETHUSDT']
    # collector = BybitDataCollector()
    collector = BinanceDataCollector()
    formation_start_str = '2022-06-01 00:00:00'
    formation_end_str = '2022-12-31 23:59:59'
    data = collector.get_multiple_symbols_data(symbols=coins, start_str=formation_start_str, end_str=formation_end_str)
    marginal_fitter = MarginalFitter()
    marginal_fitter.fit_assets(data, coins)