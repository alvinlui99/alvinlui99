import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import norm, t

from selection import Selector
from marginal_fitter import MarginalFitter
from copula_fitter import CopulaFitter
from collector import BinanceDataCollector

class CopulaBasedStrategy:
    def __init__(self):
        self.collector = BinanceDataCollector()
        self.selector = Selector()
        self.marginal_fitter = MarginalFitter()
        self.copula_fitter = CopulaFitter()

    def asset_names_from_selected_pairs(self, selected_pairs: list[tuple[str, str, float]]) -> list[str]:
        asset_names = set()
        for c1, c2 in selected_pairs:
            asset_names.add(c1)
            asset_names.add(c2)
        return list(asset_names)

    def gaussian_conditional_prob(self, cop, u1, u2):
        rho = cop.sigma[0,1]  # correlation coefficient
        z1 = norm.ppf(u1)
        z2 = norm.ppf(u2)
        
        # Conditional mean and variance
        cond_mu = rho * z2
        cond_var = 1 - rho**2
        return norm.cdf((z1 - cond_mu)/np.sqrt(cond_var))

    def t_conditional_prob(self, cop, u1, u2):
        rho = cop.sigma[0,1]
        df = cop._df

        x1 = t.ppf(u1, df=df)
        x2 = t.ppf(u2, df=df)

        
        # Conditional parameters
        cond_mu = rho * x2
        cond_scale = np.sqrt((df + x2**2) * (1 - rho**2) / (df + 1))
        return t.cdf((x1 - cond_mu)/cond_scale, df + 1)

    def run(self, data: Dict[str, pd.DataFrame], current_pairs: List[tuple[str, str]]):
        signals = {}
        selected_pairs = self.selector.run(data) + current_pairs

        asset_names = self.asset_names_from_selected_pairs(selected_pairs)
        marginal_summary = self.marginal_fitter.fit_assets(data, asset_names)
        copula_summary = self.copula_fitter.fit_assets(selected_pairs, marginal_summary)
        for c1, c2 in selected_pairs:
            current_returns = self.get_current_returns([c1, c2], data)
            mi = self.compute_mispricing_index((c1, c2), current_returns, marginal_summary, copula_summary)
            signals[f'{c1}-{c2}'] = {
                'mi': mi,
                'c1': c1,
                'c2': c2,
                'current_returns': current_returns,
                'marginal_summary': marginal_summary,
                'copula_summary': copula_summary
            }
        return signals
    
    def compute_mispricing_index(self,
                                 pair: tuple[str, str],
                                 returns: pd.DataFrame,
                                 marginal_summary: dict,
                                 copula_summary: dict) -> pd.Series:
        """
        Compute the mispricing index (MI) time series for a given asset pair using copulae.

        Args:
            pair: tuple of (asset1, asset2)
            returns: pd.DataFrame with columns for asset1 and asset2
            marginal_summary: dict with fitted marginal CDFs for each asset
            copula_summary: dict with fitted copula objects for each pair

        Returns:
            pd.Series of MI_Y_given_X for each time point
        """
        asset1, asset2 = pair
        marginal_X = marginal_summary[asset1]['best_dist_object']
        params_X = marginal_summary[asset1]['params']
        marginal_Y = marginal_summary[asset2]['best_dist_object']
        params_Y = marginal_summary[asset2]['params']
        copula = copula_summary[f'{asset1}-{asset2}']['best_copula_object']

        # Transform returns to uniforms
        u = marginal_X.cdf(returns[asset1], *params_X)
        v = marginal_Y.cdf(returns[asset2], *params_Y)

        copula_name = copula_summary[f'{asset1}-{asset2}']['best_copula_name']
        if copula_name == 'gaussian':
            mi = self.gaussian_conditional_prob(copula, u, v)
        elif copula_name == 't':
            mi = self.t_conditional_prob(copula, u, v)

        return mi
    
    def get_current_returns(self, symbols: list[str], data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame([{symbol: data[symbol]['close'].pct_change().dropna().iloc[-1] for symbol in symbols}])