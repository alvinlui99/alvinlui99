import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple

class MarginalFitter:
    """
    Fits multiple candidate distributions to asset returns and selects the best fit
    using KS test, Anderson-Darling test, AIC, and BIC.
    """
    def __init__(self):
        self.candidate_distributions = [
            't', 'norm', 'genextreme', 'logistic', 'genpareto', 'laplace'
        ]

    def fit(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Fit all candidate distributions to the data.
        Args:
            data: 1D array of returns
        Returns:
            Dictionary with distribution name as key and fit result as value
        """
        fit_results = {}
        for dist_name in self.candidate_distributions:
            dist = getattr(stats, dist_name)
            try:
                params = dist.fit(data)
                fit_results[dist_name] = params
            except Exception as e:
                fit_results[dist_name] = None
        return fit_results

    def evaluate(self, data: np.ndarray, fit_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate each fitted distribution using KS, AD, AIC, and BIC.
        Args:
            data: 1D array of returns
            fit_results: output from self.fit
        Returns:
            DataFrame with metrics for each distribution
        """
        results = []
        n = len(data)
        for dist_name, params in fit_results.items():
            if params is None:
                continue
            dist = getattr(stats, dist_name)
            # KS test
            ks_stat, ks_p = stats.kstest(data, dist_name, args=params)
            # Anderson-Darling test (only for some dists)
            try:
                ad_result = stats.anderson(data, dist=dist_name)
                ad_stat = ad_result.statistic
            except Exception:
                ad_stat = np.nan
            # Log-likelihood
            loglik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik
            bic = k * np.log(n) - 2 * loglik
            results.append({
                'distribution_name': dist_name,
                'distribution_object': dist,
                'params': params,
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'ad_stat': ad_stat,
                'aic': aic,
                'bic': bic
            })
        return pd.DataFrame(results)

    def select_best(self, eval_df: pd.DataFrame, criterion: str = 'aic') -> Tuple[str, Dict[str, float]]:
        """
        Select the best distribution based on the given criterion (AIC or BIC).
        Args:
            eval_df: DataFrame from self.evaluate
            criterion: 'aic' or 'bic'
        Returns:
            (best distribution name, row as dict)
        """
        best_row = eval_df.loc[eval_df[criterion].idxmin()]
        return best_row

    def fit_assets(self, data_dict: Dict[str, pd.DataFrame], asset_names: list, returns_col: str = 'close', criterion: str = 'aic', pvalue_threshold: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        For each asset, compute returns, fit, evaluate, and select the best distribution.
        Args:
            data_dict: dict of {asset: DataFrame}
            asset_names: list of asset names to process
            returns_col: column to use for returns calculation (default 'close')
            criterion: 'aic' or 'bic' for best fit selection
            pvalue_threshold: threshold for KS p-value to flag poor fit (default 0.05)
        Returns:
            Dictionary {asset: {'best_dist': str, 'best_row': dict, 'eval_df': DataFrame, 'all_p_below_threshold': bool}}
        """
        summary = {}
        for asset in asset_names:
            df = data_dict[asset]
            returns = df[returns_col].pct_change().dropna().values
            fit_results = self.fit(returns)
            eval_df = self.evaluate(returns, fit_results)
            best_row = self.select_best(eval_df, criterion=criterion)
            all_p_below = (eval_df['ks_p'] < pvalue_threshold).all()
            if all_p_below:
                print(f"Warning: No good fit for {asset} (all KS p < {pvalue_threshold})")
            # Compute uniform marginals using the best-fit CDF
            dist = best_row['distribution_object']
            params = best_row['params']
            uniform = dist.cdf(returns, *params)
            summary[asset] = {
                'best_dist_name': best_row['distribution_name'],
                'best_dist_object': best_row['distribution_object'],
                'best_row': best_row,
                'eval_df': eval_df,
                'all_p_below_threshold': all_p_below,
                'returns': returns,
                'uniform': uniform,
                'params': params
            }
            print(f"Fitted {asset} with marginal distribution {best_row['distribution_name']} with p-value {best_row['ks_p']:.3f}")
        return summary