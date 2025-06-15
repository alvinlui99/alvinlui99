import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from copulae import GaussianCopula, StudentCopula

class CopulaFitter:
    """
    Fits Gaussian and t copulas to uniform marginals and selects the best fit using AIC/BIC.
    """
    def __init__(self):
        self.copula_families = ['gaussian', 't']
        self.copula_classes = {
            'gaussian': GaussianCopula,
            't': StudentCopula
        }

    def fit(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        fit_results = {}
        data = np.column_stack([u, v])
        for family in self.copula_families:
            copula_class = self.copula_classes[family]
            cop = copula_class()
            cop.fit(data)
            fit_results[family] = cop
        return fit_results

    def _empirical_copula(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Calculate the empirical copula for the data."""
        n = len(u)
        empirical = np.zeros(n)
        for i in range(n):
            empirical[i] = np.sum((u <= u[i]) & (v <= v[i])) / n
        return empirical

    def _cvm_statistic(self, empirical: np.ndarray, theoretical: np.ndarray) -> float:
        """Calculate the Cramer-von Mises statistic."""
        return np.sum((empirical - theoretical) ** 2)

    def evaluate(self, u: np.ndarray, v: np.ndarray, fit_results: Dict[str, Any]) -> pd.DataFrame:
        results = []
        n = len(u)
        data = np.column_stack([u, v])
        empirical_copula = self._empirical_copula(u, v)
        
        for family, cop in fit_results.items():
            try:
                # Calculate log-likelihood and information criteria
                loglik = np.sum(np.log(cop.pdf(data)))
                k = 2 if family == 't' else 1
                aic = 2 * k - 2 * loglik
                bic = k * np.log(n) - 2 * loglik
                
                # Calculate CVM statistic
                theoretical_copula = cop.cdf(data)
                cvm_stat = self._cvm_statistic(empirical_copula, theoretical_copula)
                
            except Exception as e:
                print(f"Error fitting {family}: {e}")
                aic = np.nan
                bic = np.nan
                cvm_stat = np.nan
                
            results.append({
                'copula_name': family,
                'copula': cop,
                'aic': aic,
                'bic': bic,
                'cvm_statistic': cvm_stat
            })
        return pd.DataFrame(results)

    def select_best(self, eval_df: pd.DataFrame, criterion: str = 'aic') -> Tuple[str, Dict[str, float]]:
        best_row = eval_df.loc[eval_df[criterion].idxmin()]
        return best_row

    def fit_assets(self, selected_pairs: list, marginal_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        summary = {}
        for c1, c2 in selected_pairs:
            u = marginal_summary[c1]['uniform']
            v = marginal_summary[c2]['uniform']
            fit_results = self.fit(u, v)
            eval_df = self.evaluate(u, v, fit_results)
            best_row = self.select_best(eval_df, criterion='aic')
            summary[f'{c1}-{c2}'] = {
                'best_copula_name': best_row['copula_name'],
                'best_copula_object': best_row['copula'],
                'best_row': best_row,
                'eval_df': eval_df
            }
        return summary