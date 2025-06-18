import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from datetime import datetime
from typing import Dict
from scipy.optimize import minimize
from scipy import stats

class TawnType1Copula:
    def __init__(self, theta: float = 2.0, psi: float = 0.5, epsilon: float = 1e-8):
        self.theta = theta
        self.psi = psi
        self.epsilon = epsilon

    def fit(self, u: np.ndarray, v: np.ndarray) -> Dict:
        def neg_log_likelihood(params: np.ndarray) -> float:
            self.theta, self.psi = params
            pdf_vals = self.pdf(u, v)
            return -np.sum(np.log(np.maximum(pdf_vals, 1e-10)))
        
        result = minimize(neg_log_likelihood, [self.theta, self.psi],
                          bounds=[(1.001, 10), (0.001, 0.999)],
                          method='L-BFGS-B')
        self.theta, self.psi = result.x
        output = {
            'theta': self.theta,
            'psi': self.psi,
            'loglik': -result.fun
        }
        return output

    def pickands_function(self, t):
        """Pickands dependence function A(t)"""
        if abs(self.theta - 1) < 1e-10:
            return 1 - self.psi + self.psi
        return 1 - self.psi + self.psi * ((1-t)**self.theta + t**self.theta)**(1/self.theta)

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        log_u, log_v = np.log(u), np.log(v)
        log_uv = log_u + log_v
        
        t = log_v / log_uv
        A_t = self.pickands_function(t)
        return np.exp(log_uv * A_t)
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        ln_u = -np.log(u)
        ln_v = -np.log(v)
        
        A = (1 - self.psi)*ln_u**self.theta
        B = self.psi*ln_v**self.theta
        S = (A + B)**(1/self.theta)
        exp_S = np.exp(-S)
        
        # First derivatives
        dA_du = -self.theta*(1 - self.psi)*ln_u**(self.theta - 1)/u
        dB_dv = -self.theta*self.psi*ln_v**(self.theta - 1)/v
        
        # Second derivatives
        term1 = dA_du * dB_dv
        term2 = (S**(1 - 2*self.theta)) * (self.theta - 1 + S)
        
        return exp_S * term1 * term2
    
    def h_function(self, u, v, which=1):
        """Calculate h-functions using numerical differentiation"""
        epsilon = 1e-6
        
        if which == 1:  # h1(u|v) = ∂C(u,v)/∂v
            v_plus = np.clip(v + epsilon, epsilon, 1 - epsilon)
            v_minus = np.clip(v - epsilon, epsilon, 1 - epsilon)
            c_plus = self.cdf(u, v_plus)
            c_minus = self.cdf(u, v_minus)
            return (c_plus - c_minus) / (v_plus - v_minus)
        else:  # h2(v|u) = ∂C(u,v)/∂u
            u_plus = np.clip(u + epsilon, epsilon, 1 - epsilon)
            u_minus = np.clip(u - epsilon, epsilon, 1 - epsilon)
            c_plus = self.cdf(u_plus, v)
            c_minus = self.cdf(u_minus, v)
            return (c_plus - c_minus) / (u_plus - u_minus)
    
    def mispricing_index(self, u1, u2):
        """Calculate mispricing indices"""
        h1 = self.h_function(u1, u2, which=1)
        h2 = self.h_function(u2, u1, which=2)

        return h1, h2

class CopulaFitter:
    """
    Fits Gaussian and t copulas to uniform marginals and selects the best fit using AIC/BIC.
    """
    def __init__(self):
        self.copulae = {}
        self.copula_params = []
        self.summary = []

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

    def evaluate(self, u: np.ndarray, v: np.ndarray, pair: tuple[str, str]) -> Dict:
        c1, c2 = pair
        empirical_copula = self._empirical_copula(u, v)
        theoretical_copula = self.copulae[pair].cdf(u, v)
        cvm_stat = self._cvm_statistic(empirical_copula, theoretical_copula)

        results = {
            'c1': c1,
            'c2': c2,
            'theta': self.copulae[pair].theta,
            'psi': self.copulae[pair].psi,
            'cvm_statistic': cvm_stat
        }

        return results

    def fit_assets(self, selected_pairs: list, formation_data: Dict[str, pd.DataFrame], marginal_params: pd.DataFrame) -> Dict[str, Dict]:
        for pair in selected_pairs:
            c1, c2 = pair
            return1 = formation_data[c1]['close'].pct_change().dropna()
            return2 = formation_data[c2]['close'].pct_change().dropna()
            params1 = marginal_params[marginal_params['asset'] == c1]
            params2 = marginal_params[marginal_params['asset'] == c2]
            u = stats.t.cdf(return1,
                            params1['df'],
                            params1['loc'],
                            params1['scale'])
            v = stats.t.cdf(return2,
                            params2['df'],
                            params2['loc'],
                            params2['scale'])
            self.copulae[pair] = TawnType1Copula()
            self.copulae[pair].fit(u, v)
            self.copula_params.append({
                'c1': c1,
                'c2': c2,
                'theta': self.copulae[pair].theta,
                'psi': self.copulae[pair].psi
            })
            self.summary.append(self.evaluate(u, v, pair))

    def save_copula_params(self, file_extension: str = ''):
        pd.DataFrame(self.copula_params).to_csv(f'model_params/copula_params{file_extension}.csv', index=False)
        pd.DataFrame(self.summary).to_csv(f'model_params/copula_summary{file_extension}.csv', index=False)
            
def exp_smoothed_return(data: pd.Series, window: int = 12, alpha: float = 0.1) -> float:
    returns = data[-window:].pct_change()
    returns = returns.dropna()
    returns = returns.ewm(alpha=alpha).mean()
    return returns.iloc[-1]

if __name__ == "__main__":
    from collector import BybitDataCollector, BinanceDataCollector
    from config import Config

    # collector = BybitDataCollector()
    collector = BinanceDataCollector()
    marginal_p_values = pd.read_csv('model_params/marginal_p_values.csv')
    marginal_params = pd.read_csv('model_params/marginal_params.csv')
    coins = marginal_p_values[marginal_p_values['p_value'] > 0]['asset'].unique().tolist()
    
    formation_start_str = '2022-06-01 00:00:00'
    formation_end_str = '2022-12-31 23:59:59'
    formation_data = collector.get_multiple_symbols_data(symbols=coins, start_str=formation_start_str, end_str=formation_end_str)
    copula_fitter = CopulaFitter()
    
    from backtest import select_pairs
    selected_pairs = select_pairs(coins, formation_data)
    copula_fitter.fit_assets(selected_pairs, formation_data, marginal_params)
    copula_fitter.save_copula_params()