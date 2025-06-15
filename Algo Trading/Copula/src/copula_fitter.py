import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    def evaluate(self, u: np.ndarray, v: np.ndarray, pair: str) -> Dict:
        empirical_copula = self._empirical_copula(u, v)
        theoretical_copula = self.copulae[pair].cdf(u, v)
        cvm_stat = self._cvm_statistic(empirical_copula, theoretical_copula)

        results = {
            'copula_name': pair,
            'copula': self.copulae[pair],
            'theta': self.copulae[pair].theta,
            'psi': self.copulae[pair].psi,
            'cvm_statistic': cvm_stat
        }

        return results

    def fit_assets(self, selected_pairs: list, marginal_summary: Dict[str, Dict]) -> Dict[str, Dict]:
        summary = {}
        for c1, c2 in selected_pairs:
            pair = f'{c1}-{c2}'
            u = marginal_summary[c1]['uniform']
            v = marginal_summary[c2]['uniform']
            self.copulae[pair] = TawnType1Copula()
            self.copulae[pair].fit(u, v)
            summary[pair] = self.evaluate(u, v, pair)
        return summary
    
def exp_smoothed_return(data: pd.Series, window: int = 12, alpha: float = 0.1) -> float:
    returns = data[-window:].pct_change()
    returns = returns.dropna()
    returns = returns.ewm(alpha=alpha).mean()
    return returns.iloc[-1]

if __name__ == "__main__":
    from collector import BinanceDataCollector
    from config import Config
    from marginal_fitter import MarginalFitter
    from itertools import combinations

    start_time = datetime.now()
    coins = Config().coins
    collector = BinanceDataCollector()

    # # Create combined DataFrame
    combined_df = pd.DataFrame()
    
    formation_start_str = '2023-01-01 00:00:00'
    formation_end_str = '2023-06-30 23:59:59'
    backtest_start_str = '2023-07-01 00:00:00'
    backtest_end_str = '2023-12-31 23:59:59'

    formation_data = collector.get_multiple_symbols_data(symbols=coins, start_str=formation_start_str, end_str=formation_end_str)
    marginal_summary = MarginalFitter().fit_assets(formation_data, coins)
    df = pd.DataFrame(marginal_summary).T
    params_df = pd.DataFrame(df['params'].tolist(), index=df.index, columns=['shape', 'scale', 'location'])
    marginal_df = pd.concat([params_df, df['p_value']], axis=1)

    copula_fitter = CopulaFitter()
    selected_pairs = list(combinations(coins, 2))
    summary = copula_fitter.fit_assets(selected_pairs, marginal_summary)
    copula_df = pd.DataFrame(summary).T[['theta', 'psi', 'cvm_statistic']]

    backtest_data = collector.get_multiple_symbols_data(symbols=coins, start_str=backtest_start_str, end_str=backtest_end_str)
    
    counter = 0
    output = pd.DataFrame()
    for pair, copula in copula_fitter.copulae.items():
        counter += 1
        print(f'progress: {counter}/{len(copula_fitter.copulae)}')
        c1, c2 = pair.split('-')
        timestamps = sorted(list(set.intersection(*[set(backtest_data[asset].index) for asset in [c1, c2]])))
        return1_list = np.zeros(len(timestamps)-1)
        return2_list = np.zeros(len(timestamps)-1)
        u1_list = np.zeros(len(timestamps)-1)
        u2_list = np.zeros(len(timestamps)-1)
        h1_list = np.zeros(len(timestamps)-1)
        h2_list = np.zeros(len(timestamps)-1)
        for t in range(1, len(timestamps)):
            window = 13   # 13 hours
            if t < window:
                continue
            return1 = exp_smoothed_return(backtest_data[c1].loc[timestamps[t-window:t], 'close'])
            return2 = exp_smoothed_return(backtest_data[c2].loc[timestamps[t-window:t], 'close'])
            u1 = stats.t.cdf(return1, *marginal_summary[c1]['params'])
            u2 = stats.t.cdf(return2, *marginal_summary[c2]['params'])
            h1, h2 = copula.mispricing_index(u1, u2)
            return1_list[t-1] = return1
            return2_list[t-1] = return2
            u1_list[t-1] = u1
            u2_list[t-1] = u2
            h1_list[t-1] = h1
            h2_list[t-1] = h2

        output = pd.concat([output, pd.DataFrame({
            'pair': [pair] * (len(timestamps)-1),
            'timestamp': timestamps[1:],
            'return1': return1_list,
            'return2': return2_list,
            'u1': u1_list,
            'u2': u2_list,
            'h1': h1_list,
            'h2': h2_list
        })], ignore_index=True)
    output.to_csv('copula_mispricing_index.csv', index=False)
    print(f'Time taken: {datetime.now() - start_time}')