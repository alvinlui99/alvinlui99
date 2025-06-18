from statsmodels.tsa.stattools import coint
from scipy import stats
import pywt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from collector import BybitDataCollector, BinanceDataCollector
from copula_fitter import TawnType1Copula

class Backtest:
    def __init__(self):
        self.initial_balance = 1000000
        self.wallet_balance = self.initial_balance
        self.margin_balance = self.initial_balance
        self.pnls = []
        self.commissions = []
        self.equity_curve = [self.initial_balance]
        self.trades = []
        self.num_trades = 0

    def backtest_config(self,
               data: dict,
               selected_pairs: list[tuple[str, str]],
               marginal_params: pd.DataFrame,
               copula_params: pd.DataFrame):
        self.config = Config()
        self.data = data
        self.selected_pairs = selected_pairs
        self.marginal_params = marginal_params
        self.copula_params = copula_params
        self.copulae = {}
        for c1, c2 in selected_pairs:
            params = self.copula_params[(self.copula_params['c1'] == c1) & (self.copula_params['c2'] == c2)].to_dict('records')[0]
            self.copulae[f'{c1}-{c2}'] = TawnType1Copula(theta=params['theta'], psi=params['psi'])

        self.current_position = {}
        self.side = None
        self.stop_loss = None

    def exp_smoothed_return(self, data: pd.Series, alpha: float = 0.3) -> float:
        returns = data.pct_change().dropna()
        returns = returns.ewm(alpha=alpha).mean()
        return returns.iloc[-1]

    def mad(self, data) -> float:
        return np.median(np.abs(data - np.median(data)))

    def denoise_return(self, data: pd.Series, wavelet: str = 'db4', level: int = 5) -> float:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = self.mad(coeffs[-level])
        uthresh = sigma * np.sqrt(2*np.log(len(data)))
        coeffs[1:] = (pywt.threshold(i, value=uthresh) for i in coeffs[1:])
        denoised_data = pywt.waverec(coeffs, wavelet)
        return denoised_data[-1]

    def get_qty(self, prices1: pd.Series, prices2: pd.Series, margin_balance: float) -> float:
        hedge_ratio = np.sum(prices1 * prices2) / np.sum(prices2 ** 2)
        # hedge_ratio = np.polyfit(prices2, prices1, 1)[0]
        qty1 = margin_balance * self.config.investable_budget_pc / (prices1.iloc[-1] + prices2.iloc[-1] * hedge_ratio)
        qty2 = qty1 * hedge_ratio
        return round(qty1, 3), round(qty2, 3)

    def run(self, file_extension: str = ''):
        period_returns = []
        for pair in self.selected_pairs:
            self.__init__()
            result = []
            c1, c2 = pair
            timestamps = sorted(list(set.intersection(*[set(self.data[asset].index) for asset in [c1, c2]])))
            entry_price1 = 0
            entry_price2 = 0
            qty1 = 0
            qty2 = 0
            stop_loss = None
            side = None
            margin_balance = self.initial_balance
            wallet_balance = self.initial_balance
            for t in range(1, len(timestamps)):
                if t < self.config.ewm_window:
                    continue

                price1 = self.data[c1].loc[timestamps[t], 'close']
                price2 = self.data[c2].loc[timestamps[t], 'close']

                # Stop loss
                pnl = (price1 - entry_price1) * qty1 + (price2 - entry_price2) * qty2
                commission = 0
                if pnl < margin_balance * self.config.stop_loss_pc:
                    stop_loss = 'stop loss from ' + side
                    commission = self.config.commission_pc * (abs(qty1) * price1 + abs(qty2) * price2)
                    wallet_balance += pnl - commission
                    pnl = 0
                    side = None
                    qty1 = 0
                    qty2 = 0
                    entry_price1 = 0
                    entry_price2 = 0

                # Entry
                return1 = self.exp_smoothed_return(self.data[c1].loc[timestamps[t-self.config.ewm_window:t], 'close'])
                return2 = self.exp_smoothed_return(self.data[c2].loc[timestamps[t-self.config.ewm_window:t], 'close'])
                params1 = self.marginal_params[self.marginal_params['asset'] == c1].to_dict('records')[0]
                params2 = self.marginal_params[self.marginal_params['asset'] == c2].to_dict('records')[0]
                u1 = stats.t.cdf(return1, params1['df'], params1['loc'], params1['scale'])
                u2 = stats.t.cdf(return2, params2['df'], params2['loc'], params2['scale'])
                h1, _ = self.copulae[f'{c1}-{c2}'].mispricing_index(u1, u2)
                if h1 > self.config.long_threshold and side == None and stop_loss != 'stop loss from long':
                    # qty1 = round(margin_balance * self.config.investable_budget_pc * 0.5 / price1, 3)
                    # qty2 = -round(margin_balance * self.config.investable_budget_pc * 0.5 / price2, 3)
                    qty1, qty2 = self.get_qty(self.data[c1].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                              self.data[c2].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                              margin_balance)
                    qty2 *= -1
                    entry_price1 = price1
                    entry_price2 = price2
                    commission = self.config.commission_pc * (abs(qty1) * price1 + abs(qty2) * price2)
                    wallet_balance -= commission
                    side = 'long'
                    stop_loss = None
                elif h1 < self.config.short_threshold and side == None and stop_loss != 'stop loss from short':
                    # qty1 = -round(margin_balance * self.config.investable_budget_pc * 0.5 / price1, 3)
                    # qty2 = round(margin_balance * self.config.investable_budget_pc * 0.5 / price2, 3)
                    qty1, qty2 = self.get_qty(self.data[c1].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                              self.data[c2].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                              margin_balance)
                    qty1 *= -1
                    entry_price1 = price1
                    entry_price2 = price2
                    commission = self.config.commission_pc * (abs(qty1) * price1 + abs(qty2) * price2)
                    wallet_balance -= commission
                    side = 'short'
                    stop_loss = None
                # Exit
                elif h1 < self.config.long_exit_threshold and side == 'long':
                    commission = self.config.commission_pc * (abs(qty1) * price1 + abs(qty2) * price2)
                    wallet_balance += pnl - commission
                    pnl = 0
                    qty1 = 0
                    qty2 = 0
                    entry_price1 = 0
                    entry_price2 = 0
                    side = None
                elif h1 > self.config.short_exit_threshold and side == 'short':
                    commission = self.config.commission_pc * (abs(qty1) * price1 + abs(qty2) * price2)
                    wallet_balance += pnl - commission
                    pnl = 0
                    qty1 = 0
                    qty2 = 0
                    entry_price1 = 0
                    entry_price2 = 0
                    side = None
                margin_balance = wallet_balance + pnl

                output = {
                    'price1': price1,
                    'price2': price2,
                    'u1': u1,
                    'u2': u2,
                    'h1': h1,
                    'qty1': qty1,
                    'qty2': qty2,
                    'entry_price1': entry_price1,
                    'entry_price2': entry_price2,
                    'pnl': pnl,
                    'commission': commission,
                    'margin_balance': margin_balance,
                    'wallet_balance': wallet_balance,
                }
                result.append(output)
            period_returns.append(margin_balance/self.initial_balance - 1)
            pd.DataFrame(result).to_csv(f'backtest_results/{c1}-{c2}{file_extension}.csv', index=False)
        return np.mean(period_returns)
        
def select_pairs(coins: list[str], data: dict):
    pairs_with_stats = []
    for i in range(len(coins)):
        for j in range(i+1, len(coins)):
            c1, c2 = coins[i], coins[j]
            coint_t, _, _ = coint(data[c1]['close'], data[c2]['close'])
            pairs_with_stats.append((c1, c2, coint_t))
    
    pairs_with_stats.sort(key=lambda x: x[2], reverse=True)
    output = pd.DataFrame(pairs_with_stats, columns=['c1', 'c2', 't-stat'])
    output.to_csv('model_params/pairs_with_stats.csv', index=False)
    selected_pairs = [(c1, c2) for c1, c2, _ in pairs_with_stats[:Config().num_pairs]]
    return selected_pairs

def crypto_backtest(date_start_str: str, date_end_str: str, file_extension: str = ''):
    # collector = BybitDataCollector()
    collector = BinanceDataCollector()
    marginal_params = pd.read_csv(f'model_params/marginal_params{file_extension}.csv')
    copula_params = pd.read_csv(f'model_params/copula_params{file_extension}.csv')
    selected_pairs = list(zip(copula_params['c1'], copula_params['c2']))
    coins = list(set([coin for pair in selected_pairs for coin in pair]))

    backtest_data = collector.get_multiple_symbols_data(symbols=coins, start_str=date_start_str, end_str=date_end_str)

    backtest = Backtest()
    backtest.backtest_config(
        data=backtest_data,
        selected_pairs=selected_pairs,
        marginal_params=marginal_params,
        copula_params=copula_params
    )
    return backtest.run(file_extension=file_extension)

if __name__ == "__main__":
    # stock_backtest()
    crypto_backtest(date_start_str='2023-01-01 00:00:00', date_end_str='2023-06-30 23:59:59')