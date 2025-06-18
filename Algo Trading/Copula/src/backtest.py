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
               copula_params: pd.DataFrame,
               backtest_start_str: str,
               backtest_end_str: str,
               file_extension: str,
               portfolio: dict):
        self.config = Config()
        self.data = data
        self.selected_pairs = selected_pairs
        self.marginal_params = marginal_params
        self.copula_params = copula_params
        self.copulae = {}
        for c1, c2 in selected_pairs:
            params = self.copula_params[(self.copula_params['c1'] == c1) & (self.copula_params['c2'] == c2)].to_dict('records')[0]
            self.copulae[f'{c1}-{c2}'] = TawnType1Copula(theta=params['theta'], psi=params['psi'])

        self.backtest_start_str = backtest_start_str
        self.backtest_end_str = backtest_end_str
        self.file_extension = file_extension
        self.portfolio = portfolio
        # portfolio: {
        #     'wallet_balance': float,                 # wallet balance does not include unrealised pnl
        #     'margin_balance': float,                 # margin balance includes unrealised pnl
        #     'asset': {
        #         'symbol': float                      # symbol: qty
        #     }
        #     'pair': {
        #         'pair_name': {
        #             'qty1': float,
        #             'qty2': float,
        #             'entry_price1': float,
        #             'entry_price2': float,
        #             'side': int, # 1: long, -1: short
        #             'stop_loss': int, # 1: stop loss, do not enter long, -1: stop loss, do not enter short, 0: no stop loss
        #         }
        #     }
        # }

    def exp_smoothed_return(self, data: pd.Series, alpha: float = 0.3) -> float:
        returns = data.pct_change().dropna()
        returns = returns.ewm(alpha=alpha).mean()
        return returns.iloc[-1]

    def get_qty(self, prices1: pd.Series, prices2: pd.Series, margin_balance: float) -> float:
        hedge_ratio = np.sum(prices1 * prices2) / np.sum(prices2 ** 2)
        # hedge_ratio = np.polyfit(prices2, prices1, 1)[0]
        qty1 = margin_balance * self.config.investable_budget_pc / (prices1.iloc[-1] + prices2.iloc[-1] * hedge_ratio)
        qty2 = qty1 * hedge_ratio
        return round(qty1, 3), round(qty2, 3)

    def update_margin_balance(self, current_price_dict: dict) -> None:
        self.portfolio['margin_balance'] = self.portfolio['wallet_balance']
        for pair in self.portfolio['pair']:
            c1, c2 = pair.split('-')
            price1 = current_price_dict[c1]
            price2 = current_price_dict[c2]
            entry_price1 = self.portfolio['pair'][pair]['entry_price1']
            entry_price2 = self.portfolio['pair'][pair]['entry_price2']
            qty1 = self.portfolio['pair'][pair]['qty1']
            qty2 = self.portfolio['pair'][pair]['qty2']
            self.portfolio['margin_balance'] += (price1 - entry_price1) * qty1 + (price2 - entry_price2) * qty2

    def execute_netting(self, netting: dict, current_price_dict: dict) -> None:
        for asset, qty in netting.items():
            current_price = current_price_dict[asset]
            commission = self.config.commission_pc * (abs(qty) * current_price)
            self.portfolio['asset'][asset] += qty
            self.portfolio['wallet_balance'] -= commission

    def run(self):
        timestamps = sorted(list(set.intersection(*[set(self.data[asset].index) for pair in self.selected_pairs for asset in pair])))
        for t in range(self.config.ewm_window, len(timestamps)):
            current_price_dict = {asset: self.data[asset].loc[timestamps[t], 'close'] for asset in self.data.keys()}
            self.update_margin_balance(current_price_dict)
            netting = {
                asset: 0 for asset in self.data.keys()
            }
            for pair in self.selected_pairs:
                c1, c2 = pair
                pair_name = f'{c1}-{c2}'
                price1 = current_price_dict[c1]
                price2 = current_price_dict[c2]
                return1 = self.exp_smoothed_return(self.data[c1].loc[timestamps[t-self.config.ewm_window:t], 'close'])
                return2 = self.exp_smoothed_return(self.data[c2].loc[timestamps[t-self.config.ewm_window:t], 'close'])
                params1 = self.marginal_params[self.marginal_params['asset'] == c1].to_dict('records')[0]
                params2 = self.marginal_params[self.marginal_params['asset'] == c2].to_dict('records')[0]
                u1 = stats.t.cdf(return1, params1['df'], params1['loc'], params1['scale'])
                u2 = stats.t.cdf(return2, params2['df'], params2['loc'], params2['scale'])
                h1, _ = self.copulae[f'{c1}-{c2}'].mispricing_index(u1, u2)
                if self.portfolio['pair'][pair_name]['side'] != 0:      # If there is existing position
                    side = self.portfolio['pair'][pair_name]['side']
                    qty1 = self.portfolio['pair'][pair_name]['qty1']
                    qty2 = self.portfolio['pair'][pair_name]['qty2']
                    entry_price1 = self.portfolio['pair'][pair_name]['entry_price1']
                    entry_price2 = self.portfolio['pair'][pair_name]['entry_price2']
                    pnl = (price1 - entry_price1) * qty1 + (price2 - entry_price2) * qty2
                    # if False:
                    #     pass
                    if pnl < self.portfolio['margin_balance'] * self.config.stop_loss_pc:       # If stop loss, close position and set stop_loss
                        self.portfolio['pair'][pair_name] = {
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0,
                            'side': 0,
                            'stop_loss': side,                                      # If stop loss from long, set stop_loss to 1, do not enter long again. If stop loss from short, set stop_loss to -1, do not enter short again
                        }
                        # realise pnl on wallet balance
                        self.portfolio['wallet_balance'] += pnl

                        # leave commission to netting
                        netting[c1] -= qty1
                        netting[c2] -= qty2
                        # print(f'{pair_name} stop loss: {pnl:.2f}')
                    else:                                                                       # If not stop loss, check for exit
                        if side == 1 and h1 < self.config.long_exit_threshold:
                            self.portfolio['pair'][pair_name] = {
                                'qty1': 0,
                                'qty2': 0,
                                'entry_price1': 0,
                                'entry_price2': 0,
                                'side': 0,
                                'stop_loss': 0,
                            }
                            # realise pnl on wallet balance
                            self.portfolio['wallet_balance'] += pnl

                            # leave commission to netting
                            netting[c1] -= qty1
                            netting[c2] -= qty2
                            # print(f'{pair_name} long exit: {pnl:.2f}. Entry price 1: {entry_price1:.2f}. Entry price 2: {entry_price2:.2f}. Exit price 1: {price1:.2f}. Exit price 2: {price2:.2f}. Entry Size: {abs(qty1) * entry_price1 + abs(qty2) * entry_price2:.2f}.')
                        elif side == -1 and h1 > self.config.short_exit_threshold:
                            self.portfolio['pair'][pair_name] = {
                                'qty1': 0,
                                'qty2': 0,
                                'entry_price1': 0,
                                'entry_price2': 0,
                                'side': 0,
                                'stop_loss': 0,
                            }
                            # realise pnl on wallet balance
                            self.portfolio['wallet_balance'] += pnl

                            # leave commission to netting
                            netting[c1] -= qty1
                            netting[c2] -= qty2
                            # print(f'{pair_name} short exit: {pnl:.2f}. Entry price 1: {entry_price1:.2f}. Entry price 2: {entry_price2:.2f}. Exit price 1: {price1:.2f}. Exit price 2: {price2:.2f}. Entry Size: {abs(qty1) * entry_price1 + abs(qty2) * entry_price2:.2f}.')
                else:       # If no existing position, check for entry
                    if h1 > self.config.long_threshold and self.portfolio['pair'][pair_name]['stop_loss'] != 1:
                        qty1, qty2 = self.get_qty(self.data[c1].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                                  self.data[c2].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                                  self.portfolio['margin_balance'])
                        qty2 *= -1
                        self.portfolio['pair'][pair_name] = {
                            'qty1': qty1,
                            'qty2': qty2,
                            'entry_price1': price1,
                            'entry_price2': price2,
                            'side': 1,
                            'stop_loss': 0,
                        }
                        netting[c1] += qty1
                        netting[c2] += qty2
                    elif h1 < self.config.short_threshold and self.portfolio['pair'][pair_name]['stop_loss'] != -1:
                        qty1, qty2 = self.get_qty(self.data[c1].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                                  self.data[c2].loc[timestamps[t-self.config.ewm_window:t], 'close'],
                                                  self.portfolio['margin_balance'])
                        qty1 *= -1
                        self.portfolio['pair'][pair_name] = {
                            'qty1': qty1,
                            'qty2': qty2,
                            'entry_price1': price1,
                            'entry_price2': price2,
                            'side': -1,
                            'stop_loss': 0,
                        }
                        netting[c1] += qty1
                        netting[c2] += qty2
            self.execute_netting(netting, current_price_dict)
            self.update_margin_balance(current_price_dict)
        return self.portfolio


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
    crypto_backtest(date_start_str='2023-01-01 00:00:00', date_end_str='2023-06-30 23:59:59')