from statsmodels.tsa.stattools import coint
from scipy import stats
import pywt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from collector import BinanceDataCollector
from copula_fitter import CopulaFitter
from marginal_fitter import MarginalFitter

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
               marginal_summary: dict,
               copula_fitter: CopulaFitter):
        self.config = Config()
        self.data = data
        self.selected_pairs = selected_pairs
        self.marginal_summary = marginal_summary
        self.copula_fitter = copula_fitter
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
        hedge_ratio = np.polyfit(prices2, prices1, 1)[0]
        qty1 = margin_balance * self.config.investable_budget_pc / (prices1.iloc[-1] + prices2.iloc[-1] * hedge_ratio)
        qty2 = qty1 * hedge_ratio
        return round(qty1, 3), round(qty2, 3)

    def run(self):
        ending_balances = []
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
                u1 = stats.t.cdf(return1, *self.marginal_summary[c1]['params'])
                u2 = stats.t.cdf(return2, *self.marginal_summary[c2]['params'])
                h1, _ = self.copula_fitter.copulae[f'{c1}-{c2}'].mispricing_index(u1, u2)
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
            ending_balances.append((f'{c1}-{c2}', margin_balance))
            pd.DataFrame(result).to_csv(f'backtest_results/{c1}-{c2}.csv', index=False)
        pd.DataFrame(ending_balances, columns=['pair', 'ending_balance']).to_csv('backtest_results/ending_balances.csv', index=False)
        
def select_pairs(coins: list[str], data: dict):
    pairs_with_stats = []
    for i in range(len(coins)):
        for j in range(i+1, len(coins)):
            c1, c2 = coins[i], coins[j]
            coint_t, p_value, _ = coint(data[c1]['close'], data[c2]['close'])
            if coint_t > Config().coint_threshold:
                pairs_with_stats.append((c1, c2, coint_t, p_value))
    
    pairs_with_stats.sort(key=lambda x: x[2], reverse=True)
    # selected_pairs = [(c1, c2) for c1, c2, _, _ in pairs_with_stats]
    output = pd.DataFrame(pairs_with_stats, columns=['c1', 'c2', 't-stat', 'p-value'])
    output.to_csv('backtest_results/pairs_with_stats.csv', index=False)
    selected_pairs = [(c1, c2) for c1, c2, _, _ in pairs_with_stats[:Config().num_pairs]]
    return selected_pairs

def crypto_backtest():
    collector = BinanceDataCollector()
    
    # formation_start_str = '2022-01-01 00:00:00'
    # formation_end_str = '2023-12-31 23:59:59'
    backtest_start_str = '2024-01-01 00:00:00'
    backtest_end_str = '2024-12-31 23:59:59'
    
    coins = Config().coins

    # formation_data = collector.get_multiple_symbols_data(symbols=coins, start_str=formation_start_str, end_str=formation_end_str)
    # marginal_summary = MarginalFitter().fit_assets(formation_data, coins)
    # copula_fitter = CopulaFitter()
    
    selected_pairs = select_pairs(coins, formation_data)
    # copula_summary = copula_fitter.fit_assets(selected_pairs, marginal_summary)
    
    backtest_data = collector.get_multiple_symbols_data(symbols=coins, start_str=backtest_start_str, end_str=backtest_end_str)

    backtest = Backtest()
    backtest.backtest_config(
        data=backtest_data,
        selected_pairs=selected_pairs,
        marginal_summary=marginal_summary,
        copula_fitter=copula_fitter
    )
    results = backtest.run()

def stock_backtest():
    import yfinance as yf
    
    collector = BinanceDataCollector()
    
    formation_start_str = '2022-01-01'
    formation_end_str = '2022-12-31'
    backtest_start_str = '2023-01-01'
    backtest_end_str = '2023-12-31'
    
    stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA',
              'NVDA', 'META', 'NFLX', 'CSCO', 'INTC',
              'QCOM', 'ORCL', 'IBM', 'SAP', 'IBM']
    data = yf.download(stocks, start='2022-01-01', end='2022-12-31')
    
    # Transform data into dictionary of DataFrames
    data_dict = {}
    for stock in stocks:
        data_dict[stock] = pd.DataFrame({
            'open': data['Open'][stock],
            'high': data['High'][stock],
            'low': data['Low'][stock],
            'close': data['Close'][stock],
            'volume': data['Volume'][stock]
        })
    
    print("Data structure:", {k: v.shape for k, v in data_dict.items()})
    print("\nSample data for first stock:", data_dict[stocks[0]].head())
    
    marginal_summary = MarginalFitter().fit_assets(data_dict, stocks)
    copula_fitter = CopulaFitter()
    selected_pairs = select_pairs(stocks, data_dict)
    copula_summary = copula_fitter.fit_assets(selected_pairs, marginal_summary)
    
    backtest_data = yf.download(stocks, start=backtest_start_str, end=backtest_end_str)

    backtest = Backtest()
    backtest.backtest_config(
        data=backtest_data,
        selected_pairs=selected_pairs,
        marginal_summary=marginal_summary,
        copula_fitter=copula_fitter
    )
    results = backtest.run()

if __name__ == "__main__":
    # stock_backtest()
    crypto_backtest()