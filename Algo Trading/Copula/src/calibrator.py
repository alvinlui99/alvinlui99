import pandas as pd
from statsmodels.tsa.stattools import coint
from datetime import datetime

from marginal_fitter import MarginalFitter
from copula_fitter import CopulaFitter
from collector import BybitDataCollector, BinanceDataCollector
from backtest import Backtest
from config import Config

def remove_pairs(portfolio: dict):
    for pair in list(portfolio['pair'].keys()):
        if portfolio['pair'][pair]['side'] == 0 and portfolio['pair'][pair]['stop_loss'] == 0:
            portfolio['pair'].pop(pair)
    return portfolio

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

def update_portfolio(portfolio: dict, selected_pairs: list[tuple[str, str]]):
    for pair_name in list(portfolio['pair'].keys()):
        c1, c2 = pair_name.split('-')
        pair = (c1, c2)
        if portfolio['pair'][pair_name]['side'] == 0 and pair not in selected_pairs:
            portfolio['pair'].pop(pair_name)
    for pair in selected_pairs:
        c1, c2 = pair
        pair_name = f'{c1}-{c2}'
        if pair_name not in portfolio['pair']:
            portfolio['pair'][pair_name] = {
                'qty1': 0,
                'qty2': 0,
                'entry_price1': 0,
                'entry_price2': 0,
                'side': 0,
                'stop_loss': 0,
            }
    return portfolio

if __name__ == '__main__':
    coins = Config().coins
    collector = BinanceDataCollector()

    date_list = [
        # (formation_start_str, formation_end_str, backtest_start_str, backtest_end_str, file_extension)
        # ('2022-01-01 00:00:00','2022-01-31 23:59:59','2022-01-25 00:00:00','2022-02-28 23:59:59','_iter_1'),
        # ('2022-02-01 00:00:00','2022-02-28 23:59:59','2022-02-22 00:00:00','2022-03-31 23:59:59','_iter_2'),
        # ('2022-03-01 00:00:00','2022-03-31 23:59:59','2022-03-25 00:00:00','2022-04-30 23:59:59','_iter_3'),
        # ('2022-04-01 00:00:00','2022-04-30 23:59:59','2022-04-24 00:00:00','2022-05-31 23:59:59','_iter_4'),
        # ('2022-05-01 00:00:00','2022-05-31 23:59:59','2022-05-25 00:00:00','2022-06-30 23:59:59','_iter_5'),
        # ('2022-06-01 00:00:00','2022-06-30 23:59:59','2022-06-24 00:00:00','2022-07-31 23:59:59','_iter_6'),
        # ('2022-07-01 00:00:00','2022-07-31 23:59:59','2022-07-25 00:00:00','2022-08-31 23:59:59','_iter_7'),
        # ('2022-08-01 00:00:00','2022-08-31 23:59:59','2022-08-25 00:00:00','2022-09-30 23:59:59','_iter_8'),
        # ('2022-09-01 00:00:00','2022-09-30 23:59:59','2022-09-24 00:00:00','2022-10-31 23:59:59','_iter_9'),
        # ('2022-10-01 00:00:00','2022-10-31 23:59:59','2022-10-25 00:00:00','2022-11-30 23:59:59','_iter_10'),
        # ('2022-11-01 00:00:00','2022-11-30 23:59:59','2022-11-24 00:00:00','2022-12-31 23:59:59','_iter_11'),
        # ('2022-12-01 00:00:00','2022-12-31 23:59:59','2022-12-25 00:00:00','2023-01-31 23:59:59','_iter_12'),
        ('2023-01-01 00:00:00','2023-01-31 23:59:59','2023-01-25 00:00:00','2023-02-28 23:59:59','_iter_13'),
        ('2023-02-01 00:00:00','2023-02-28 23:59:59','2023-02-22 00:00:00','2023-03-31 23:59:59','_iter_14'),
        ('2023-03-01 00:00:00','2023-03-31 23:59:59','2023-03-25 00:00:00','2023-04-30 23:59:59','_iter_15'),
        ('2023-04-01 00:00:00','2023-04-30 23:59:59','2023-04-24 00:00:00','2023-05-31 23:59:59','_iter_16'),
        ('2023-05-01 00:00:00','2023-05-31 23:59:59','2023-05-25 00:00:00','2023-06-30 23:59:59','_iter_17'),
        ('2023-06-01 00:00:00','2023-06-30 23:59:59','2023-06-24 00:00:00','2023-07-31 23:59:59','_iter_18'),
        ('2023-07-01 00:00:00','2023-07-31 23:59:59','2023-07-25 00:00:00','2023-08-31 23:59:59','_iter_19'),
        ('2023-08-01 00:00:00','2023-08-31 23:59:59','2023-08-25 00:00:00','2023-09-30 23:59:59','_iter_20'),
        ('2023-09-01 00:00:00','2023-09-30 23:59:59','2023-09-24 00:00:00','2023-10-31 23:59:59','_iter_21'),
        ('2023-10-01 00:00:00','2023-10-31 23:59:59','2023-10-25 00:00:00','2023-11-30 23:59:59','_iter_22'),
        ('2023-11-01 00:00:00','2023-11-30 23:59:59','2023-11-24 00:00:00','2023-12-31 23:59:59','_iter_23'),
        ('2023-12-01 00:00:00','2023-12-31 23:59:59','2023-12-25 00:00:00','2024-01-31 23:59:59','_iter_24'),
        ('2024-01-01 00:00:00','2024-01-31 23:59:59','2024-01-25 00:00:00','2024-02-29 23:59:59','_iter_25'),
        ('2024-02-01 00:00:00','2024-02-29 23:59:59','2024-02-23 00:00:00','2024-03-31 23:59:59','_iter_26'),
        ('2024-03-01 00:00:00','2024-03-31 23:59:59','2024-03-25 00:00:00','2024-04-30 23:59:59','_iter_27'),
        ('2024-04-01 00:00:00','2024-04-30 23:59:59','2024-04-24 00:00:00','2024-05-31 23:59:59','_iter_28'),
        ('2024-05-01 00:00:00','2024-05-31 23:59:59','2024-05-25 00:00:00','2024-06-30 23:59:59','_iter_29'),
        ('2024-06-01 00:00:00','2024-06-30 23:59:59','2024-06-24 00:00:00','2024-07-31 23:59:59','_iter_30'),
        ('2024-07-01 00:00:00','2024-07-31 23:59:59','2024-07-25 00:00:00','2024-08-31 23:59:59','_iter_31'),
        ('2024-08-01 00:00:00','2024-08-31 23:59:59','2024-08-25 00:00:00','2024-09-30 23:59:59','_iter_32'),
        ('2024-09-01 00:00:00','2024-09-30 23:59:59','2024-09-24 00:00:00','2024-10-31 23:59:59','_iter_33'),
        ('2024-10-01 00:00:00','2024-10-31 23:59:59','2024-10-25 00:00:00','2024-11-30 23:59:59','_iter_34'),
        ('2024-11-01 00:00:00','2024-11-30 23:59:59','2024-11-24 00:00:00','2024-12-31 23:59:59','_iter_35'),
        ('2024-12-01 00:00:00','2024-12-31 23:59:59','2024-12-25 00:00:00','2025-01-31 23:59:59','_iter_36'),
        ('2025-01-01 00:00:00','2025-01-31 23:59:59','2025-01-25 00:00:00','2025-02-28 23:59:59','_iter_37'),
        ('2025-02-01 00:00:00','2025-02-28 23:59:59','2025-02-22 00:00:00','2025-03-31 23:59:59','_iter_38'),
        ('2025-03-01 00:00:00','2025-03-31 23:59:59','2025-03-25 00:00:00','2025-04-30 23:59:59','_iter_39'),
        ('2025-04-01 00:00:00','2025-04-30 23:59:59','2025-04-24 00:00:00','2025-05-31 23:59:59','_iter_40')
    ]
    current_num_pairs = 0
    balance = 1000000
    portfolio = {
        'wallet_balance': balance,
        'margin_balance': balance,
        'asset': {
            coin: 0 for coin in coins
        },
        'pair': {}
    }
    equity_curve = [balance]
    
    for formation_start_str, formation_end_str, backtest_start_str, backtest_end_str, file_extension in date_list:
        start_time = datetime.now()
        data = collector.get_multiple_symbols_data(symbols=coins, start_str=formation_start_str, end_str=formation_end_str)
        selected_pairs = select_pairs(coins, data)
        portfolio = update_portfolio(portfolio, selected_pairs)

        for pair in portfolio['pair'].keys():
            print(pair)

        break

        marginal_fitter = MarginalFitter()
        marginal_fitter.fit_assets(data, coins, returns_col='close')
        marginal_fitter.save_marginal_params(file_extension)
        marginal_params = pd.read_csv(f'model_params/marginal_params{file_extension}.csv')

        copula_fitter = CopulaFitter()
        copula_fitter.fit_assets(selected_pairs, data, marginal_params)
        copula_fitter.save_copula_params(file_extension)
        copula_params = pd.read_csv(f'model_params/copula_params{file_extension}.csv')
        backtest = Backtest()
        backtest.backtest_config(
            data=data,
            selected_pairs=selected_pairs,
            marginal_params=marginal_params,
            copula_params=copula_params,
            backtest_start_str=backtest_start_str,
            backtest_end_str=backtest_end_str,
            file_extension=file_extension,
            portfolio=portfolio
        )
        portfolio = backtest.run()
        equity_curve.append(portfolio['margin_balance'])
        end_time = datetime.now()
        print(f'{file_extension} margin_balance: {portfolio["margin_balance"]:.2f}')
        print(f'Time taken: {end_time - start_time}')
    
    pd.DataFrame(equity_curve).to_csv('backtest_results/equity_curve.csv', index=False)