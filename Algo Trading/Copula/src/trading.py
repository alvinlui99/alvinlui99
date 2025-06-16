import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

from binance.um_futures import UMFutures

from collector import BinanceDataCollector
from copula_fitter import TawnType1Copula
from statsmodels.tsa.stattools import coint
from config import Config

class Trader:
    def __init__(self,
                 selected_pairs: list[str],
                 marginal_params: pd.DataFrame,
                 copula_params: pd.DataFrame,
                 api_key: str = None, 
                 api_secret: str = None,
                 base_url: str = 'https://testnet.binancefuture.com'):
        if api_key is None:
            api_key = os.getenv('BINANCE_API_KEY')
        if api_secret is None:
            api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
        self.collector = BinanceDataCollector()
        self.selected_pairs = []
        for pair in selected_pairs:
            c1, c2 = pair.split('-')
            self.selected_pairs.append((c1, c2))
        self.marginal_params = marginal_params
        self.copula_params = copula_params
        self.copulae = {}
        for pair in selected_pairs:
            params = self.copula_params[self.copula_params['pair'] == pair].to_dict('records')[0]
            self.copulae[pair] = TawnType1Copula(theta=params['theta'], psi=params['psi'])
        
        self.config = Config()

    def get_price_history(self, symbol: str, interval: str = '1h') -> pd.Series:
        klines = self.client.mark_price_klines(symbol=symbol, interval=interval)
        klines = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'ignore_1', 'close_time', 'ignore_2', 'ignore_3', 'ignore_4', 'ignore_5', 'ignore_6'])
        klines['close'] = klines['close'].astype(float)
        klines = klines.set_index('timestamp')
        return klines['close']

    def read_positions(self) -> pd.DataFrame:
        return pd.read_csv(f'trading_results/positions.csv')

    def write_positions(self, positions: pd.DataFrame):
        '''
        pair: c1-c2
        qty1: quantity of c1
        qty2: quantity of c2
        entry_price1: entry price of c1
        entry_price2: entry price of c2
        margin_balance: margin balance
        wallet_balance: wallet balance
        stop_loss: 1 for stop loss from long, -1 for stop loss from short, 0 for no position
        side: 1 for long, -1 for short, 0 for no position
        '''
        positions.to_csv(f'trading_results/positions.csv', index=False)

    def exp_smoothed_return(self, data: pd.Series, alpha: float = 0.3) -> float:
        returns = data.pct_change().dropna()
        returns = returns.ewm(alpha=alpha).mean()
        return returns.iloc[-1]
    
    def get_qty(self, c1:str, c2:str, prices1: pd.Series, prices2: pd.Series, price1: float, price2: float, margin_balance: float) -> float:
        hedge_ratio = np.polyfit(prices2, prices1, 1)[0]
        qty1 = margin_balance * self.config.investable_budget_pc * self.config.leverage / (price1 + price2 * hedge_ratio)
        qty2 = qty1 * hedge_ratio
        return round(qty1, self.get_precision(c1)), round(qty2, self.get_precision(c2))

    def get_precision(self, symbol: str) -> int:
        info = self.client.exchange_info()
        for d in info['symbols']:
            if d['symbol'] == symbol:
                return d['quantityPrecision']

    def run(self):            
        while True:
            positions = self.read_positions()
            netting = {c: 0 for c in self.config.coins}         # accumulate order size for each coin
            for pair in self.selected_pairs:
                print(f'Processing pair: {pair}')
                c1, c2 = pair
                position = positions.loc[positions['pair'] == f'{c1}-{c2}'].to_dict('records')[0]
                price1 = float(self.client.mark_price(symbol=c1)['markPrice'])
                price2 = float(self.client.mark_price(symbol=c2)['markPrice'])
                pnl = (price1 - position['entry_price1']) * position['qty1'] + (price2 - position['entry_price2']) * position['qty2']
                if pnl < position['margin_balance'] * self.config.stop_loss_pc:
                    netting[c1] -= position['qty1']         # Stop loss
                    netting[c2] -= position['qty2']         # Stop loss
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'stop_loss'] = 1 if position['qty1'] > 0 else -1
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty1'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty2'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price1'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price2'] = 0
                    self.write_positions(positions)
                    print(f'Stop loss triggered for pair: {pair}')
                price_history1 = self.get_price_history(c1)
                price_history2 = self.get_price_history(c2)
                return1 = self.exp_smoothed_return(price_history1.iloc[-self.config.ewm_window:])
                return2 = self.exp_smoothed_return(price_history2.iloc[-self.config.ewm_window:])
                params1 = self.marginal_params[self.marginal_params['asset'] == c1].to_dict('records')[0]
                params2 = self.marginal_params[self.marginal_params['asset'] == c2].to_dict('records')[0]
                u1 = stats.t.cdf(return1, params1['df'], params1['loc'], params1['scale'])
                u2 = stats.t.cdf(return2, params2['df'], params2['loc'], params2['scale'])

                h1, _ = self.copulae[f'{c1}-{c2}'].mispricing_index(u1, u2)
                
                # Log h1 value
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_log = pd.DataFrame({
                    'timestamp': [current_time],
                    'pair': [f'{c1}-{c2}'],
                    'h1': [h1]
                })
                new_log.to_csv('trading_results/h1_logs.csv', mode='a', header=False, index=False)

                if h1 > self.config.long_threshold and position['stop_loss'] != 1 and position['side'] == 0:
                    qty1, qty2 = self.get_qty(c1, c2,
                                              price_history1.iloc[-self.config.ewm_window:],
                                              price_history2.iloc[-self.config.ewm_window:],
                                              price1, price2, position['margin_balance'])
                    qty2 *= -1
                    entry_price1 = price1
                    entry_price2 = price2
                    side = 1
                    stop_loss = 0
                    commission = self.config.commission_pc * (abs(qty1) * price1 + abs(qty2) * price2)
                    wallet_balance = position['wallet_balance'] - commission
                    margin_balance = wallet_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty1'] = qty1
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty2'] = qty2
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price1'] = entry_price1
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price2'] = entry_price2
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'margin_balance'] = margin_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'wallet_balance'] = wallet_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'stop_loss'] = stop_loss
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'side'] = side
                    netting[c1] += qty1
                    netting[c2] += qty2
                    print(f'Long position opened for pair: {pair}')
                elif h1 < self.config.short_threshold and position['stop_loss'] != -1 and position['side'] == 0:
                    qty1, qty2 = self.get_qty(c1, c2,
                                              price_history1.iloc[-self.config.ewm_window:],
                                              price_history2.iloc[-self.config.ewm_window:],
                                              price1, price2, position['margin_balance'])
                    qty1 *= -1
                    entry_price1 = price1
                    entry_price2 = price2
                    side = -1
                    stop_loss = 0
                    commission = self.config.commission_pc * (abs(qty1) * price1 + abs(qty2) * price2)
                    wallet_balance = position['wallet_balance'] - commission
                    margin_balance = wallet_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty1'] = qty1
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty2'] = qty2
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price1'] = entry_price1
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price2'] = entry_price2
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'margin_balance'] = margin_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'wallet_balance'] = wallet_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'stop_loss'] = stop_loss
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'side'] = side
                    netting[c1] += qty1
                    netting[c2] += qty2
                    print(f'Short position opened for pair: {pair}')
                elif h1 < self.config.long_exit_threshold and position['side'] == 1:
                    commission = self.config.commission_pc * (abs(position['qty1']) * price1 + abs(position['qty2']) * price2)
                    netting[c1] -= position['qty1']
                    netting[c2] -= position['qty2']
                    wallet_balance = position['wallet_balance'] + pnl - commission
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'wallet_balance'] = wallet_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'pnl'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty1'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty2'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price1'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price2'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'stop_loss'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'side'] = 0
                    print(f'Long position closed for pair: {pair}')
                elif h1 > self.config.short_exit_threshold and position['side'] == -1:
                    commission = self.config.commission_pc * (abs(position['qty1']) * price1 + abs(position['qty2']) * price2)
                    netting[c1] -= position['qty1']
                    netting[c2] -= position['qty2']
                    wallet_balance = position['wallet_balance'] + pnl - commission
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'wallet_balance'] = wallet_balance
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'pnl'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty1'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'qty2'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price1'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'entry_price2'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'stop_loss'] = 0
                    positions.loc[positions['pair'] == f'{c1}-{c2}', 'side'] = 0
                    print(f'Short position closed for pair: {pair}')
                self.write_positions(positions)
            for c in netting:
                if netting[c] > 0:
                    self.client.new_order(symbol=c, side='BUY', quantity=netting[c], type='MARKET')
                elif netting[c] < 0:
                    self.client.new_order(symbol=c, side='SELL', quantity=abs(netting[c]), type='MARKET')
            time.sleep(60)

def select_pairs(coins: list[str], data: dict):
    pairs_with_stats = []
    for i in range(len(coins)):
        for j in range(i+1, len(coins)):
            c1, c2 = coins[i], coins[j]
            coint_t, p_value, _ = coint(data[c1]['close'], data[c2]['close'])
            if coint_t > Config().coint_threshold:
                pairs_with_stats.append((c1, c2, coint_t, p_value))
    
    pairs_with_stats.sort(key=lambda x: x[2], reverse=True)
    output = pd.DataFrame(pairs_with_stats, columns=['c1', 'c2', 't-stat', 'p-value'])
    output.to_csv('trading_results/pairs_with_stats.csv', index=False)
    selected_pairs = [(c1, c2) for c1, c2, _, _ in pairs_with_stats[:Config().num_pairs]]
    return selected_pairs

if __name__ == '__main__':
    positions = pd.read_csv('trading_results/positions.csv')
    selected_pairs = positions['pair'].unique().tolist()
    marginal_params = pd.read_csv('model_params/marginal_params.csv')
    copula_params = pd.read_csv('model_params/copula_params.csv')
    trader = Trader(
        selected_pairs=selected_pairs,
        marginal_params=marginal_params,
        copula_params=copula_params
    )
    trader.run()