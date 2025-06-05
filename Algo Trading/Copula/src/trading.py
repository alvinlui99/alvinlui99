import os
import numpy as np

from ta.volatility import AverageTrueRange
from binance.um_futures import UMFutures

from collector import BinanceDataCollector
from copula_based_strategy import CopulaBasedStrategy
from config import Config

class Trader:
    def __init__(self, 
                 api_key: str = None, 
                 api_secret: str = None,
                 base_url: str = 'https://testnet.binancefuture.com'):
        if api_key is None:
            api_key = os.getenv('BINANCE_API_KEY')
        if api_secret is None:
            api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
        self.collector = BinanceDataCollector()

    def get_hedge_ratio(self, c1: str, c2: str, window: int = None) -> float:
        if window is None:
            window = Config().window
        df = self.collector.get_multiple_symbols_data([c1, c2])
        hedge_ratio = np.polyfit(df[c2]['close'], df[c1]['close'], 1)[0]
        return hedge_ratio

    def trade_from_signals(self, signals: dict,
                           long_threshold: float = None,
                           short_threshold: float = None):
        if long_threshold is None:
            long_threshold = Config().long_threshold
        if short_threshold is None:
            short_threshold = Config().short_threshold
        target_positions = {}
        prices = {}
        position_sizes = {}

        investable_budget = float(self.client.account()['totalMarginBalance']) * Config().investable_budget_pc / Config().max_positions
        for signal in signals.values():
            c1 = signal['c1']
            c2 = signal['c2']
            mi = signal['mi']

            # Netting quantity for all positions of the same asset
            if mi > long_threshold:
                print(f'{c1}-{c2}: long')
                hedge_ratio = self.get_hedge_ratio(c1, c2)

                price_1 = float(self.client.mark_price(c1)['markPrice']) if prices.get(c1) is None else prices[c1]
                price_2 = float(self.client.mark_price(c2)['markPrice']) if prices.get(c2) is None else prices[c2]
                prices[c1] = price_1
                prices[c2] = price_2

                unit_1 = investable_budget / (price_1 + price_2 * hedge_ratio)
                unit_2 = unit_1 * hedge_ratio
                target_positions[c1] = target_positions.get(c1, 0) - unit_1
                target_positions[c2] = target_positions.get(c2, 0) + unit_2
            elif mi < short_threshold:
                print(f'{c1}-{c2}: short')
                hedge_ratio = self.get_hedge_ratio(c1, c2)

                price_1 = float(self.client.mark_price(c1)['markPrice']) if prices.get(c1) is None else prices[c1]
                price_2 = float(self.client.mark_price(c2)['markPrice']) if prices.get(c2) is None else prices[c2]
                prices[c1] = price_1
                prices[c2] = price_2

                unit_1 = investable_budget / (price_1 + price_2 * hedge_ratio)
                unit_2 = unit_1 * hedge_ratio
                target_positions[c1] = target_positions.get(c1, 0) + unit_1
                target_positions[c2] = target_positions.get(c2, 0) - unit_2
        return target_positions

if __name__ == '__main__':
    trader = Trader()
    strategy = CopulaBasedStrategy()
    signals = strategy.run()
    target_positions, prices, position_sizes = trader.trade_from_signals(signals)