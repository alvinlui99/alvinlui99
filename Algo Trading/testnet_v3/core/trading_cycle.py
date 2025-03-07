# Standard library imports
import logging

# Third-party imports
from binance.um_futures import UMFutures

# Local application/library specific imports
from config import BaseConfig
from utils import utils
from strategy import Strategy

class TradingCycle:
    def __init__(self, client: UMFutures, strategy: Strategy):
        self.client = client
        self.strategy = strategy

    def run(self):
        """Main trading cycle that runs every hour"""
        try:
            klines = utils.get_klines_from_symbols(
                client=self.client,
                symbols=BaseConfig.SYMBOLS,
                timeframe=BaseConfig.DATA_TIMEFRAME)
            signals = self.strategy.get_signals(klines)
            if signals:
                self._execute_signals(signals)
        except Exception as e:
            logging.error(f"Error in trading cycle: {str(e)}")
            logging.error("Full error:", exc_info=True)

    def _execute_signals(self, signals):
        pass
