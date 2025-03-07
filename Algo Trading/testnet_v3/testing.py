# Standard library imports
import time
import schedule
import logging
from datetime import datetime

# Third-party imports
from binance.um_futures import UMFutures

# Local application/library specific imports
from config import setup_logging, BaseConfig, BinanceConfig
from utils import utils
from core import TradingCycle

if __name__ == "__main__":
    dfs = utils.load_dfs_from_csv(BaseConfig.SYMBOLS, 'data/klines.csv')
    for symbol in BaseConfig.SYMBOLS:
        df = dfs[symbol][['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        print(df)
        break
