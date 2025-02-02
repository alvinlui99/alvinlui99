# import schedule
import time
from config import setup_logging
from core import TradingCycle
from config import ModelConfig
import utils

if __name__ == "__main__":
    setup_logging()
    trading_cycle = TradingCycle()
    data = trading_cycle.get_historical_klines()
    utils.save_historical_klines(data, ModelConfig.DATA_PATH)

