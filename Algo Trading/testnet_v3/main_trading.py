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
from strategy import LGBMstrategy
if __name__ == "__main__":
    setup_logging()
    client = UMFutures(BinanceConfig.API_KEY, BinanceConfig.API_SECRET, base_url=BinanceConfig.BASE_URL)
    model = utils.load_model("lgbm_model.pkl")
    trading_cycle = TradingCycle(client, LGBMstrategy(model))
    while True:
        logging.info("Running trading cycle")
        logging.info(f"Current time: {datetime.now()}")
        trading_cycle.run()
        schedule.run_pending()
        time.sleep(60)