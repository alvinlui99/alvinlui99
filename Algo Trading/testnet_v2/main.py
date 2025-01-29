# import schedule
import time
from config import *
from core import TradingCycle
from utils import utils
# from services import BinanceService, PortfolioService
# from datetime import datetime, timedelta

if __name__ == "__main__":
    setup_logging()
    trading_cycle = TradingCycle()
    # trading_cycle.run()
    trading_cycle._setup_trading_cycle()
    start_time = utils.convert_str_to_datetime(ModelConfig.TRAIN_START_DATE)
    end_time = utils.convert_str_to_datetime(ModelConfig.VAL_END_DATE)
    trading_cycle.preprocessor.save_training_data(trading_cycle.get_market_data(start_time, end_time))
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)