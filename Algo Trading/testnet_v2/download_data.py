from config import setup_logging
from core import TradingCycle
from config import BaseConfig, ModelConfig
from utils import utils

if __name__ == "__main__":
    setup_logging()
    trading_cycle = TradingCycle()
    for symbol in ModelConfig.GeneralConfig.SYMBOLS:
        data = trading_cycle.binance_service.get_historical_klines(
            symbol,
            utils.convert_str_to_datetime(ModelConfig.TRAIN_START_DATE),
            utils.convert_str_to_datetime(ModelConfig.TRAIN_END_DATE),
            BaseConfig.DATA_TIMEFRAME)
        print(f"data: {data}")
        # utils.save_historical_klines(data, 'data', BaseConfig.DATA_TIMEFRAME)
        break