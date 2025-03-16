from binance.um_futures import UMFutures
from config import setup_logging, BaseConfig, BinanceConfig
from utils import utils

if __name__ == "__main__":
    setup_logging()
    client = UMFutures(BinanceConfig.API_KEY, BinanceConfig.API_SECRET, base_url=BinanceConfig.BASE_URL)
    all_klines = utils.get_klines_from_symbols(
        client=client,
        symbols=BaseConfig.SYMBOLS,
        timeframe=BaseConfig.DATA_TIMEFRAME,
        startTime=BaseConfig.TRAIN_START_DATE,
        endTime=BaseConfig.TRAIN_END_DATE)
    utils.save_dfs_to_csv(all_klines, 'data/klines.csv')