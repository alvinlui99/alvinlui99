import pandas as pd

from config import setup_logging
from core import TradingCycle
from config import BaseConfig, ModelConfig
from utils import utils
from models import LGBMRegressorModel

if __name__ == "__main__":
    setup_logging()
    trading_cycle = TradingCycle()