from typing import Dict, List

# Trading pairs to consider
TRADING_PAIRS: List[str] = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'SOL/USDT',
    'ADA/USDT',
    'DOT/USDT',
    'AVAX/USDT',
    'MATIC/USDT'
]

# Portfolio settings
MAX_POSITION_SIZE: float = 0.2  # Maximum 20% of portfolio per asset
TARGET_LEVERAGE: float = 2.0    # Default leverage
MAX_LEVERAGE: float = 5.0       # Maximum allowed leverage

# Risk management
MAX_DRAWDOWN: float = 0.15      # Maximum 15% drawdown
STOP_LOSS_PERCENTAGE: float = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE: float = 0.04  # 4% take profit

# Technical analysis parameters
LOOKBACK_PERIOD: int = 30       # Days of historical data for analysis
REBALANCE_THRESHOLD: float = 0.1 # 10% deviation triggers rebalancing

# Machine learning parameters
FEATURE_WINDOW: int = 24        # Hours of data for feature engineering
PREDICTION_WINDOW: int = 4       # Hours to predict ahead
MODEL_UPDATE_FREQUENCY: int = 24 # Hours between model updates

# API settings
BINANCE_API_KEY: str = ""       # To be filled from .env
BINANCE_API_SECRET: str = ""    # To be filled from .env

# Trading schedule
TRADING_INTERVAL: str = "1h"    # Trading interval (1h, 4h, 1d)
REBALANCE_INTERVAL: str = "1d"  # Portfolio rebalancing interval 