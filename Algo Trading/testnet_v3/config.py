"""
Configuration module for the algorithmic trading system.

This module contains configuration classes for different components of the system.
"""
import os
import logging
import logging.handlers
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BaseConfig:
    """Base configuration with general settings."""
    
    # System paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_DIR = os.path.join(ROOT_DIR, "model", "trained_models")
    
    # Ensure directories exist
    for directory in [DATA_DIR, LOG_DIR, MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Trading symbols
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", 
               "SOLUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT"]
    
    # Data timeframes
    DATA_TIMEFRAME = "1h"  # Default timeframe
    
    # Default lookback periods
    DEFAULT_LOOKBACK = 100
    
    # Version
    VERSION = "1.0.0"
    
    # Debug mode
    DEBUG = False
    

class BinanceConfig:
    """Configuration for Binance API connection."""
    
    # API credentials (from environment variables)
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # Base URLs
    PRODUCTION_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"
    
    # Default to testnet for safety
    USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "t")
    BASE_URL = TESTNET_URL if USE_TESTNET else PRODUCTION_URL
    
    # Rate limits
    WEIGHT_LIMIT_MINUTE = 1200  # Default weight limit per minute
    ORDER_LIMIT_SECOND = 50     # Default order limit per second
    
    # Position defaults
    LEVERAGE = 1  # Default leverage (1x)
    MARGIN_TYPE = "ISOLATED"  # Default margin type
    
    # Timeouts (seconds)
    REQUEST_TIMEOUT = 10
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds


class ModelConfig:
    """Configuration for the machine learning model."""
    
    # LightGBM model parameters
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }
    
    # Training parameters
    NUM_BOOST_ROUND = 100
    EARLY_STOPPING_ROUNDS = 10
    TRAIN_TEST_SPLIT_RATIO = 0.8
    VAL_TEST_SPLIT_RATIO = 0.5  # from the remaining 20%
    
    # Feature engineering settings
    FEATURE_CONFIG = {
        'trend': {
            'enabled': True,
            'sma_periods': [5, 10, 20, 50],
            'ema_periods': [5, 10, 20]
        },
        'momentum': {
            'enabled': True,
            'rsi_periods': [14],
            'macd': True
        },
        'volatility': {
            'enabled': True,
            'bb_period': 20,
            'atr_period': 14
        },
        'volume': {
            'enabled': True,
            'obv': True,
            'volume_sma_periods': [5]
        },
        'custom': {
            'enabled': True,
            'price_to_sma': True,
            'returns': [1, 3, 5],
            'volatility_periods': [5, 10]
        }
    }
    
    # Model file naming
    MODEL_FILE_PREFIX = "model_"
    MODEL_FILE_EXTENSION = ".lgb"


class TradingConfig:
    """Configuration for trading strategies and execution."""
    
    # Trading frequency
    DEFAULT_INTERVAL_MINUTES = 60
    
    # Default maximum allocation
    MAX_ALLOCATION = 0.25  # 25% of capital per position
    
    # Risk management
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    MAX_DRAWDOWN = 0.20    # Maximum drawdown allowed
    
    # Order defaults
    DEFAULT_ORDER_TYPE = "MARKET"
    TIME_IN_FORCE = "GTC"  # Good Till Cancelled
    
    # Position settings
    ALLOW_SHORTS = True  # Allow short positions
    MAX_POSITIONS = 5    # Maximum number of concurrent positions
    
    # Signal thresholds
    SIGNAL_THRESHOLD = 0.002  # Minimum predicted price change to generate a signal
    
    # Trade sizing
    USE_FIXED_POSITION_SIZE = False
    FIXED_POSITION_SIZE = 0.01  # If fixed position size is used
    
    # Emergency shutdown
    EMERGENCY_STOP_LOSS = 0.50  # 50% loss triggers emergency shutdown


class LoggingConfig:
    """Configuration for logging."""
    
    # Log levels
    CONSOLE_LOG_LEVEL = logging.INFO
    FILE_LOG_LEVEL = logging.DEBUG
    
    # Log formatting
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Log file settings
    LOG_FILENAME = f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    LOG_FILE_PATH = os.path.join(BaseConfig.LOG_DIR, LOG_FILENAME)
    MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT = 5  # Number of backup logs to keep
    
    # Other logging settings
    CAPTURE_WARNINGS = True
    PROPAGATE_EXCEPTIONS = True


def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LoggingConfig.CONSOLE_LOG_LEVEL)
    console_format = logging.Formatter(LoggingConfig.LOG_FORMAT, LoggingConfig.DATE_FORMAT)
    console_handler.setFormatter(console_format)
    
    # Create file handler
    os.makedirs(BaseConfig.LOG_DIR, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LoggingConfig.LOG_FILE_PATH,
        maxBytes=LoggingConfig.MAX_LOG_SIZE,
        backupCount=LoggingConfig.BACKUP_COUNT
    )
    file_handler.setLevel(LoggingConfig.FILE_LOG_LEVEL)
    file_format = logging.Formatter(LoggingConfig.LOG_FORMAT, LoggingConfig.DATE_FORMAT)
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Capture warnings
    if LoggingConfig.CAPTURE_WARNINGS:
        logging.captureWarnings(True)
    
    return logger


class DataConfig:
    """Configuration for data fetching and processing."""
    
    # Default timeframes available
    AVAILABLE_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    
    # Default timeframe
    DEFAULT_TIMEFRAME = "1h"
    
    # Default limits
    DEFAULT_LIMIT = 1000  # Number of candles to fetch
    
    # Historical data storage
    STORE_AS_CSV = True
    CSV_FILENAME_TEMPLATE = "{symbol}_{timeframe}.csv"
    
    # Derived features
    CALCULATE_RETURNS = True
    CALCULATE_LOG_RETURNS = True
    
    # Data cleanup
    REMOVE_DUPLICATE_TIMESTAMPS = True
    FILL_MISSING_VALUES = True
    HANDLE_OUTLIERS = True
    
    # Other settings
    ADJUST_TIMEZONE = False
    TARGET_TIMEZONE = "UTC"