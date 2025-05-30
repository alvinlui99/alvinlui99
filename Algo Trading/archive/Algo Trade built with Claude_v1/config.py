"""Configuration settings for the Machine Learning Trading System"""

# ===== System Settings =====
TEST_MODE = True           # Set to True for quick testing
USE_TRAINED_MODEL = False  # Whether to use pre-trained model
SAVE_MODEL = True         # Whether to save model after training
MODEL_PATH = "best_model.keras"
DATA_PATH = "../Binance Data/1h"
DATA_TIMEFRAME = "1h"

# ===== Trading Settings =====
class TradingConfig:
    # Position and Risk Management
    MAX_POSITION_PCT = 0.3       # Maximum position size as percentage of portfolio
    MAX_POSITION_CHANGE = 0.2    # Maximum allowed position change in one trade
    MIN_CASH = 10.0             # Minimum cash balance to maintain
    
    # Trading Costs and Frequency
    COMMISSION_RATE = 0.00045    # Binance futures trading fee (0.045%)
    REBALANCE_THRESHOLD = 0.1    # Minimum weight deviation to trigger rebalance
    RISK_FREE_RATE = 0.02        # Risk-free rate for Sharpe ratio calculation
    COMMISSION_BUFFER_PERCENTAGE = 0.01  # Reserve 1% for commissions 
    
    # Performance Calculation
    TRADING_DAYS = 252
    PERIODS_PER_DAY = 24        # For hourly data
    ANNUALIZATION_FACTOR = TRADING_DAYS * PERIODS_PER_DAY

# ===== Model Settings =====
class ModelConfig:
    # Dataset Split
    TRAIN_SIZE = 0.7            # 70% for training
    VALIDATION_SIZE = 0.15      # 15% for validation
    TEST_SIZE = 0.15           # 15% for testing

# ===== Feature Engineering Settings =====
class FeatureConfig:
    # Basic Feature Windows
    LOOKBACK_PERIOD = 14        # Base lookback period for features
    
    # Technical Indicators
    TECHNICAL_INDICATORS = {
        'RSI_PERIOD': 14,
        'MACD_FAST': 12,
        'MACD_SLOW': 26,
        'MACD_SIGNAL': 9,
        'BB_PERIOD': 20,
        'BB_STD': 2
    }

# ===== Test Mode Settings =====
if TEST_MODE:
    # Simplified indicators for testing
    FeatureConfig.TECHNICAL_INDICATORS = {
        'RSI_PERIOD': 5,
        'MACD_FAST': 5,
        'MACD_SLOW': 10,
        'MACD_SIGNAL': 3,
        'BB_PERIOD': 5,
        'BB_STD': 2
    } 

# Add this constant
COMMISSION_BUFFER_PERCENTAGE = 0.01  # Reserve 1% for commissions 