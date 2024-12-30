"""Configuration settings for the Market Neutral Trading System"""

# ===== Data Settings =====
DATA_TIMEFRAME = "1h"
DATA_PATH = "../Binance Data/1h"
MODEL_PATH = "best_model.keras"

# ===== Trading Parameters =====
# Position and Risk Management
MAX_POSITION_PCT = 0.3      # Maximum position size as percentage of portfolio
MAX_POSITION_CHANGE = 0.2   # Maximum allowed position change in one trade
MIN_CASH = 10.0            # Minimum cash balance to maintain

# Trading Costs and Frequency
COMMISSION_RATE = 0.00045   # Binance futures trading fee (0.045%)
REBALANCE_INTERVAL = 24     # Rebalance every 24 hours
REBALANCE_THRESHOLD = 0.1   # Minimum weight deviation to trigger rebalance

# ===== Model Settings =====
# Data Processing
LOOKBACK_PERIOD = 14        # Periods for technical indicators
SEQUENCE_LENGTH = 20        # Length of input sequences for LSTM

# Dataset Split
TRAIN_SIZE = 0.7           # 70% for training
VALIDATION_SIZE = 0.15     # 15% for validation
TEST_SIZE = 0.15          # 15% for testing

# LSTM Architecture
LSTM_UNITS = [128, 64, 32] # Units in LSTM layers
DENSE_UNITS = [16, 8]      # Units in dense layers
BATCH_SIZE = 32
EPOCHS = 100

# Training Parameters
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Model Control
USE_TRAINED_MODEL = False    # Whether to use pre-trained model
SAVE_MODEL = True          # Whether to save model after training

# ===== Technical Indicators =====
TECHNICAL_INDICATORS = {
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BB_PERIOD': 20,
    'BB_STD': 2
}

# ===== Performance Calculation =====
TRADING_DAYS = 252
PERIODS_PER_DAY = 24       # For hourly data
ANNUALIZATION_FACTOR = TRADING_DAYS * PERIODS_PER_DAY

# ===== Test Mode Settings =====
TEST_MODE = False           # Set to True for quick testing

if TEST_MODE:
    # Reduce complexity for testing
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 16
    EPOCHS = 2
    LSTM_UNITS = [32, 16]
    DENSE_UNITS = [8]
    EARLY_STOPPING_PATIENCE = 2
    REDUCE_LR_PATIENCE = 1
    
    # Simplified indicators for testing
    TECHNICAL_INDICATORS = {
        'RSI_PERIOD': 5,
        'MACD_FAST': 5,
        'MACD_SLOW': 10,
        'MACD_SIGNAL': 3,
        'BB_PERIOD': 5,
        'BB_STD': 2
    } 