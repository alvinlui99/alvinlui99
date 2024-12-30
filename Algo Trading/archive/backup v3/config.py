"""Configuration settings for the Market Neutral Trading System"""

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
    REBALANCE_INTERVAL = 24      # Rebalance every 24 hours
    REBALANCE_THRESHOLD = 0.1    # Minimum weight deviation to trigger rebalance
    
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
    
    # LSTM Architecture
    SEQUENCE_LENGTH = 20        # Length of input sequences for LSTM
    LSTM_UNITS = 128           # Units in LSTM layer
    DENSE_UNITS = 64           # Units in dense layer
    DROPOUT_RATE = 0.2
    
    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5

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

# ===== Regime Detection Settings =====
class RegimeConfig:
    # Regime Detection Parameters
    N_REGIMES = 3              # Number of market regimes
    LOOKBACK_PERIOD = 60       # Periods for regime detection
    
    # Feature Calculation Windows
    VOLATILITY_WINDOW = 20     # Window for volatility calculation
    SMA_SHORT_WINDOW = 20      # Short-term SMA window
    SMA_LONG_WINDOW = 50       # Long-term SMA window
    
    # HMM Model Settings
    HMM_COVARIANCE_TYPE = "full"
    HMM_N_ITER = 100
    
    # Regime-Specific Portfolio Settings
    REGIME_WEIGHTS = {
        0: {  # Low Vol - More aggressive
            'weights': [0.4, 0.3, 0.3],
            'leverage': 1.0
        },
        1: {  # Medium Vol - Balanced
            'weights': [0.333, 0.333, 0.334],
            'leverage': 0.8
        },
        2: {  # High Vol - Conservative
            'weights': [0.5, 0.3, 0.2],
            'leverage': 0.5
        }
    }
    
    # Strategy Mixing Weights
    REGIME_MIXING = {
        0: {'lstm': 0.7, 'equal': 0.3},    # Low Vol
        1: {'lstm': 0.5, 'equal': 0.5},    # Medium Vol
        2: {'lstm': 0.3, 'equal': 0.7}     # High Vol
    }
    
    # Weight Combination Parameters
    STRATEGY_WEIGHT_RATIO = 0.7  # Weight for strategy combination
    BASE_WEIGHT_RATIO = 0.3     # Weight for regime base weights

# ===== Test Mode Settings =====
if TEST_MODE:
    # Reduce complexity for testing
    ModelConfig.SEQUENCE_LENGTH = 5
    ModelConfig.BATCH_SIZE = 16
    ModelConfig.EPOCHS = 2
    ModelConfig.LSTM_UNITS = 32
    ModelConfig.DENSE_UNITS = 16
    ModelConfig.EARLY_STOPPING_PATIENCE = 2
    ModelConfig.REDUCE_LR_PATIENCE = 1
    
    # Simplified indicators for testing
    FeatureConfig.TECHNICAL_INDICATORS = {
        'RSI_PERIOD': 5,
        'MACD_FAST': 5,
        'MACD_SLOW': 10,
        'MACD_SIGNAL': 3,
        'BB_PERIOD': 5,
        'BB_STD': 2
    } 