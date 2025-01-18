"""Configuration settings for the Market Neutral Trading System"""

# ===== System Settings =====
TEST_MODE = True           # Set to True for quick testing
USE_TRAINED_MODEL = False  # Whether to use pre-trained model
SAVE_MODEL = True         # Whether to save model after training
MODEL_PATH = "best_model.keras"
DATA_PATH = "../Binance Data/15m"
DATA_TIMEFRAME = "M15"

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
    N_REGIMES = 3              # Bull, Neutral, Bear
    LOOKBACK_PERIOD = 30       # Base lookback period
    
    # Feature Calculation Windows
    VOLATILITY_WINDOW = 14     # Short-term volatility window
    SMA_SHORT_WINDOW = 10      # Fast MA
    SMA_MID_WINDOW = 20        # Medium MA
    SMA_LONG_WINDOW = 40       # Slow MA
    
    # HMM Model Settings
    HMM_COVARIANCE_TYPE = "diag"  # Use diagonal covariance
    HMM_N_ITER = 200              # Number of iterations
    
    # Parameter Ranges for Validation
    VALIDATION_RANGES = {
        'leverage': {
            0: (1.0, 1.5),     # Bull: Higher leverage range
            1: (0.7, 1.2),     # Neutral: Moderate leverage range
            2: (0.3, 0.8)      # Bear: Lower leverage range
        },
        'stop_loss': {
            0: (0.03, 0.07),   # Bull: Tighter stops
            1: (0.05, 0.10),   # Neutral: Moderate stops
            2: (0.07, 0.15)    # Bear: Wider stops
        },
        'take_profit': {
            0: (0.10, 0.20),   # Bull: Higher targets
            1: (0.07, 0.15),   # Neutral: Moderate targets
            2: (0.05, 0.10)    # Bear: Lower targets
        },
        'position_size': {
            0: (0.8, 1.0),     # Bull: Larger positions
            1: (0.5, 0.8),     # Neutral: Moderate positions
            2: (0.2, 0.5)      # Bear: Smaller positions
        }
    }
    
    # Initial Portfolio Settings (to be tuned by validation)
    REGIME_WEIGHTS = {
        0: {  # Bull Market
            'weights': [0.5, 0.3, 0.2],  # More aggressive allocation
            'leverage': 1.2,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'max_position': 1.0,
            'rebalance_threshold': 0.1
        },
        1: {  # Neutral Market
            'weights': [0.4, 0.3, 0.3],  # Balanced allocation
            'leverage': 1.0,
            'stop_loss': 0.07,
            'take_profit': 0.12,
            'max_position': 0.8,
            'rebalance_threshold': 0.15
        },
        2: {  # Bear Market
            'weights': [0.6, 0.3, 0.1],  # More defensive allocation
            'leverage': 0.5,
            'stop_loss': 0.10,
            'take_profit': 0.08,
            'max_position': 0.5,
            'rebalance_threshold': 0.2
        }
    }
    
    # Strategy Mixing Weights (to be tuned by validation)
    REGIME_MIXING = {
        0: {  # Bull Market
            'lstm': 0.7,       # More algorithmic
            'equal': 0.3
        },
        1: {  # Neutral Market
            'lstm': 0.5,       # Balanced
            'equal': 0.5
        },
        2: {  # Bear Market
            'lstm': 0.3,       # More passive
            'equal': 0.7
        }
    }
    
    # Performance Metrics for Validation
    VALIDATION_METRICS = {
        'sharpe_ratio': 0.5,   # Weight for Sharpe Ratio
        'sortino_ratio': 0.5,  # Weight for Sortino Ratio
    }
    
    # Risk Management Parameters (to be tuned)
    RISK_PARAMS = {
        'max_correlation': 0.7,        # Maximum correlation between assets
        'min_liquidity': 1000000,      # Minimum daily volume
        'max_sector_exposure': 0.4,    # Maximum exposure to any sector
        'vol_target': {
            0: 0.20,  # Bull: Higher vol target
            1: 0.15,  # Neutral: Moderate vol target
            2: 0.10   # Bear: Lower vol target
        }
    }
    
    # Portfolio Constraints
    CONSTRAINTS = {
        'min_weight': 0.05,    # Minimum position weight
        'max_weight': 0.4,     # Maximum position weight
        'max_turnover': {      # Maximum portfolio turnover
            0: 0.4,  # Bull: Higher turnover allowed
            1: 0.3,  # Neutral: Moderate turnover
            2: 0.2   # Bear: Lower turnover
        }
    }
    
    # Weight Combination Parameters
    STRATEGY_WEIGHT_RATIO = 0.7   # Weight for strategy combination
    BASE_WEIGHT_RATIO = 0.3      # Weight for regime base weights
    
    # Data frequency settings
    DATA_FREQUENCY = 'M15'  # 15-minute, 'H' for hourly, 'D' for daily
    MINUTES_PER_HOUR = 60
    HOURS_PER_DAY = 24
    
    # Feature timeframes (in base frequency units)
    TIMEFRAMES = {
        'H': {  # Hourly
            'WEEKLY': 7 * 24,
            'SMA_21': 21 * 24,
            'MONTHLY': 30 * 24,
            'SMA_50': 50 * 24,
            'QUARTERLY': 90 * 24,
            'SMA_200': 200 * 24
        },
        'D': {  # Daily
            'WEEKLY': 7,
            'SMA_21': 21,
            'MONTHLY': 30,
            'SMA_50': 50,
            'QUARTERLY': 90,
            'SMA_200': 200
        },
        'M15': {  # 15-minute timeframes
            'WEEKLY': 7 * 24 * 4,
            'SMA_21': 21 * 24 * 4,
            'MONTHLY': 30 * 24 * 4,
            'SMA_50': 50 * 24 * 4,
            'QUARTERLY': 90 * 24 * 4,
            'SMA_200': 200 * 24 * 4
        }
    }
    
    # Regime Smoothing Parameters
    SMOOTHING = {
        'M15': {  # 15-minute data
            'WINDOW': 12,      # 3 hours (12 * 15min)
            'THRESHOLD': 0.6   # Need 60% confidence to switch
        },
        'H': {    # Hourly data
            'WINDOW': 4,       # 4 hours
            'THRESHOLD': 0.6
        },
        'D': {    # Daily data
            'WINDOW': 3,       # 3 days
            'THRESHOLD': 0.7   # More conservative for daily
        }
    }
    
    N_FEATURES = 10  # Number of features to select
    
    @classmethod
    def get_max_lookback(cls) -> int:
        """Get maximum lookback period based on data frequency"""
        timeframes = cls.TIMEFRAMES[cls.DATA_FREQUENCY]
        return max(timeframes.values())
    
    @classmethod
    def get_smoothing_params(cls) -> dict:
        """Get smoothing parameters based on data frequency"""
        return cls.SMOOTHING[cls.DATA_FREQUENCY]

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