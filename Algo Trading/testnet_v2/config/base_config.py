class BaseConfig:
    # SWITCHES
    BACKTEST_SWITCH = True
    LOAD_SCALER_SWITCH = False
    LOAD_MODEL_SWITCH = False
    LOAD_TRAINING_DATA_SWITCH = False

    # PATHS
    DATA_PATH = "data"
    DATA_TIMEFRAME = "1h"

    # CONSTANTS
    HOURS_PER_DAY = 24
    DAYS_PER_YEAR = 365
    HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR