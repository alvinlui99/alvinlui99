class BaseConfig:
    # SWITCHES
    BACKTEST_SWITCH = True
    LOAD_SCALER_SWITCH = False
    LOAD_MODEL_SWITCH = False
    LOAD_TRAINING_DATA_SWITCH = False

    # PATHS
    DATA_PATH = "data"
    DATA_TIMEFRAME = "1h"
    
    SYMBOLS = ['ADAUSDT', 'BNBUSDT', 'BTCUSDT', 'EOSUSDT', 'ETHUSDT', 'LTCUSDT', 'NEOUSDT', 'QTUMUSDT', 'XRPUSDT']

    # CONSTANTS
    HOURS_PER_DAY = 24
    DAYS_PER_YEAR = 365
    HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR
    
    if BACKTEST_SWITCH:
        TRAIN_START_DATE = "2020-01-01"
        # TRAIN_END_DATE = "2020-02-01"
        TRAIN_END_DATE = "2022-12-31"
        VAL_START_DATE = "2023-01-01"
        VAL_END_DATE = "2023-12-31"
        TEST_START_DATE = "2024-01-01"
        TEST_END_DATE = "2024-12-31"

        SCALER_PATH  = "models/trained/backtest/scalers/feature_scaler.joblib"
        MODEL_FOLDER = "models/trained/backtest/lgb_models"
        MODEL_NAME = "lgb_model"
        MODEL_EXTENSION = ".txt"

    else:
        TRAIN_START_DATE = "2020-01-01"
        TRAIN_END_DATE = "2023-12-31"
        VAL_START_DATE = "2024-01-01"
        VAL_END_DATE = "2024-12-31"

        SCALER_PATH  = "models/trained/production/scalers/feature_scaler.joblib"
        MODEL_FOLDER = "models/trained/production/lgb_models"
        MODEL_NAME = "lgb_model"
        MODEL_EXTENSION = ".txt"