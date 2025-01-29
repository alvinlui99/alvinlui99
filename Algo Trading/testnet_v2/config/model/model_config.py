from ..base_config import BaseConfig

class ModelConfig:
    BACKTEST_SWITCH = BaseConfig.BACKTEST_SWITCH
    if BACKTEST_SWITCH:
        TRAIN_START_DATE = "2020-01-01"
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

    PRICE_COLUMN = "close"

    MODEL_PARAMS = {
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'objective': 'regression',
        'n_jobs': -1,
        'importance_type': 'gain',
        'min_child_samples': 10,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'verbose': 0
    }