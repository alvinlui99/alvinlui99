import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from typing import Dict
import os
from config import BaseConfig, ModelConfig, TradingConfig
from sklearn.metrics import mean_squared_error
from .. import Model
from utils import utils

logger = logging.getLogger(__name__)

class LGBMRegressorModel(Model):
    def __init__(self):
        super().__init__()
        self.models = {}        

    def configure(self) -> None:
        """Configure the model with the given configurations"""
        self.is_configured = True

    def train(self,
              train_data: Dict[str, pd.DataFrame],
              train_target: Dict[str, pd.Series],
              val_data: Dict[str, pd.DataFrame],
              val_target: Dict[str, pd.Series]
              ) -> Dict[str, float]:
        """Train the LightGBM model"""
        if not self.is_configured:
            raise ValueError("Model must be configured before training")
        
        models = {}
        train_data_array = utils.data_df_to_nparray(train_data)
        val_data_array = utils.data_df_to_nparray(val_data)

        for i, symbol in enumerate(train_data.keys()):
            model = lgb.LGBMRegressor(**ModelConfig.MODEL_PARAMS)
            model.fit(
                train_data_array,
                train_target[symbol],
                eval_set=[(val_data_array, val_target[symbol])],
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                eval_metric='mse'
            )
            models[symbol] = model
        return models
    
    def predict(self) -> Dict[str, float]:
        """Make predictions using the loaded model"""
        if not self.is_configured:
            raise ValueError("Model not configured")
        if not self.is_trained:
            raise ValueError("Model not trained")
        
