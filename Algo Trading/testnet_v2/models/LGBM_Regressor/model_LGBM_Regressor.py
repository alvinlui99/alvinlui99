import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from typing import Dict
import os
from config import BaseConfig, ModelConfig, FeatureConfig, TradingConfig
from sklearn.metrics import mean_squared_error
from .. import Model

logger = logging.getLogger(__name__)

class LGBMRegressorModel(Model):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.target_risk = None
        self.feature_importances = []
        self.best_iterations = []
        self.training_metrics = []

    def configure(self) -> None:
        """Configure the model with the given configurations"""
        self.is_configured = True

    def train(self) -> Dict[str, float]:
        """Train the LightGBM model"""
        if not self.is_configured:
            raise ValueError("Strategy must be configured before training")
    
    def predict(self) -> Dict[str, float]:
        """Make predictions using the loaded model"""
        if not self.is_configured:
            raise ValueError("Model not configured")
        if not self.is_trained:
            raise ValueError("Model not trained")
