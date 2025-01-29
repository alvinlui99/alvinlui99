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
        self.configure()

    def configure(self) -> None:
        """Configure the model with the given configurations"""
        self.base_config = BaseConfig()
        self.model_config = ModelConfig()
        self.feature_config = FeatureConfig()
        self.trading_config = TradingConfig()
        self.symbols = self.trading_config.SYMBOLS
        self.features = self.feature_config.FEATURE_NAMES
        self.model_folder = self.model_config.MODEL_FOLDER
        self.model_name = self.model_config.MODEL_NAME
        self.model_extension = self.model_config.MODEL_EXTENSION
        self.is_configured = True

    def train(
        self,
        dataset: dict[str, dict[str, pd.DataFrame]]
    ) -> Dict[str, float]:
        """Train the LightGBM model"""
        self.models = {}
        self.target_risk = None
        self.feature_importances = []
        self.best_iterations = []
        self.training_metrics = []
    
        if not self.is_configured:
            raise ValueError("Strategy must be configured before training")
        
        try:
            train_data_X = dataset['train_data_X']
            train_data_y = dataset['train_data_y']
            val_data_X = dataset['val_data_X']
            val_data_y = dataset['val_data_y']
            self.target_risk = self.calculate_target_risk(train_data_X)

            feature_names = []
            for feature in self.features:
                feature_names.extend([f"{feature}_{symbol}" for symbol in self.symbols])
                feature_names.extend([f"market_{feature}_mean", f"market_{feature}_std"])

            for i, symbol in enumerate(self.symbols):
                model = lgb.LGBMRegressor(**self.model_config.MODEL_PARAMS)
                model.fit(
                    train_data_X[symbol],
                    train_data_y[symbol],
                    eval_set=[(val_data_X[symbol], val_data_y[symbol])],
                    callbacks=[lgb.early_stopping(stopping_rounds=50)],
                    eval_metric='mse',
                    feature_name=feature_names
                )
                self.models[symbol] = model
                self.best_iterations.append(model.best_iteration_)

                # Store feature importance
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                self.feature_importances.append({
                    'symbol': symbol,
                    'importances': importance_dict
                })

                # Store training metrics for this asset
                train_pred = model.predict(train_data_X[symbol])
                val_pred = model.predict(val_data_X[symbol])
                asset_metrics = {
                    'symbol': symbol,
                    'train_mse': mean_squared_error(train_data_y[symbol], train_pred),
                    'val_mse': mean_squared_error(val_data_y[symbol], val_pred),
                    'best_iteration': model.best_iteration_
                }
                self.training_metrics.append(asset_metrics)
            
            self.is_trained = True

            # Save model and preprocessor
            self._save_model()

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def predict_weights(self, data: np.ndarray) -> Dict[str, float]:
        """Make predictions using the loaded model"""
        if not self.is_configured:
            raise ValueError("Model not configured")
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        weights = {}
        total_weight = 0
        try:
            for symbol in self.symbols:
                weights[symbol] = self.models[symbol].predict(data)
                total_weight += weights[symbol]
            for symbol in self.symbols:
                weights[symbol] /= total_weight
            return weights
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
        
    def calculate_target_risk(
            self,
            data: Dict[str, pd.DataFrame],
            risk_multiplier: float = 1.0
    ) -> float:
        """
        Calculate appropriate target risk based on hourly historical data
        
        Args:
            historical_data: Historical hourly price data for all assets
            risk_multiplier: Adjust risk tolerance (default=1.0, higher=more aggressive)
        
        Returns:
            float: Target portfolio risk (annualized volatility)
        """
        # Calculate returns
        returns_df = pd.DataFrame()
        for symbol in self.symbols:
            returns_df[symbol] = data[symbol]['close'].pct_change()
        
        # Constants for annualization
        HOURS_PER_DAY = self.base_config.HOURS_PER_DAY
        HOURS_PER_YEAR = self.base_config.HOURS_PER_YEAR
        
        # Window for rolling calculations (e.g., 30 days of hourly data)
        window = 30 * HOURS_PER_DAY
        
        # 1. Equal-weight portfolio risk (annualized)
        equal_weights = np.ones(len(self.symbols)) / len(self.symbols)
        cov_matrix = returns_df.cov() * HOURS_PER_YEAR  # Annualize hourly covariance
        equal_weight_risk = np.sqrt(equal_weights.T @ cov_matrix @ equal_weights)
        
        # 2. Average individual asset volatility (annualized)
        individual_vols = returns_df.std() * np.sqrt(HOURS_PER_YEAR)  # Annualize hourly volatility
        avg_asset_vol = individual_vols.mean()
        
        # 3. Minimum variance portfolio risk
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones(len(self.symbols))
            min_var_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
            min_var_risk = np.sqrt(min_var_weights.T @ cov_matrix @ min_var_weights)
        except np.linalg.LinAlgError:
            min_var_risk = equal_weight_risk  # Fallback if matrix is singular
        
        # Set target risk between min variance and equal weight
        target_risk = (min_var_risk + equal_weight_risk) / 2 * risk_multiplier
        
        return target_risk

    def load_model(self) -> None:
        """Load the LightGBM models from files"""
        try:
            self.models = {}
            for symbol in self.symbols:
                model_path = os.path.join(
                    self.model_folder,
                    f"{self.model_name}_{symbol}{self.model_extension}"
                )
                if os.path.exists(model_path):
                    # Create a new LGBMRegressor with the same parameters
                    model = lgb.LGBMRegressor(**self.model_config.MODEL_PARAMS)
                    # Load the booster and attach it to the regressor
                    model._Booster = lgb.Booster(model_file=model_path)
                    # Set other necessary attributes
                    model._n_features = len(model._Booster.feature_name())
                    model._n_features_in = model._n_features
                    model._feature_name = model._Booster.feature_name()
                    
                    self.models[symbol] = model
                    logger.info(f"LightGBM model for {symbol} loaded successfully")
                else:
                    logger.warning(f"Model file for {symbol} not found. New model will be trained.")
                    self.models[symbol] = None
            
            # Set is_trained flag based on whether all models were loaded
            self.is_trained = all(model is not None for model in self.models.values())
            
            if not self.is_trained:
                logger.info("Not all models were loaded. Training will be performed.")
                self.train()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models = {}
            self.is_trained = False

    def _save_model(self) -> None:
        """Save model and preprocessor to files"""
        try:
            os.makedirs(os.path.dirname(self.model_folder), exist_ok=True)
            for symbol, model in self.models.items():
                model_path = os.path.join(
                    self.model_folder,
                    f"{self.model_name}_{symbol}{self.model_extension}"
                )
                model.booster_.save_model(model_path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise