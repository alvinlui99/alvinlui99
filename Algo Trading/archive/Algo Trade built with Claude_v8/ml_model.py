from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

class PortfolioMLModel:
    """Machine Learning model for portfolio weight prediction"""
    
    def __init__(self):
        self.is_configured = False
        self.is_trained = False
        self.scaler = StandardScaler()
        self.models = None
        self.feature_importances = None
        self.best_iterations = None
        self.training_metrics = None

    def prepare_training_data(self, historical_data: Dict[str, pd.DataFrame]) -> tuple:
        """
        Prepare training data with both asset-specific and market-wide features
        """
        # Group similar features across symbols (current approach)
        feature_groups = {}
        for feature in self.features:
            feature_data = pd.concat([
                historical_data[symbol][feature].rename(symbol)
                for symbol in self.symbols
            ], axis=1)
            feature_groups[feature] = feature_data
            
            # Add market-wide aggregates
            feature_groups[f"market_{feature}_mean"] = feature_data.mean(axis=1)
            feature_groups[f"market_{feature}_std"] = feature_data.std(axis=1)
        
        # Combine features
        all_features = []
        common_index = None
        
        for feature_name, feature_data in feature_groups.items():
            if common_index is None:
                common_index = feature_data.index
            else:
                feature_data = feature_data.loc[common_index]
            
            if isinstance(feature_data, pd.DataFrame):
                all_features.append(feature_data.values)
            else:  # Series (market-wide features)
                all_features.append(feature_data.values.reshape(-1, 1))
        
        X = np.hstack(all_features)
        
        # Calculate forward returns for each symbol
        # First create a DataFrame with all prices
        prices = pd.concat([
            historical_data[symbol]['price'].rename(symbol)
            for symbol in self.symbols
        ], axis=1).loc[common_index]
        
        # Calculate forward returns in one step
        returns = prices.pct_change(1).shift(-1)
        
        # Clean up data by removing NaN rows from both X and returns
        valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(returns).any(axis=1))
        X = X[valid_rows]
        y = returns[valid_rows].values
        
        return X, y

    def predict_weights(self, current_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Predict portfolio weights"""
        feature_values = []
        
        for feature in self.features:
            feature_group = []
            for symbol in self.symbols:
                if symbol not in current_data:
                    raise ValueError(f"Missing data for symbol {symbol}")
                
                data = current_data[symbol]
                if feature not in data.index:
                    raise ValueError(f"Missing feature {feature} for symbol {symbol}")
                
                feature_group.append(data[feature])
                
            feature_values.extend(feature_group)
            feature_array = np.array(feature_group)
            feature_values.append(np.mean(feature_array))
            feature_values.append(np.std(feature_array))
        
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        raw_weights = np.array([model.predict(X_scaled)[0] for model in self.models])
        
        # Ensure non-negative weights and normalize
        weights = np.maximum(raw_weights, 0)
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return dict(zip(self.symbols, weights))
    
    def train(self, train_data: Dict[str, pd.DataFrame], val_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Train the ML model using historical data with validation
        """
        if not self.is_configured:
            raise ValueError("Strategy must be configured before training")
            
        # Prepare training data
        X_train, y_train = self.prepare_training_data(train_data)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Prepare validation data
        X_val, y_val = self.prepare_training_data(val_data)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.models = []
        self.feature_importances = []
        self.best_iterations = []
        training_metrics = []

        feature_names = []
        for feature in self.features:
            feature_names.extend([f"{feature}_{symbol}" for symbol in self.symbols])
            feature_names.extend([f"market_{feature}_mean", f"market_{feature}_std"])

        for i, symbol in enumerate(self.symbols):
            model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.1,
                num_leaves=31,
                objective='regression',
                n_jobs=-1,
                importance_type='gain',
                min_child_samples=10,
                reg_alpha=0.01,
                reg_lambda=0.01,
                verbose=0
            )
            model.fit(
                X_train_scaled, 
                y_train[:, i],
                eval_set=[(X_val_scaled, y_val[:, i])],
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                eval_metric='mse',
                feature_name=feature_names
            )

            self.models.append(model)
            self.best_iterations.append(model.best_iteration_)
        
            # Store feature importance
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            self.feature_importances.append({
                'symbol': symbol,
                'importances': importance_dict
            })
            
            # Store training metrics for this asset
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            asset_metrics = {
                'symbol': symbol,
                'train_mse': mean_squared_error(y_train[:, i], train_pred),
                'val_mse': mean_squared_error(y_val[:, i], val_pred),
                'best_iteration': model.best_iteration_
            }
            training_metrics.append(asset_metrics)

        self.is_trained = True
        self.training_metrics = training_metrics

        # Calculate overall metrics
        train_predictions = self.predict_raw(X_train_scaled)
        val_predictions = self.predict_raw(X_val_scaled)

        metrics = {
            'train': self._calculate_metrics(y_train, train_predictions),
            'validation': self._calculate_metrics(y_val, val_predictions),
            'feature_importance': self.feature_importances,
            'asset_metrics': training_metrics
        }
    
        return metrics

    def configure(self, features: List[str], symbols: List[str]) -> None:
        """
        Configure strategy with LightGBM model by default
        """
        self.features = features
        self.symbols = symbols
        self.is_configured = True

    def predict_raw(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Make raw predictions without weight normalization
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return np.array([model.predict(X_scaled) for model in self.models]).T
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Mean Squared Error per asset
        mse = np.mean((y_true - y_pred) ** 2, axis=0)
        
        # R-squared per asset
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy per asset
        direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred), axis=0)
        
        # Average metrics across assets
        metrics = {
            'mse_avg': float(np.mean(mse)),
            'mse_per_asset': {f'asset_{i}': float(mse[i]) for i in range(len(mse))},
            'r2_avg': float(np.mean(r2)),
            'r2_per_asset': {f'asset_{i}': float(r2[i]) for i in range(len(r2))},
            'directional_accuracy_avg': float(np.mean(direction_correct)),
            'directional_accuracy_per_asset': {
                f'asset_{i}': float(direction_correct[i]) for i in range(len(direction_correct))
            }
        }
        
        return metrics