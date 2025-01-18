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
        
        # ===================  COV  =========================

        # Add rolling covariance features
        returns_df = pd.DataFrame()
        for symbol in self.symbols:
            returns_df[symbol] = historical_data[symbol]['price'].pct_change().fillna(0)
        
        # Calculate rolling correlation/covariance features
        window = 60  # 60-hour rolling window
        for i, symbol1 in enumerate(self.symbols):
            for j, symbol2 in enumerate(self.symbols):
                if i < j:  # Only calculate each pair once
                    # Rolling correlation
                    corr = returns_df[symbol1].rolling(window, min_periods=1).corr(returns_df[symbol2])
                    feature_groups[f'corr_{symbol1}_{symbol2}'] = corr.fillna(0)
                    
                    # Rolling covariance
                    cov = returns_df[symbol1].rolling(window, min_periods=1).cov(returns_df[symbol2])
                    feature_groups[f'cov_{symbol1}_{symbol2}'] = cov.fillna(0)
                    
                    # Rolling beta
                    market_var = returns_df[symbol2].rolling(window, min_periods=1).var()
                    beta = cov.copy()
                    valid_var = market_var > 1e-8
                    beta[valid_var] = cov[valid_var] / market_var[valid_var]
                    beta[~valid_var] = 0
                    feature_groups[f'beta_{symbol1}_{symbol2}'] = beta
        
        # Add portfolio-level risk metrics
        returns_matrix = returns_df.values
        rolling_cov_matrices = []
        div_scores = []

        # Pre-fill with initial values
        initial_cov = np.cov(returns_matrix[:window].T)
        if np.isnan(initial_cov).any():
            initial_cov = np.zeros_like(initial_cov)

        for i in range(window):
            rolling_cov_matrices.append(initial_cov)
            try:
                eigenvalues = np.linalg.eigvals(initial_cov)
                if np.isnan(eigenvalues).any() or np.isinf(eigenvalues).any():
                    div_scores.append(0)
                else:
                    div_score = 1 - (max(eigenvalues) / (sum(eigenvalues) + 1e-8))
                    div_scores.append(div_score)
            except np.linalg.LinAlgError:
                div_scores.append(0)
        
        # Calculate rolling covariance matrices
        for i in range(window, len(returns_matrix)):
            window_data = returns_matrix[i-window:i]
            try:
                cov_matrix = np.cov(window_data.T)
                if np.isnan(cov_matrix).any():
                    cov_matrix = rolling_cov_matrices[-1]  # Use previous matrix if current is invalid
                rolling_cov_matrices.append(cov_matrix)
                
                eigenvalues = np.linalg.eigvals(cov_matrix)
                if np.isnan(eigenvalues).any() or np.isinf(eigenvalues).any():
                    div_scores.append(div_scores[-1])  # Use previous score if current is invalid
                else:
                    div_score = 1 - (max(eigenvalues) / (sum(eigenvalues) + 1e-8))
                    div_scores.append(div_score)
            except np.linalg.LinAlgError:
                rolling_cov_matrices.append(rolling_cov_matrices[-1])
                div_scores.append(div_scores[-1])
        
        feature_groups['portfolio_div_score'] = pd.Series(div_scores, index=returns_df.index)


        # ===================  COV  =========================

        # Clean up data by removing NaN rows from both X and returns
        valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(returns).any(axis=1))
        X = X[valid_rows]
        y = returns[valid_rows].values
        
        return X, y

    def predict_weights(self, current_data: Dict[str, pd.DataFrame], max_asset_weight: float = 0.4) -> Dict[str, float]:
        """
        Predict portfolio weights with risk constraints
        
        Args:
            current_data: Current market data
            max_asset_weight: Maximum weight for any single asset
        """
        # Prepare features
        feature_values = self._prepare_prediction_features(current_data)
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        weights = np.array([model.predict(X_scaled)[0] for model in self.models])
        
        # Calculate recent covariance matrix
        recent_returns = []
        window = 60
        for symbol in self.symbols:
            price_history = current_data[symbol]['price']  # Assume we have this
            returns = price_history.pct_change().dropna().values[-window:]
            recent_returns.append(returns)
        
        cov_matrix = np.cov(np.array(recent_returns))
        
        # Apply risk constraints
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Risk-adjusted weights
        if portfolio_risk > self.target_risk:  # Assume we have this as a parameter
            scaling_factor = self.target_risk / portfolio_risk
            weights = weights * scaling_factor
        
        # Apply position limits
        weights = np.minimum(weights, max_asset_weight)
        weights = np.maximum(weights, 0)
        
        # Renormalize
        weights = weights / np.sum(weights)
        
        return dict(zip(self.symbols, weights))
    
    def train(self, train_data: Dict[str, pd.DataFrame], val_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Train the ML model using historical data with validation
        """
        if not self.is_configured:
            raise ValueError("Strategy must be configured before training")
        
        self.target_risk = self.calculate_target_risk(train_data)

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
    
    def _prepare_prediction_features(self, current_data: Dict[str, pd.DataFrame]) -> List[float]:
        """Helper method to prepare features for prediction"""
        feature_values = []

        for feature in self.features:
            feature_group = []
            for symbol in self.symbols:
                if symbol not in current_data:
                    raise ValueError(f"Missing data for symbol {symbol}")
                
                data = current_data[symbol]
                if feature not in data.columns:
                    raise ValueError(f"Missing feature {feature} for symbol {symbol}")
                
                feature_group.append(data[feature].iloc[-1])
                
            feature_values.extend(feature_group)
            feature_array = np.array(feature_group)
            # Add market-wide features
            feature_values.append(np.mean(feature_array))
            feature_values.append(np.std(feature_array))
        
        return feature_values
    
    def calculate_target_risk(self, historical_data: Dict[str, pd.DataFrame], risk_multiplier: float = 1.0) -> float:
        """
        Calculate appropriate target risk based on hourly historical data
        
        Args:
            historical_data: Historical hourly price data for all assets
            risk_multiplier: Adjust risk tolerance (default=1.0, higher=more aggressive)
        
        Returns:
            float: Target portfolio risk (annualized volatility)
        """
        # Calculate historical returns
        returns_df = pd.DataFrame()
        for symbol in self.symbols:
            returns_df[symbol] = historical_data[symbol]['price'].pct_change()
        
        # Constants for annualization
        HOURS_PER_DAY = 24
        DAYS_PER_YEAR = 365
        HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR
        
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