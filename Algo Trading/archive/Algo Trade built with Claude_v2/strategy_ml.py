from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from strategy import Strategy
from portfolio import Portfolio
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from config import StrategyConfig

class MLStrategy(Strategy):
    """
    Machine Learning based trading strategy that predicts portfolio weights
    based on multiple assets' features
    """
    
    def __init__(self):
        """Initialize ML Strategy with default empty state"""
        super().__init__()
        self.features: Optional[List[str]] = None
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.is_configured = False
        self.symbols = None
        # Add new parameters
        self.min_rebalance_period = 24 * 7  # Change to weekly instead of daily
        self.rebalance_threshold = 0.10  # Increase to 10% deviation threshold
        self.last_rebalance_time = 0
        self.leverage = StrategyConfig.LEVERAGE
        
    def configure(self, 
                 features: List[str],
                 symbols: List[str]) -> None:
        """
        Configure strategy with LightGBM model by default
        """
        self.features = features
        self.symbols = symbols
        base_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            objective='regression',
            n_jobs=-1,
            importance_type='gain',
            min_child_samples=10,  # Reduce to allow more specific predictions
            reg_alpha=0.01,        # Reduce L1 regularization
            reg_lambda=0.01,       # Reduce L2 regularization
            verbose=0
        )
        # Wrap with MultiOutputRegressor for multiple asset weights
        self.model = MultiOutputRegressor(base_model)
            
        self.scaler = StandardScaler()
        self.is_configured = True
        
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Get feature importances from each model
        importances = []
        for i, estimator in enumerate(self.model.estimators_):
            symbol = self.symbols[i]
            
            # Get raw feature names
            feature_names = []
            for f in self.features:
                # Asset-specific features
                for s in self.symbols:
                    feature_names.append(f"{f}_{s}")
                # Market features
                feature_names.extend([f"market_{f}_mean", f"market_{f}_std"])
            
            # Get importance scores
            importance = estimator.feature_importances_
            
            # Create DataFrame for this symbol
            imp_df = pd.DataFrame({
                'symbol': symbol,
                'feature': feature_names,
                'importance': importance
            })
            importances.append(imp_df)
            
        return pd.concat(importances, ignore_index=True)

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

    def train(self, train_data: Dict[str, pd.DataFrame], val_data: Dict[str, pd.DataFrame],
              features: List[str], symbols: List[str]) -> Dict[str, float]:
        """
        Train the ML model using historical data with validation
        """
        self.configure(features, symbols)
        if not self.is_configured:
            raise ValueError("Strategy must be configured before training")
            
        # Prepare training data
        X_train, y_train = self.prepare_training_data(train_data)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_val, y_val = self.prepare_training_data(val_data)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate validation metrics
        val_predictions = self.model.predict(X_val_scaled)
        val_metrics = self._calculate_metrics(y_val, val_predictions)
        
        self.is_trained = True
        return val_metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics"""
        # Portfolio returns using predicted weights
        portfolio_returns = (y_pred * y_true).sum(axis=1)
        
        # Calculate metrics
        return {
            'sharpe_ratio': np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std(),
            'mean_return': portfolio_returns.mean() * 252,  # Annualized
            'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns)
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / rolling_max - 1
        return np.min(drawdowns)

    def predict_weights(self, current_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Predict portfolio weights for each symbol
        """
        if not self.is_configured or not self.is_trained:
            raise ValueError("Strategy must be configured and trained")
            
        # Prepare features as before...
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
        raw_weights = self.model.predict(X_scaled)[0]
        
        # Ensure non-negative weights and normalize to sum to 1
        weights = np.maximum(raw_weights, 0)  # Ensure non-negative
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            weights = weights / weight_sum  # Normalize to sum to 1
        else:
            # If all weights are 0, use equal weights
            weights = np.ones_like(weights) / len(weights)
        
        return dict(zip(self.symbols, weights))

    def should_rebalance(self, timestep: int) -> bool:
        """Determine if rebalancing is needed based on thresholds and timing"""
        # Check if minimum time has passed since last rebalance
        if timestep - self.last_rebalance_time < self.min_rebalance_period:
            return False
        return True

    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 test_data: Dict[str, pd.DataFrame], timestep: int = 0) -> Dict[str, Dict[str, float]]:
        """Execute strategy with rebalancing controls"""
        if not self.is_configured or not self.is_trained:
            raise ValueError("Strategy must be configured and trained")
        
        current_data = {
            symbol: df.iloc[timestep] 
            for symbol, df in test_data.items()
        }
        
        weights = self.predict_weights(current_data)
        
        # Check if rebalancing is needed
        if not self.should_rebalance(timestep):
            return {}  # Return empty dict to indicate no trades needed
            
        # If rebalancing, update last rebalance time
        self.last_rebalance_time = timestep
        
        # Calculate positions with commission-aware sizing
        total_equity = portfolio.get_total_value(current_prices)
        adjusted_equity = self.get_commission_adjusted_equity(total_equity)
        
        signals = {}
        for symbol, weight in weights.items():
            price = current_prices[symbol]
            if weight > 0:
                signals[symbol] = {
                    'quantity': (weight * adjusted_equity * self.leverage) / price,
                    'leverage': self.leverage
                }
            else:
                signals[symbol] = {
                    'quantity': 0,
                    'leverage': self.leverage
                }
                
        return signals 