from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from strategy import Strategy
from portfolio import Portfolio
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from config import StrategyConfig
from regime_detector import RegimeDetector

class MLStrategy(Strategy):
    """
    Machine Learning based trading strategy that predicts portfolio weights
    based on multiple assets' features
    """
    
    def __init__(self, features: List[str], symbols: List[str]):
        """Initialize ML Strategy with default empty state"""
        super().__init__()
        self.is_trained = False
        self.min_rebalance_period = 24 * 7  # Change to weekly instead of daily
        self.rebalance_threshold = 0.10  # Increase to 10% deviation threshold
        self.last_rebalance_time = 0
        self.leverage = StrategyConfig.LEVERAGE

        self.features = features
        self.symbols = symbols

        self.model_lgbm = None

        self.regime_detector = RegimeDetector()
        self.scaler = StandardScaler()

        self.leverage_components = {
            'regime': 0,
            'trend': 0,
            'volatility': 0,
            'confidence': 0,
            'momentum': 0,
            'drawdown': 0
        }
        
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Get feature importances from each model
        importances = []
        for i, estimator in enumerate(self.model_lgbm.estimators_):
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
        
        # Calculate multi-period forward returns
        returns_1d = prices.pct_change(1).shift(-1)
        returns_5d = prices.pct_change(5).shift(-5)
        returns_20d = prices.pct_change(20).shift(-20)
        
        # Combine different horizons
        # y = (0.5 * returns_1d + 0.3 * returns_5d + 0.2 * returns_20d)
        y = returns_1d
        
        # Clean up data
        valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[valid_rows]
        y = y[valid_rows]
        
        return X, y

    def train(self, train_data: Dict[str, pd.DataFrame]) -> None:
        X_train, y_train = self.prepare_training_data(train_data)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        params = {
            'objective': 'regression',  # Change to standard regression
            'learning_rate': 0.01,
            'verbose': -1,
            'n_jobs': -1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
        }
        
        # Use MultiOutputRegressor
        self.model_lgbm = MultiOutputRegressor(
            lgb.LGBMRegressor(**params)
        )
        
        self.model_lgbm.fit(X_train_scaled, y_train)
        self.is_trained = True

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
        if not self.is_trained:
            raise ValueError("Strategy must be trained")
        
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
        raw_weights = self.model_lgbm.predict(X_scaled)
        
        # Scale predictions to increase spread
        scaled_weights = raw_weights * 5.0  # Amplify differences
        
        # Apply softmax for final weights
        weights = self._softmax(scaled_weights.reshape(1, -1))[0]
        
        # Ensure minimum weight threshold
        weights[weights < 0.05] = 0  # Zero out very small weights
        if weights.sum() > 0:
            weights = weights / weights.sum()  # Renormalize
        else:
            weights = np.ones_like(weights) / len(weights)  # Fallback to equal weight
            
        return dict(zip(self.symbols, weights))

    def should_rebalance(self, timestep: int, current_weights: Dict[str, float],
                         target_weights: Dict[str, float]) -> bool:
        """Determine if rebalancing is needed based on thresholds and timing"""
        # Check if minimum time has passed since last rebalance
        if timestep - self.last_rebalance_time < self.min_rebalance_period:
            return False

        # Calculate maximum deviation from target
        max_deviation = max(
            abs(current_weights.get(s, 0) - target_weights.get(s, 0))
            for s in self.symbols
        )
        
        # Rebalance if deviation exceeds threshold
        return max_deviation > self.rebalance_threshold

    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 test_data: Dict[str, pd.DataFrame], equity_curve: pd.Series) -> Dict[str, Dict[str, float]]:
        """Execute strategy with regime-based leverage adjustment"""
        timestep = len(equity_curve)
        if not self.is_trained:
            raise ValueError("Strategy must be trained")
        
        current_data = {
            symbol: df.iloc[timestep] 
            for symbol, df in test_data.items()
        }
        
        current_weights = portfolio.get_weights(current_prices)
        target_weights = self.predict_weights(current_data)
        
        # Check if rebalancing is needed
        if not self.should_rebalance(timestep, current_weights, target_weights):
            return {}  # Return empty dict to indicate no trades needed
            
        # If rebalancing, update last rebalance time
        self.last_rebalance_time = timestep
        
        # Calculate positions with commission-aware sizing and regime-adjusted leverage
        total_equity = portfolio.get_total_value(current_prices)
        adjusted_equity = self.get_commission_adjusted_equity(total_equity)

        # Detect market regime
        regime_leverage = self.regime_detector.get_regime_leverage(test_data, target_weights, equity_curve)
        
        self.leverage_components = self.regime_detector.leverage_components

        signals = {}
        for symbol, weight in target_weights.items():
            price = current_prices[symbol]
            if weight > 0:
                signals[symbol] = {
                    'quantity': (weight * adjusted_equity * regime_leverage) / price,
                    'leverage': 1 # regime_leverage  # Use regime-adjusted leverage
                }
            else:
                signals[symbol] = {
                    'quantity': 0,
                    'leverage': 1 # regime_leverage
                }
                
        return signals
    
    def custom_objective(self, preds, train_data):
        """
        Custom LightGBM objective function to maximize portfolio Sharpe ratio
        with numerical stability improvements
        """
        returns = train_data.get_label()
        n_samples = len(returns) // len(self.symbols)
        returns = returns.reshape(n_samples, len(self.symbols))
        
        preds = preds.reshape(n_samples, len(self.symbols))
        weights = self._softmax(preds)
        
        # Calculate portfolio-level metrics
        portfolio_returns = np.sum(weights * returns, axis=1)
        
        grad = np.zeros_like(preds)
        hess = np.ones_like(preds) * 0.1
        
        try:
            for i in range(len(self.symbols)):
                # Portfolio return gradient
                port_grad = returns[:, i] - portfolio_returns
                
                # Weight gradient through softmax
                weight_grad = weights[:, i] * (1 - weights[:, i])
                
                # Combine gradients
                grad[:, i] = (np.mean(port_grad * weight_grad) * 100.0)
                
                # Dynamic hessian based on prediction confidence
                hess[:, i] = np.abs(grad[:, i]) + 0.1
                
            # Clip gradients for stability
            grad = np.clip(grad, -10.0, 10.0)
            hess = np.clip(hess, 0.1, 10.0)
            
        except Exception as e:
            print(f"Error in objective calculation: {e}")
            grad = np.zeros_like(preds)
            hess = np.ones_like(preds) * 0.1
        
        return grad.flatten(), hess.flatten()

    def _softmax(self, x):
        """Convert raw predictions to valid portfolio weights"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)