from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from strategy import Strategy
from portfolio import Portfolio
from regime_detector import RegimeDetector
from ml_model import PortfolioMLModel

class MLStrategy(Strategy):
    """
    Machine Learning based trading strategy that predicts portfolio weights
    based on multiple assets' features
    """
    
    def __init__(self):
        """Initialize ML Strategy with default empty state"""
        super().__init__()
        self.features: Optional[List[str]] = None
        self.model = PortfolioMLModel()
        self.regime_detector = None
        self.is_configured = False
        self.is_trained = False
        self.symbols = None
        # Add new parameters
        self.min_rebalance_period = 24 * 7  # Change to weekly instead of daily
        self.rebalance_threshold = 0.10  # Increase to 10% deviation threshold
        self.last_rebalance_time = 0

        self.leverage_components = {
            'regime': 0,
            'trend': 0,
            'volatility': 0,
            'confidence': 0,
            'momentum': 0,
            'drawdown': 0
        }
        
    def configure(self, 
                 features: List[str],
                 symbols: List[str]) -> None:
        """
        Configure strategy with LightGBM model by default
        """
        self.features = features
        self.symbols = symbols

        # Wrap with MultiOutputRegressor for multiple asset weights
        self.model.configure(features, symbols)
        self.regime_detector = RegimeDetector()
        self.is_configured = True

    def train(self, train_data: Dict[str, pd.DataFrame], val_data: Dict[str, pd.DataFrame],
              features: List[str], symbols: List[str]) -> Dict[str, float]:
        """
        Train the ML model using historical data with validation
        """
        self.configure(features, symbols)
        if not self.is_configured:
            raise ValueError("Model must be configured before training")
        
        self.model.train(train_data, val_data)
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

    def predict_weights(self, current_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Predict portfolio weights for each symbol
        """
        if not self.is_configured or not self.is_trained:
            raise ValueError("Strategy must be configured and trained")
            
        return self.model.predict_weights(current_data)

    def should_rebalance(self, timestep: int, current_weights: Dict[str, float],
                         target_weights: Dict[str, float]) -> bool:
        """Determine if rebalancing is needed based on thresholds and timing"""
        # Check if minimum time has passed since last rebalance
        if timestep - self.last_rebalance_time < self.min_rebalance_period:
            return False

        # Check if the weights have changed significantly
        if max(abs(np.array(list(current_weights.values())) - np.array(list(target_weights.values())))) > self.rebalance_threshold:
            return True
        
        return False

    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 test_data: Dict[str, pd.DataFrame], equity_curve: pd.Series) -> Dict[str, Dict[str, float]]:
        """Execute strategy with regime-based leverage adjustment"""
        timestep = len(equity_curve) + 1
        if not self.is_configured or not self.is_trained:
            raise ValueError("Strategy must be configured and trained")
        
        current_data = {
            symbol: df.iloc[:timestep]
            for symbol, df in test_data.items()
        }

        current_weights = portfolio.get_weights()
        weights = self.predict_weights(current_data)
        
        # Check if rebalancing is needed
        if not self.should_rebalance(timestep, current_weights, weights):
            return {}  # Return empty dict to indicate no trades needed
            
        # If rebalancing, update last rebalance time
        self.last_rebalance_time = timestep
        
        # Calculate positions with commission-aware sizing and regime-adjusted leverage
        total_equity = portfolio.get_total_value(current_prices)
        adjusted_equity = self.get_commission_adjusted_equity(total_equity)
        

        # Detect market regime
        regime_leverage = self.regime_detector.get_regime_leverage(test_data, weights, equity_curve)
        self.leverage_components = self.regime_detector.leverage_components

        signals = {}
        for symbol, weight in weights.items():
            price = current_prices[symbol]
            if weight > 0:
                signals[symbol] = {
                    'quantity': (weight * adjusted_equity * regime_leverage) / price,
                    'leverage': regime_leverage
                }
            else:
                signals[symbol] = {
                    'quantity': 0,
                    'leverage': regime_leverage
                }
                
        return signals 