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
        self.min_rebalance_period = 24 * 7
        self.rebalance_threshold = 0.15
        self.last_rebalance_time = 0
        self.min_trade_value = 1000  # Minimum trade value in quote currency
        self.min_weight_change = 0.05  # Minimum 5% weight change per asset

        self.leverage_components = {
            'regime': 0,
            'trend': 0,
            'volatility': 0,
            'confidence': 0,
            'momentum': 0,
            'drawdown': 0
        }
        
        # Stop loss parameters
        self.drawdown_threshold = -0.15      # 15% drawdown trigger
        self.recovery_threshold = -0.10      # 10% drawdown for recovery
        self.stop_loss_active = False
        self.peak_equity = None
        self.stop_loss_entry_time = None     # Track when stop loss was triggered
        self.min_stop_loss_duration = 24*7   # Minimum 1 week in stop loss
        self.last_stop_loss_rebalance = 0    # Track last rebalance during stop loss
        self.stop_loss_rebalance_period = 24*3  # Wait 3 days between stop loss rebalances


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
        """Enhanced rebalancing logic with controlled stop-loss trading"""
        
        # Stop loss rebalancing logic
        if self.stop_loss_active:
            # Only rebalance if enough time has passed since last stop-loss rebalance
            time_since_last_rebalance = (timestep - self.last_stop_loss_rebalance)
            
            if time_since_last_rebalance < self.stop_loss_rebalance_period:
                return False
                
            # Calculate total deviation
            total_deviation = sum(abs(target_weights.get(symbol, 0) - current_weights.get(symbol, 0))
                                for symbol in self.symbols)
            
            # During stop loss, only rebalance for larger deviations
            if total_deviation > self.rebalance_threshold * 2:  # Double the normal threshold
                self.last_stop_loss_rebalance = timestep
                return True
            return False
            
        # Normal rebalancing checks
        if timestep - self.last_rebalance_time < self.min_rebalance_period:
            return False

        total_weight_change = sum(abs(target_weights.get(symbol, 0) - current_weights.get(symbol, 0))
                                for symbol in self.symbols)
        return total_weight_change > self.rebalance_threshold
    
    def filter_trades(self, current_weights: Dict[str, float],
                      target_weights: Dict[str, float],
                      total_equity: float) -> Dict[str, float]:
        """Filter out small trades to reduce unnecessary trading"""
        filtered_weights = {symbol: 0 for symbol in self.symbols}
        to_be_adjusted_weights = {}

        for symbol in self.symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            weight_change = abs(target_weight - current_weight)
            
            # Calculate trade value
            trade_value = weight_change * total_equity
            
            # Skip small trades
            if trade_value < self.min_trade_value or weight_change < self.min_weight_change:
                filtered_weights[symbol] = current_weight
            else:
                to_be_adjusted_weights[symbol] = target_weight
        
        # Renormalize weights
        total_adjusted_weight = sum(to_be_adjusted_weights.values())
        total_filtered_weight = sum(filtered_weights.values())
        if total_adjusted_weight > 0:
            filtered_weights = {
                symbol: weight / total_adjusted_weight * (1 - total_filtered_weight) 
                for symbol, weight in to_be_adjusted_weights.items()
            }
            
        return filtered_weights

    def check_stop_loss(self, equity_curve: pd.Series, timestep: int) -> bool:
        """
        Check and update stop loss status
        Returns True if stop loss is active
        """
        current_equity = equity_curve.iloc[-1]
        
        # Update peak equity only if not in stop loss
        if not self.stop_loss_active and (self.peak_equity is None or current_equity > self.peak_equity):
            self.peak_equity = current_equity
            
        current_drawdown = (current_equity - self.peak_equity) / self.peak_equity
        
        # Check for stop loss trigger
        if not self.stop_loss_active and current_drawdown <= self.drawdown_threshold:
            self.stop_loss_active = True
            self.stop_loss_entry_time = timestep
            self.last_stop_loss_rebalance = timestep
            
        # Check for recovery conditions
        elif self.stop_loss_active:
            min_time_elapsed = (timestep - self.stop_loss_entry_time) >= self.min_stop_loss_duration
            drawdown_recovered = current_drawdown > self.recovery_threshold
            
            if min_time_elapsed and drawdown_recovered:
                self.stop_loss_active = False
                self.peak_equity = current_equity  # Reset peak for new trades
                self.stop_loss_entry_time = None
        
        return self.stop_loss_active

    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 test_data: Dict[str, pd.DataFrame], equity_curve: pd.Series) -> Dict[str, Dict[str, float]]:
        """Execute strategy with regime-based leverage adjustment"""
        timestep = len(equity_curve) + 1
        if not self.is_configured or not self.is_trained:
            raise ValueError("Strategy must be configured and trained")
        
        stop_loss_active = False if equity_curve.empty else self.check_stop_loss(equity_curve, timestep)

        current_data = {
            symbol: df.iloc[:timestep]
            for symbol, df in test_data.items()
        }

        current_weights = portfolio.get_weights()
        target_weights = self.predict_weights(current_data)
        
        # Check if rebalancing is needed
        if not self.should_rebalance(timestep, current_weights, target_weights):
            return {}  # Return empty dict to indicate no trades needed
            
        # If rebalancing, update last rebalance time
        self.last_rebalance_time = timestep
        
        # Calculate positions with commission-aware sizing and regime-adjusted leverage
        total_equity = portfolio.get_total_value(current_prices)
        adjusted_equity = self.get_commission_adjusted_equity(total_equity)
        
        filtered_weights = self.filter_trades(current_weights, target_weights, adjusted_equity)
        if filtered_weights == current_weights:
            return {}  # Return empty dict to indicate no trades needed

        # Detect market regime
        if stop_loss_active:
            regime_leverage = 1
            self.leverage_components = {k: 1.0 for k in self.leverage_components}
        else:
            regime_leverage = self.regime_detector.get_regime_leverage(
                test_data, filtered_weights, equity_curve, timestep)
            self.leverage_components = self.regime_detector.leverage_components

        signals = {}
        for symbol, weight in filtered_weights.items():
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