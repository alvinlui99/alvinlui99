import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer.
        
        Args:
            returns_data (pd.DataFrame): Historical returns data for assets
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        """
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns_data.columns)
        self.weights = None
        self.optimal_portfolio = None

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between assets."""
        return self.returns.corr()

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """Calculate covariance matrix between assets."""
        return self.returns.cov()

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio return."""
        return np.sum(self.returns.mean() * weights) * 252  # Annualized return

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))

    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        return (self.portfolio_return(weights) - self.risk_free_rate) / self.portfolio_volatility(weights)

    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        return -self.sharpe_ratio(weights)

    def optimize_portfolio(self, constraints: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Optimize portfolio weights to maximize Sharpe ratio.
        
        Args:
            constraints (Dict): Additional constraints for optimization
            
        Returns:
            Tuple[np.ndarray, Dict]: Optimal weights and portfolio metrics
        """
        # Initial weights (equal allocation)
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # weights >= 0
        ]
        
        if constraints:
            constraints_list.extend(constraints)

        # Optimization
        result = minimize(
            self.negative_sharpe_ratio,
            init_weights,
            method='SLSQP',
            constraints=constraints_list
        )

        self.weights = result.x
        
        # Calculate portfolio metrics
        portfolio_metrics = {
            'weights': self.weights,
            'return': self.portfolio_return(self.weights),
            'volatility': self.portfolio_volatility(self.weights),
            'sharpe_ratio': self.sharpe_ratio(self.weights),
            'correlation_matrix': self.calculate_correlation_matrix()
        }
        
        self.optimal_portfolio = portfolio_metrics
        return self.weights, portfolio_metrics

    def get_rebalance_suggestions(self, current_weights: np.ndarray, threshold: float = 0.1) -> Dict:
        """
        Get suggestions for portfolio rebalancing based on current weights.
        
        Args:
            current_weights (np.ndarray): Current portfolio weights
            threshold (float): Threshold for rebalancing
            
        Returns:
            Dict: Rebalancing suggestions
        """
        if self.optimal_portfolio is None:
            logger.error("Portfolio not optimized yet")
            return None
            
        weight_differences = self.optimal_portfolio['weights'] - current_weights
        rebalance_suggestions = {}
        
        for i, diff in enumerate(weight_differences):
            if abs(diff) > threshold:
                rebalance_suggestions[self.returns.columns[i]] = {
                    'current_weight': current_weights[i],
                    'target_weight': self.optimal_portfolio['weights'][i],
                    'adjustment': diff
                }
                
        return rebalance_suggestions 