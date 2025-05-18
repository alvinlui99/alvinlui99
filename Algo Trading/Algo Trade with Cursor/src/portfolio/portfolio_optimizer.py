import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, 
                 symbols: List[str],
                 returns: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer.
        
        Args:
            symbols (List[str]): List of trading pairs
            returns (pd.DataFrame): Historical returns data
            risk_free_rate (float): Annual risk-free rate (default: 2%)
        """
        self.symbols = symbols
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        
        # Calculate basic statistics
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        
        # Number of assets
        self.n_assets = len(symbols)
        
        # Constraints for optimization
        self.constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # weights >= 0
        ]
        
        # Bounds for optimization (0 <= weight <= 1)
        self.bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Optimization options
        self.options = {
            'maxiter': 1000,
            'ftol': 1e-8,
            'disp': True
        }
    
    def mean_variance_optimization(self, 
                                 target_return: Optional[float] = None,
                                 risk_aversion: float = 1.0) -> Tuple[np.ndarray, float, float]:
        """
        Perform mean-variance optimization.
        
        Args:
            target_return (float, optional): Target portfolio return
            risk_aversion (float): Risk aversion parameter (default: 1.0)
            
        Returns:
            Tuple[np.ndarray, float, float]: (optimal weights, portfolio return, portfolio risk)
        """
        def objective(weights):
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            if target_return is not None:
                # Minimize risk subject to target return
                return portfolio_risk
            else:
                # Maximize utility function: return - risk_aversion * risk^2
                return -(portfolio_return - risk_aversion * portfolio_risk ** 2)
        
        # Initial guess (equal weights)
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Add target return constraint if specified
        constraints = self.constraints.copy()
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(self.mean_returns * x) - target_return
            })
        
        try:
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                constraints=constraints,
                options=self.options
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                # Try with different initial weights if first attempt fails
                if target_return is not None:
                    # Try with weights concentrated on highest return asset
                    max_return_idx = np.argmax(self.mean_returns)
                    x0 = np.zeros(self.n_assets)
                    x0[max_return_idx] = 1.0
                    result = minimize(
                        objective,
                        x0,
                        method='SLSQP',
                        bounds=self.bounds,
                        constraints=constraints,
                        options=self.options
                    )
            
            optimal_weights = result.x
            portfolio_return = np.sum(self.mean_returns * optimal_weights)
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            
            return optimal_weights, portfolio_return, portfolio_risk
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            # Return equal weights as fallback
            weights = np.array([1/self.n_assets] * self.n_assets)
            ret = np.sum(self.mean_returns * weights)
            risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return weights, ret, risk
    
    def risk_parity_optimization(self) -> Tuple[np.ndarray, float, float]:
        """
        Perform risk parity optimization.
        
        Returns:
            Tuple[np.ndarray, float, float]: (optimal weights, portfolio return, portfolio risk)
        """
        def objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            risk_contributions = weights * (np.dot(self.cov_matrix, weights) / portfolio_risk)
            return np.sum((risk_contributions - portfolio_risk/self.n_assets) ** 2)
        
        # Initial guess (equal weights)
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        try:
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                constraints=self.constraints,
                options=self.options
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
            
            optimal_weights = result.x
            portfolio_return = np.sum(self.mean_returns * optimal_weights)
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            
            return optimal_weights, portfolio_return, portfolio_risk
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            # Return equal weights as fallback
            weights = np.array([1/self.n_assets] * self.n_assets)
            ret = np.sum(self.mean_returns * weights)
            risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return weights, ret, risk
    
    def minimum_variance_optimization(self) -> Tuple[np.ndarray, float, float]:
        """
        Find the minimum variance portfolio.
        
        Returns:
            Tuple[np.ndarray, float, float]: (optimal weights, portfolio return, portfolio risk)
        """
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Initial guess (equal weights)
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        try:
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                constraints=self.constraints,
                options=self.options
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
            
            optimal_weights = result.x
            portfolio_return = np.sum(self.mean_returns * optimal_weights)
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            
            return optimal_weights, portfolio_return, portfolio_risk
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            # Return equal weights as fallback
            weights = np.array([1/self.n_assets] * self.n_assets)
            ret = np.sum(self.mean_returns * weights)
            risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return weights, ret, risk
    
    def maximum_sharpe_optimization(self) -> Tuple[np.ndarray, float, float]:
        """
        Find the maximum Sharpe ratio portfolio.
        
        Returns:
            Tuple[np.ndarray, float, float]: (optimal weights, portfolio return, portfolio risk)
        """
        def objective(weights):
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            if portfolio_risk == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Try different initial weights
        initial_weights = [
            np.array([1/self.n_assets] * self.n_assets),  # Equal weights
            np.array([1.0, 0.0, 0.0]),  # Concentrated on first asset
            np.array([0.0, 1.0, 0.0]),  # Concentrated on second asset
            np.array([0.0, 0.0, 1.0])   # Concentrated on third asset
        ]
        
        best_result = None
        best_sharpe = -np.inf
        
        for x0 in initial_weights:
            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=self.bounds,
                    constraints=self.constraints,
                    options=self.options
                )
                
                if result.success:
                    weights = result.x
                    ret = np.sum(self.mean_returns * weights)
                    risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                    sharpe = (ret - self.risk_free_rate) / risk
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = (weights, ret, risk)
                
            except Exception as e:
                logger.warning(f"Optimization attempt failed: {str(e)}")
                continue
        
        if best_result is None:
            logger.warning("All optimization attempts failed, returning equal weights")
            weights = np.array([1/self.n_assets] * self.n_assets)
            ret = np.sum(self.mean_returns * weights)
            risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return weights, ret, risk
        
        return best_result
    
    def get_optimal_weights(self, 
                          strategy: str = 'mean_variance',
                          **kwargs) -> Dict[str, float]:
        """
        Get optimal portfolio weights using specified strategy.
        
        Args:
            strategy (str): Optimization strategy ('mean_variance', 'risk_parity', 
                          'minimum_variance', 'maximum_sharpe')
            **kwargs: Additional arguments for the optimization strategy
            
        Returns:
            Dict[str, float]: Optimal weights for each asset
        """
        if strategy == 'mean_variance':
            weights, _, _ = self.mean_variance_optimization(**kwargs)
        elif strategy == 'risk_parity':
            weights, _, _ = self.risk_parity_optimization()
        elif strategy == 'minimum_variance':
            weights, _, _ = self.minimum_variance_optimization()
        elif strategy == 'maximum_sharpe':
            weights, _, _ = self.maximum_sharpe_optimization()
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        return dict(zip(self.symbols, weights)) 