import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import deque
from typing import Optional, List, Dict
from trading_types import PortfolioDataFrame, PortfolioRow

class Asset:
    """Represents a single trading asset with price and return tracking."""
    
    def __init__(self, symbol: str, maxlen: int = 250):
        """
        Initialize asset with symbol and rolling window size.
        
        Args:
            symbol: Trading pair symbol
            maxlen: Maximum length of rolling window for returns calculation (default: 250)
            
        Raises:
            ValueError: If symbol is empty or maxlen is less than 2
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if maxlen < 2:
            raise ValueError("maxlen must be at least 2")
    
        self.symbol = symbol
        self.prices = deque(maxlen=maxlen)
        self.ret_pct = deque(maxlen=maxlen)
        self.price_history = {}  # Full price history for regime detection

    def update_price(self, price: float) -> None:
        """
        Update asset price and calculate returns if enough prices available.
        
        Args:
            price: New price value to add
            
        Raises:
            ValueError: If price is invalid (non-numeric or <= 0)
        """
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(f"Invalid price value: {price}")
        self.prices.append(float(price))
        if len(self.prices) >= 2:
            self._calculate_return()

    def _calculate_return(self) -> None:
        """Calculate and store return percentage if conditions are met."""
        if len(self.prices) < 2:
            return
        
        ret_pct = (self.prices[-1] - self.prices[0]) / self.prices[0] * 100
        if not self.ret_pct or self.ret_pct[-1] != ret_pct:
            self.ret_pct.append(ret_pct)

    def get_return(self) -> deque:
        """Return the deque of return percentages."""
        return self.ret_pct

    def get_latest_price(self) -> float:
        """Return most recent price or 0.0 if no prices available."""
        return self.prices[-1] if self.prices else 0.0

    def get_price_history(self) -> Dict[str, float]:
        """
        Get the full price history.
        
        Returns:
            Dictionary mapping timestamps to prices
        """
        return self.price_history

    def set_price_history(self, history: Dict[str, float]) -> None:
        """
        Set the full price history.
        
        Args:
            history: Dictionary mapping timestamps to prices
        """
        self.price_history = history

class Portfolio:
    """Manages a collection of trading assets with portfolio optimization."""
    
    def __init__(self, symbols: List[str], initial_capital: float = 10000):
        """Initialize portfolio with given symbols and capital"""
        if not symbols:
            raise ValueError("Must provide at least one trading symbol")
            
        self.cash = initial_capital
        self.symbols = symbols
        self.n_assets = len(symbols)
        
        # Initialize portfolio DataFrame
        data = {
            'size': [0.0] * self.n_assets,
            'price': [0.0] * self.n_assets,
            'weight': [1.0/self.n_assets] * self.n_assets,
            'asset': [Asset(symbol) for symbol in symbols]
        }
        self.portfolio_df: pd.DataFrame[PortfolioRow] = pd.DataFrame(data, index=symbols)
        
        # Set correct data types
        self.portfolio_df = self.portfolio_df.astype({
            'size': 'float64'
        })
        
        # Separate DataFrame for returns
        self.returns_df = None

    def update_price(self, symbol: str, stream: dict) -> None:
        """
        Update asset price and recalculate related metrics.
        
        Args:
            symbol: Trading pair symbol
            stream: Price stream data dictionary
            
        Raises:
            ValueError: If stream data format is invalid
            KeyError: If symbol not found in portfolio
        """
        if not isinstance(stream, dict) or 'markPrice' not in stream:
            raise ValueError("Invalid stream data format")
        if symbol not in self.portfolio_df.index:
            raise KeyError(f"Symbol {symbol} not found in portfolio")

        price = float(stream["markPrice"])
        self.portfolio_df.at[symbol, 'price'] = price
        
        # Get the Asset object and update its price
        asset = self.portfolio_df.at[symbol, 'asset']
        if isinstance(asset, Asset):
            asset.update_price(price)
        else:
            # If asset is not an Asset object, create a new one
            asset = Asset(symbol)
            asset.update_price(price)
            self.portfolio_df.at[symbol, 'asset'] = asset

    def cal_asset_returns_df(self) -> None:
        """Calculate and update return metrics for all assets."""
        returns_data = {}
        
        # Collect returns for each asset
        for symbol, row in self.portfolio_df.iterrows():
            asset = row['asset']
            returns_data[symbol] = list(asset.get_return())
        
        # Create returns DataFrame
        max_len = max(len(ret) for ret in returns_data.values()) if returns_data else 0
        padded_returns = {
            symbol: ret + [0] * (max_len - len(ret))
            for symbol, ret in returns_data.items()
        }
        
        self.returns_df = pd.DataFrame(padded_returns)

    def get_asset_returns_df(self) -> pd.DataFrame:
        """Get dataframe of asset returns."""
        self.cal_asset_returns_df()
        return self.returns_df if self.returns_df is not None and not self.returns_df.empty else None

    def get_asset_cov(self) -> pd.DataFrame:
        """Calculate covariance matrix of asset returns."""
        returns_df = self.get_asset_returns_df()
        if returns_df is not None and not returns_df.empty:
            # Ensure we have enough data points
            min_periods = 2  # Minimum periods needed for covariance
            if returns_df.shape[0] < min_periods:
                return None
                
            # Calculate covariance matrix across assets
            cov = returns_df.cov()
            
            # Check if covariance matrix is valid
            if np.isnan(cov.values).any():
                return None
            
            if not np.all(np.linalg.eigvals(cov) > 0):
                # Add small diagonal values to ensure positive definiteness
                cov = cov + np.eye(cov.shape[0]) * 1e-8
            
            return cov
        return None

    def get_weights(self) -> pd.Series:
        """Get current portfolio weights, normalizing if needed."""
        weights = self.portfolio_df['weight']
        if not np.isclose(weights.sum(), 1.0):
            weights = pd.Series(1/self.n_assets, index=self.portfolio_df.index)
            self.portfolio_df['weight'] = weights
        return weights

    def get_position(self, prices: pd.Series = None, weights: pd.Series = None, 
                    budget: float = None) -> pd.Series:
        """
        Calculate position sizes based on weights and prices.
        
        Args:
            prices: Asset prices (uses current prices if None)
            weights: Portfolio weights (uses current weights if None)
            budget: Total budget to allocate (uses current cash if None)
            
        Returns:
            Series of position sizes for each asset
        """
        prices = prices if prices is not None else self.portfolio_df['price']
        weights = weights if weights is not None else self.get_weights()
        budget = budget if budget is not None else self.cash
        
        weighted_prices = (weights * prices).sum()
        if weighted_prices > 0:
            positions = weights * budget / weighted_prices
            # Update size column with new positions
            self.portfolio_df['size'] = positions
            return positions
        return pd.Series(0, index=self.portfolio_df.index)

    def compute_volatility(self, weights: np.ndarray = None, 
                         cov: pd.DataFrame = None) -> float:
        """
        Compute portfolio volatility given weights and covariance matrix.
        """
        if weights is None or cov is None:
            return float('inf')
        
        try:
            weights = np.array(weights)
            cov_array = cov.to_numpy()
            
            # Check dimensions
            if weights.shape[0] != cov_array.shape[0]:
                return float('inf')
            
            # Check for NaN values
            if np.isnan(cov_array).any() or np.isnan(weights).any():
                return float('inf')
            
            variance = np.dot(weights.T, np.dot(cov_array, weights))
            if variance < 0:
                return float('inf')
                
            return np.sqrt(variance)
            
        except Exception:
            return float('inf')

    def compute_expected_return(self, weights: np.ndarray) -> float:
        """Compute portfolio expected return"""
        returns_df = self.get_asset_returns_df()
        if returns_df is None or returns_df.empty:
            return 0.0  # Return neutral value instead of -inf
        
        try:
            # Ensure weights are valid
            if np.isnan(weights).any() or not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
                return 0.0
            
            # Calculate mean returns for each asset
            mean_returns = returns_df.mean()
            
            # Handle missing or invalid returns
            if mean_returns.isna().any():
                return 0.0
            
            # Ensure alignment between weights and returns
            if len(weights) != len(mean_returns):
                return 0.0
                
            # Calculate expected portfolio return with safeguards
            portfolio_return = np.sum(weights * mean_returns.values)  # Use .values to ensure numpy array
            
            # Handle extreme values
            if not np.isfinite(portfolio_return):
                return 0.0
                
            return -portfolio_return  # Negative because we minimize in scipy
            
        except Exception:
            return 0.0  # Return neutral value on error

    def get_optim_weights(self):
        """
        Optimize portfolio weights to maximize returns subject to constraints.
        """
        returns_df = self.get_asset_returns_df()
        if returns_df is None:
            return None
            
        # Use the actual number of assets in the returns DataFrame
        n_assets = len(returns_df.columns)
        if n_assets == 0:
            return None
            
        # Ensure symbols match between portfolio and returns
        if set(returns_df.columns) != set(self.symbols):
            return None
            
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Set bounds (0-30% per asset)
        bounds = tuple((0, 0.3) for _ in range(n_assets))
        
        try:
            result = minimize(
                fun=self.compute_expected_return,
                x0=initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
                ],
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                # Ensure weights sum to 1 and are non-negative
                weights = np.maximum(result.x, 0)  # Ensure non-negative
                weights = weights / np.sum(weights)  # Normalize to sum to 1
                result.x = weights
                return result
            
            return None
                
        except Exception:
            return None

    def get_total_value(self, current_prices: Dict[str, dict] = None) -> float:
        """Calculate total portfolio value including cash"""
        # Track processed assets
        processed_assets = set()
        total_value = self.cash
        
        for symbol, row in self.portfolio_df.iterrows():
            if symbol in processed_assets:
                continue
                
            processed_assets.add(symbol)
            size = row['size']
            if isinstance(size, pd.Series):
                size = size.iloc[0]
            size = float(size)
            
            # Get current price
            if current_prices and symbol in current_prices:
                current_price = float(current_prices[symbol]['markPrice'])
            else:
                # Fallback to asset's last price
                asset = row['asset']
                if len(asset.prices) > 0:
                    current_price = float(asset.prices[-1])
                else:
                    continue
            
            position_value = size * current_price
            total_value += position_value
        
        return total_value