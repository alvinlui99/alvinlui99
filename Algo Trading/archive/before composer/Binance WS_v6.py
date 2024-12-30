import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import deque

def load_csv_to_df(symbol:str, interval:str="15m/", parent_folder:str="../Binance Data/", index_col:str="index"):
    """
    Load and preprocess Binance data from CSV file.
    
    Args:
        symbol: Trading pair symbol
        interval: Timeframe of the data
        parent_folder: Base directory for data files
        index_col: Name of the index column
    """
    try:
        path = parent_folder + interval + symbol + ".csv"
        df = pd.read_csv(
            path,
            index_col=index_col
        )
        df.index = pd.to_datetime(df.index)
        df.index = df.index.astype('int64') // 10**6
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found for {symbol} at {path}")

def df_to_stream(symbol:str, idx, row):
    stream = {
        'symbol': symbol,
        'markPrice': row['Close'],
        'indexPrice': None,
        'estimatedSettlePrice': None,
        'lastFundingRate': None,
        'interestRate': None,
        'nextFundingTime': None,
        'time': idx
        }
    return stream

class Asset:
    """Represents a single trading asset with price and return tracking."""
    
    def __init__(self, symbol: str, maxlen: int = 14):
        """
        Initialize asset with symbol and rolling window size.
        
        Args:
            symbol: Trading pair symbol
            maxlen: Maximum length of rolling window for returns calculation
            
        Raises:
            ValueError: If symbol is empty or maxlen is less than 2
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if maxlen < 2:
            raise ValueError("maxlen must be at least 2")
            
        self.symbol = symbol
        self.prices = deque(maxlen=2)
        self.ret_pct = deque(maxlen=maxlen)

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
        if len(self.prices) == self.prices.maxlen:
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

class Portfolio:
    """Manages a collection of trading assets with portfolio optimization."""
    
    def __init__(self, symbols: list, initial_cash: float = 1000):
        """
        Initialize portfolio with trading symbols and starting cash.
        
        Args:
            symbols: List of trading pair symbols
            initial_cash: Starting portfolio value in base currency
            
        Raises:
            ValueError: If symbols list is empty
        """
        if not symbols:
            raise ValueError("Must provide at least one trading symbol")
            
        self.cash = initial_cash
        self.symbols = symbols
        self.n_assets = len(symbols)
        
        # Initialize portfolio dataframe with assets and default values
        self.portfolio_df = pd.DataFrame([
            {
                'asset': Asset(symbol),
                'price': 0.0,
                'weight': 1/self.n_assets,  # Equal weights initially
                'position': 0.0,
                'return': []
            }
            for symbol in symbols
        ], index=symbols)
        
        # Set correct data types
        self.portfolio_df = self.portfolio_df.astype({
            'price': 'float64',
            'weight': 'float64', 
            'position': 'float64'
        })

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
        self.portfolio_df.loc[symbol, 'price'] = price
        self.portfolio_df.loc[symbol, 'asset'].update_price(price)

    def cal_asset_returns_df(self) -> None:
        """Calculate and update return metrics for all assets."""
        for symbol, row in self.portfolio_df.iterrows():
            asset = row['asset']
            ret = asset.get_return()
            if ret:
                self.portfolio_df.at[symbol, 'return'] = list(ret)

    def get_asset_returns_df(self) -> pd.DataFrame:
        """Get dataframe of asset returns."""
        self.cal_asset_returns_df()
        returns_df = pd.DataFrame(self.portfolio_df['return'].tolist(), index=self.portfolio_df.index)
        return returns_df if not returns_df.empty else None

    def get_asset_cov(self) -> pd.DataFrame:
        """Calculate covariance matrix of asset returns."""
        returns_df = self.get_asset_returns_df()
        return returns_df.cov() if returns_df is not None else None

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
            return weights * budget / weighted_prices
        return pd.Series(0, index=self.portfolio_df.index)

    def compute_volatility(self, weights: np.ndarray = None, 
                         cov: pd.DataFrame = None) -> float:
        """
        Compute portfolio volatility given weights and covariance matrix.
        
        Args:
            weights: Portfolio weights array
            cov: Covariance matrix of returns
            
        Returns:
            Portfolio volatility (annualized)
        """
        if weights is None:
            weights = self.get_weights()
        if cov is None:
            cov = self.get_asset_cov()
            
        # Add debug prints
        print("Weights shape:", weights.shape)
        print("Covariance matrix shape:", cov.shape)
        print("Covariance matrix:\n", cov)
            
        weights = pd.Series(weights).reindex(cov.index)
        weights_array = weights.to_numpy()
        cov_array = cov.to_numpy()
        
        # Check for NaN values
        if np.isnan(cov_array).any():
            print("Warning: NaN values in covariance matrix")
            return float('inf')
        
        variance = np.dot(weights_array.T, np.dot(cov_array, weights_array))
        return np.sqrt(variance)

    def weight_constraint(self, weights: np.ndarray) -> float:
        """Constraint function ensuring weights sum to 1."""
        return np.sum(weights) - 1

    def get_optim_weights(self):
        """
        Optimize portfolio weights to minimize volatility subject to constraints.
        
        Returns:
            Optimization result object if successful, None otherwise
        """
        cov = self.get_asset_cov()
        if cov is None:
            print("No covariance matrix available")
            return None
            
        initial_weights = np.array(self.get_weights())
        bounds = tuple((0, 1) for _ in range(self.n_assets))  # Changed bounds to be non-negative
        
        try:
            result = minimize(
                fun=self.compute_volatility,
                x0=initial_weights,
                args=(cov,),
                method="SLSQP",
                bounds=bounds,
                constraints={'type': 'eq', 'fun': self.weight_constraint}
            )
            
            # Add debug information
            if not result.success:
                print(f"Optimization message: {result.message}")
                print(f"Optimization status: {result.status}")
                
            return result if result.success else None
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None

symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "ADAUSDT"
]

if __name__ == "__main__":
    try:
        data = {symbol: load_csv_to_df(symbol) for symbol in symbols}
        portfolio = Portfolio(symbols)
        i = 0
        for rows in zip(*(df.iterrows() for df in data.values())):
            for (symbol, df), (index, row) in zip(data.items(), rows):
                stream = df_to_stream(symbol, index, row)
                portfolio.update_price(symbol, stream)
                
                # Only optimize when processing the last symbol
                if symbol == symbols[-1]:
                    if portfolio.get_asset_returns_df() is not None:
                        result = portfolio.get_optim_weights()
                        if result is not None and result.success:
                            print(f"Optimized weights: {result.x}")
                        else:
                            print("Optimization failed or insufficient data")
                    else:
                        print("No data available for optimization")
            i += 1
            if i == 10:
                break
    except Exception as e:
        print(f"Error in main execution: {str(e)}")