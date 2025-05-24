import pandas as pd
import numpy as np
from typing import Dict

class DataProcessor:
    def __init__(self):
        pass
        
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns for a given price series.
        
        Args:
            df: DataFrame with 'close' prices
            
        Returns:
            DataFrame with added 'returns' column
        """
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility for a price series.
        
        Args:
            df: DataFrame with 'returns' column
            window: Rolling window size
            
        Returns:
            DataFrame with added 'volatility' column
        """
        df = df.copy()
        if 'returns' not in df.columns:
            df = self.calculate_returns(df)
        df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)  # Annualized
        return df
    
    def calculate_zscore(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate z-score for a price series.
        
        Args:
            df: DataFrame with 'close' prices
            window: Rolling window size
            
        Returns:
            DataFrame with added 'zscore' column
        """
        df = df.copy()
        df['zscore'] = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        return df
    
    def process_single_asset(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Process data for a single asset.
        
        Args:
            df: DataFrame with price data
            window: Rolling window size for calculations
            
        Returns:
            DataFrame with added technical indicators
        """
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df, window)
        df = self.calculate_zscore(df, window)
        return df

if __name__ == "__main__":
    # Example usage
    from collector import BinanceDataCollector
    
    # Get data for a single asset
    collector = BinanceDataCollector()
    df = collector.get_historical_klines('BTCUSDT', interval='1h', days_back=7)
    
    # Process data
    processor = DataProcessor()
    processed_df = processor.process_single_asset(df)
    
    # Print results
    print("\nProcessed data for BTCUSDT:")
    print(processed_df.tail())
