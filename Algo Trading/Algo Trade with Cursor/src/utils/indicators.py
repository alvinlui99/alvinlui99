import pandas as pd
import numpy as np
from typing import Tuple

class PairIndicators:
    @staticmethod
    def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two DataFrames on their timestamps.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Tuple of aligned DataFrames
        """
        df1.index = df1.timestamp
        df2.index = df2.timestamp
        common_index = df1.index.intersection(df2.index)
        return df1.loc[common_index], df2.loc[common_index]
    
    @staticmethod
    def calculate_correlation(df1: pd.DataFrame, df2: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate rolling correlation between two price series.
        
        Args:
            df1: First DataFrame with 'returns' column
            df2: Second DataFrame with 'returns' column
            window: Rolling window size
            
        Returns:
            Series with rolling correlation values
        """
        df1, df2 = PairIndicators.align_dataframes(df1, df2)
        return df1['returns'].rolling(window).corr(df2['returns'])
    
    @staticmethod
    def calculate_pair_zscore(df1: pd.DataFrame, df2: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate z-score of the price ratio between two assets.
        
        Args:
            df1: First DataFrame with 'close' prices
            df2: Second DataFrame with 'close' prices
            window: Rolling window size
            
        Returns:
            Series with z-score values
        """
        df1, df2 = PairIndicators.align_dataframes(df1, df2)
        ratio = df1['close'] / df2['close']
        return (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std()
    
    @staticmethod
    def calculate_spread_volatility(df1: pd.DataFrame, df2: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate volatility of the price spread between two assets.
        
        Args:
            df1: First DataFrame with 'close' prices
            df2: Second DataFrame with 'close' prices
            window: Rolling window size
            
        Returns:
            Series with spread volatility values
        """
        df1, df2 = PairIndicators.align_dataframes(df1, df2)
        spread = df1['close'] - df2['close']
        return spread.rolling(window).std()

if __name__ == "__main__":
    # Example usage
    from data.collector import BinanceDataCollector
    from data.processor import DataProcessor
    
    # Get and process data
    collector = BinanceDataCollector()
    processor = DataProcessor()
    
    # Get data for two assets
    btc_data = collector.get_historical_klines('BTCUSDT', interval='1h', days_back=7)
    eth_data = collector.get_historical_klines('ETHUSDT', interval='1h', days_back=7)
    
    # Process individual assets
    btc_processed = processor.process_single_asset(btc_data)
    eth_processed = processor.process_single_asset(eth_data)
    
    # Calculate pair indicators
    correlation = PairIndicators.calculate_correlation(btc_processed, eth_processed)
    pair_zscore = PairIndicators.calculate_pair_zscore(btc_processed, eth_processed)
    spread_vol = PairIndicators.calculate_spread_volatility(btc_processed, eth_processed)
    
    # Print results
    print("\nCorrelation between BTC and ETH:")
    print(correlation.tail())
    print("\nPair Z-score:")
    print(pair_zscore.tail())
    print("\nSpread Volatility:")
    print(spread_vol.tail())
