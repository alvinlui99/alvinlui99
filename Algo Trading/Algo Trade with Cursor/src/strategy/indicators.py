import pandas as pd
import numpy as np
from typing import Tuple

def calculate_sma(data: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data (pd.DataFrame): Price data
        period (int): Period for SMA
        column (str): Column to calculate SMA on
        
    Returns:
        pd.Series: SMA values
    """
    return data[column].rolling(window=period).mean()

def calculate_ema(data: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data (pd.DataFrame): Price data
        period (int): Period for EMA
        column (str): Column to calculate EMA on
        
    Returns:
        pd.Series: EMA values
    """
    return data[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        data (pd.DataFrame): Price data
        period (int): Period for RSI
        
    Returns:
        pd.Series: RSI values
    """
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.DataFrame, 
                  fast_period: int = 12,
                  slow_period: int = 26,
                  signal_period: int = 9,
                  column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data (pd.DataFrame): Price data
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        column (str): Column to calculate MACD on
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: MACD line, Signal line, and Histogram
    """
    fast_ema = calculate_ema(data, fast_period, column)
    slow_ema = calculate_ema(data, slow_period, column)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data: pd.DataFrame, 
                            period: int = 20,
                            std_dev: float = 2.0,
                            column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.DataFrame): Price data
        period (int): Period for moving average
        std_dev (float): Number of standard deviations
        column (str): Column to calculate Bollinger Bands on
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: Upper band, Middle band, Lower band
    """
    middle_band = calculate_sma(data, period, column)
    std = data[column].rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band 