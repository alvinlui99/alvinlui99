import pandas as pd
import numpy as np
import talib
import logging
from typing import List, Dict, Optional, Union, Tuple

class FeatureEngineer:
    """
    Feature engineering class for algorithmic trading.
    
    This class provides methods to calculate various technical indicators
    organized by category (trend, momentum, volatility, volume, custom).
    """
    
    def __init__(self, feature_config: Optional[Dict] = None, logger=None):
        """
        Initialize the feature engineer with optional configuration.
        
        Args:
            feature_config: Optional dictionary with feature configuration settings
            logger: Optional logger instance
        """
        self.feature_config = feature_config or self._default_feature_config()
        self.logger = logger or logging.getLogger(__name__)
    
    def _default_feature_config(self) -> Dict:
        """
        Default feature configuration.
        
        Returns:
            Dictionary with default feature configuration
        """
        return {
            'trend': {
                'enabled': True,
                'sma_periods': [5, 10, 20, 50],
                'ema_periods': [5, 10, 20]
            },
            'momentum': {
                'enabled': True,
                'rsi_periods': [14],
                'macd': True
            },
            'volatility': {
                'enabled': True,
                'bb_period': 20,
                'atr_period': 14
            },
            'volume': {
                'enabled': True,
                'obv': True,
                'volume_sma_periods': [5]
            },
            'custom': {
                'enabled': True,
                'price_to_sma': True,
                'returns': [1, 3, 5],
                'volatility_periods': [5, 10]
            }
        }
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all configured technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Ensure all OHLCV columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Log original shape for debugging
        original_shape = df.shape
        self.logger.debug(f"Starting feature engineering. Original shape: {original_shape}")
        
        # Extract OHLCV data
        try:
            ohlcv = self._extract_ohlcv(df)
            
            # Add indicators by category based on configuration
            if self.feature_config.get('trend', {}).get('enabled', False):
                try:
                    df = self.add_trend_indicators(df, ohlcv)
                    self.logger.debug(f"Added trend indicators. New shape: {df.shape}")
                except Exception as e:
                    self.logger.warning(f"Error adding trend indicators: {str(e)}")
            
            if self.feature_config.get('momentum', {}).get('enabled', False):
                try:
                    df = self.add_momentum_indicators(df, ohlcv)
                    self.logger.debug(f"Added momentum indicators. New shape: {df.shape}")
                except Exception as e:
                    self.logger.warning(f"Error adding momentum indicators: {str(e)}")
            
            if self.feature_config.get('volatility', {}).get('enabled', False):
                try:
                    df = self.add_volatility_indicators(df, ohlcv)
                    self.logger.debug(f"Added volatility indicators. New shape: {df.shape}")
                except Exception as e:
                    self.logger.warning(f"Error adding volatility indicators: {str(e)}")
            
            if self.feature_config.get('volume', {}).get('enabled', False):
                try:
                    df = self.add_volume_indicators(df, ohlcv)
                    self.logger.debug(f"Added volume indicators. New shape: {df.shape}")
                except Exception as e:
                    self.logger.warning(f"Error adding volume indicators: {str(e)}")
            
            if self.feature_config.get('custom', {}).get('enabled', False):
                try:
                    df = self.add_custom_indicators(df)
                    self.logger.debug(f"Added custom indicators. New shape: {df.shape}")
                except Exception as e:
                    self.logger.warning(f"Error adding custom indicators: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            # Still return the original dataframe even if feature engineering fails
        
        # Drop timestamp column if it exists
        if 'Open_time' in df.columns:
            df = df.drop(['Open_time'], axis=1)
            
        # Drop rows with NaN values
        original_rows = len(df)
        df = df.dropna()
        dropped_rows = original_rows - len(df)
        
        if dropped_rows > 0:
            self.logger.info(f"Dropped {dropped_rows} rows with NaN values (expected due to indicator lookback periods)")
        
        # Verify we have enough data
        if len(df) < 10:  # Minimum required rows
            self.logger.error("Insufficient data rows after processing")
            return pd.DataFrame()  # Return empty DataFrame to signal error
            
        return df
    
    def _extract_ohlcv(self, df: pd.DataFrame) -> Tuple:
        """
        Extract OHLCV data from dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple with open, high, low, close, volume arrays
        """
        # Ensure all data is converted to float64 (double) to avoid talib errors
        return (
            df['Open'].astype(np.float64).values,
            df['High'].astype(np.float64).values,
            df['Low'].astype(np.float64).values,
            df['Close'].astype(np.float64).values,
            df['Volume'].astype(np.float64).values
        )
    
    def add_trend_indicators(self, df: pd.DataFrame, 
                            ohlcv: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Add trend-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            ohlcv: Optional tuple with OHLCV arrays
            
        Returns:
            DataFrame with added trend indicators
        """
        if ohlcv is None:
            ohlcv = self._extract_ohlcv(df)
        
        open_price, high_price, low_price, close_price, volume = ohlcv
        
        # Simple Moving Averages
        for period in self.feature_config.get('trend', {}).get('sma_periods', []):
            df[f'SMA{period}'] = talib.SMA(close_price, timeperiod=period)
        
        # Exponential Moving Averages
        for period in self.feature_config.get('trend', {}).get('ema_periods', []):
            df[f'EMA{period}'] = talib.EMA(close_price, timeperiod=period)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame, 
                               ohlcv: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Add momentum-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            ohlcv: Optional tuple with OHLCV arrays
            
        Returns:
            DataFrame with added momentum indicators
        """
        if ohlcv is None:
            ohlcv = self._extract_ohlcv(df)
        
        open_price, high_price, low_price, close_price, volume = ohlcv
        
        # Relative Strength Index
        for period in self.feature_config.get('momentum', {}).get('rsi_periods', []):
            df[f'RSI{period}'] = talib.RSI(close_price, timeperiod=period)
        
        # MACD
        if self.feature_config.get('momentum', {}).get('macd', False):
            macd, macd_signal, macd_hist = talib.MACD(
                close_price, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame, 
                                 ohlcv: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Add volatility-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            ohlcv: Optional tuple with OHLCV arrays
            
        Returns:
            DataFrame with added volatility indicators
        """
        if ohlcv is None:
            ohlcv = self._extract_ohlcv(df)
        
        open_price, high_price, low_price, close_price, volume = ohlcv
        
        # Bollinger Bands
        bb_period = self.feature_config.get('volatility', {}).get('bb_period', 20)
        upperband, middleband, lowerband = talib.BBANDS(
            close_price, 
            timeperiod=bb_period, 
            nbdevup=2, 
            nbdevdn=2
        )
        df['BB_upper'] = upperband
        df['BB_middle'] = middleband
        df['BB_lower'] = lowerband
        df['BB_width'] = (upperband - lowerband) / middleband
        
        # Average True Range
        atr_period = self.feature_config.get('volatility', {}).get('atr_period', 14)
        df['ATR'] = talib.ATR(high_price, low_price, close_price, timeperiod=atr_period)
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame, 
                             ohlcv: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Add volume-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            ohlcv: Optional tuple with OHLCV arrays
            
        Returns:
            DataFrame with added volume indicators
        """
        if ohlcv is None:
            ohlcv = self._extract_ohlcv(df)
        
        open_price, high_price, low_price, close_price, volume = ohlcv
        
        # On-Balance Volume
        if self.feature_config.get('volume', {}).get('obv', False):
            df['OBV'] = talib.OBV(close_price, volume)
        
        # Volume SMA
        for period in self.feature_config.get('volume', {}).get('volume_sma_periods', []):
            vol_sma = talib.SMA(volume, timeperiod=period)
            df[f'Volume_SMA{period}'] = vol_sma
            
            # Calculate volume ratio with protection against division by zero
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-10  # A very small number
            # Use numpy's where to handle the division safely
            ratio = np.divide(volume, vol_sma, out=np.zeros_like(volume), where=vol_sma!=0)
            df[f'Volume_ratio_{period}'] = ratio
        
        return df
    
    def add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added custom indicators
        """
        # Make sure 'Close' is available for calculations
        if 'Close' not in df.columns:
            self.logger.warning("'Close' column not found, skipping custom indicators")
            return df
        
        # Price relative to moving averages
        if self.feature_config.get('custom', {}).get('price_to_sma', False):
            if 'SMA5' in df.columns:
                df['Price_to_SMA5'] = df['Close'] / df['SMA5'] - 1
            if 'SMA20' in df.columns:
                df['Price_to_SMA20'] = df['Close'] / df['SMA20'] - 1
            if 'SMA50' in df.columns:
                df['Price_to_SMA50'] = df['Close'] / df['SMA50'] - 1
        
        # Price returns over different periods
        for period in self.feature_config.get('custom', {}).get('returns', []):
            try:
                df[f'Return_{period}d'] = df['Close'].pct_change(period)
            except Exception as e:
                self.logger.warning(f"Error calculating {period}-day returns: {str(e)}")
        
        # Price volatility over different periods
        for period in self.feature_config.get('custom', {}).get('volatility_periods', []):
            try:
                df[f'Volatility_{period}'] = df['Close'].rolling(period).std() / df['Close']
            except Exception as e:
                self.logger.warning(f"Error calculating {period}-day volatility: {str(e)}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features that would be generated with current config.
        
        Returns:
            List of feature names
        """
        # This method would need a sample dataframe to fully implement
        # It would generate features and return their names
        # For now, it returns an empty list as a placeholder
        return []
    
    def update_config(self, new_config: Dict) -> None:
        """
        Update the feature configuration.
        
        Args:
            new_config: New configuration dictionary to merge with current config
        """
        # Deep update of configuration
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.feature_config = update_dict(self.feature_config, new_config) 