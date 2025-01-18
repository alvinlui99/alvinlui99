import numpy as np
import pandas as pd
from typing import Dict, Union
import talib

class RegimeDetector:
    """
    A class focused on trend regime detection using multiple technical indicators:
    - Moving Average Crossovers
    - ADX (Average Directional Index) for trend strength
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    """
    
    def __init__(self, 
                 lookback_period: int = 252,
                 ma_short: int = 50,
                 ma_long: int = 200,
                 adx_period: int = 14,
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        """
        Initialize the RegimeDetector with trend detection parameters.
        
        Parameters:
        -----------
        lookback_period : int
            Number of days to look back for regime detection
        ma_short : int
            Short-term moving average period
        ma_long : int
            Long-term moving average period
        adx_period : int
            Period for ADX calculation
        rsi_period : int
            Period for RSI calculation
        macd_fast/slow/signal : int
            MACD parameters
        """
        self.lookback_period = lookback_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.current_regime = None
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Union[float, int]]:
        """
        Calculate various technical indicators for trend detection.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series with enough history for calculations
            
        Returns:
        --------
        dict : Dictionary containing all calculated indicators
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Moving Averages
        ma_short = talib.SMA(close, timeperiod=self.ma_short)
        ma_long = talib.SMA(close, timeperiod=self.ma_long)
        
        # Calculate ADX
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        
        # Calculate RSI
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close, 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        return {
            'ma_short': ma_short.iloc[-1],
            'ma_long': ma_long.iloc[-1],
            'adx': adx.iloc[-1],
            'rsi': rsi.iloc[-1],
            'macd': macd.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'macd_hist': macd_hist.iloc[-1]
        }
    
    def detect_trend_regime(self, data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, Union[int, float, dict]]:
        """
        Detect market regime based on multiple trend indicators.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
            
        Returns:
        --------
        dict : Contains regime classification and confidence metrics
        """
        indicators = self.calculate_indicators(self.prepare_data(data, weights))
        
        # Initialize trend signals
        signals = {
            'ma_crossover': 0,
            'adx_trend': 0,
            'rsi_trend': 0,
            'macd_trend': 0
        }
        
        # 1. Moving Average Crossover
        signals['ma_crossover'] = 1 if indicators['ma_short'] > indicators['ma_long'] else -1
        
        # 2. ADX Trend Strength (Above 25 indicates strong trend)
        signals['adx_trend'] = 1 if indicators['adx'] > 25 else 0
        
        # 3. RSI Trend
        if indicators['rsi'] > 60:
            signals['rsi_trend'] = 1
        elif indicators['rsi'] < 40:
            signals['rsi_trend'] = -1
            
        # 4. MACD Trend
        signals['macd_trend'] = 1 if indicators['macd'] > indicators['macd_signal'] else -1
        
        # Combine signals to determine regime
        trend_score = (
            signals['ma_crossover'] * 0.4 +  # 40% weight
            signals['adx_trend'] * 0.2 +     # 20% weight
            signals['rsi_trend'] * 0.2 +     # 20% weight
            signals['macd_trend'] * 0.2      # 20% weight
        )
        
        # Determine regime
        if trend_score > 0.3:
            regime = 1  # Bullish
        elif trend_score < -0.3:
            regime = -1  # Bearish
        else:
            regime = 0  # Neutral
            
        self.current_regime = regime
    
    def get_regime_leverage(self, data: Dict[str, pd.DataFrame], weights: Dict[str, float], base_leverage: float = 1.0) -> float:
        """
        Get suggested leverage based on current regime.
        
        Parameters:
        -----------
        base_leverage : float
            Base leverage to adjust
            
        Returns:
        --------
        float : Suggested leverage
        """
        self.detect_trend_regime(data, weights)
        if self.current_regime is None:
            return base_leverage
            
        # Leverage adjustment logic based on regime
        leverage_multipliers = {
            -1: 1,    # Reduce leverage in bearish regime
            0: 2,    # Slightly reduce leverage in neutral regime
            1: 3      # Full leverage in bullish regime
        }
        
        return base_leverage * leverage_multipliers[self.current_regime]
    
    def prepare_data(self, data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for trend detection
        """
        output = pd.DataFrame(columns=['high', 'low', 'close'])
        # Initialize weighted price series
        weighted_high = pd.Series(0.0, index=next(iter(data.values())).index)
        weighted_low = pd.Series(0.0, index=next(iter(data.values())).index)
        weighted_close = pd.Series(0.0, index=next(iter(data.values())).index)

        # Calculate weighted sum for each price type
        for symbol, weight in weights.items():
            df = data[symbol]
            weighted_high += df['high'] * weight
            weighted_low += df['low'] * weight 
            weighted_close += df['price'] * weight

        # Store in output DataFrame
        output['high'] = weighted_high
        output['low'] = weighted_low
        output['close'] = weighted_close

        return output