from typing import Dict, List
import pandas as pd
import talib
import logging
from ..base import LeverageStrategy
from config.regime import RegimeConfig

class RegimeLeverageStrategy(LeverageStrategy):
    def __init__(self):
        super().__init__()

    def configure(self) -> None:
        self.is_configured = True

    def get_leverages(self, data: pd.DataFrame) -> Dict[str, int]:
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain the following columns: {required_columns}")
        
        if not self.is_configured:
            raise ValueError("Leverage Strategy is not configured")
        
        indicators = self._calculate_indicators(data)
        signals = {
            name: calculator.calculate_indicators(indicators)
            for name, calculator in self.signal_calculators.items()
        }
        weighted_signal = sum(signals[name] * RegimeConfig.SignalBlendConfig.WEIGHTS[name] for name in signals)
        return weighted_signal

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
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
        open = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        # Calculate Moving Averages
        indicators = {
            'open': open,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'ma_short': talib.SMA(close, timeperiod=RegimeConfig.IndicatorConfig.MA_SHORT),
            'ma_long': talib.SMA(close, timeperiod=RegimeConfig.IndicatorConfig.MA_LONG),
            'plus_di': talib.PLUS_DI(high, low, close, timeperiod=RegimeConfig.IndicatorConfig.ADX_PERIOD),
            'minus_di': talib.MINUS_DI(high, low, close, timeperiod=RegimeConfig.IndicatorConfig.ADX_PERIOD),
            'adx': talib.ADX(high, low, close, timeperiod=RegimeConfig.IndicatorConfig.ADX_PERIOD),
            'rsi': talib.RSI(close, timeperiod=RegimeConfig.IndicatorConfig.RSI_PERIOD),
            'atr': talib.ATR(high, low, close, timeperiod=RegimeConfig.IndicatorConfig.ATR_PERIOD),
            'psar_fast': talib.SAR(high, low, acceleration=RegimeConfig.PsarConfig.FAST_ACCELERATION, maximum=RegimeConfig.PsarConfig.FAST_MAXIMUM),
            'psar_slow': talib.SAR(high, low, acceleration=RegimeConfig.PsarConfig.SLOW_ACCELERATION, maximum=RegimeConfig.PsarConfig.SLOW_MAXIMUM)
        }

        # Calculate Ichimoku
        tenkan = talib.SMA(((high + low) / 2), timeperiod=RegimeConfig.IchimokuConfig.TENKAN_PERIOD)
        kijun = talib.SMA(((high + low) / 2), timeperiod=RegimeConfig.IchimokuConfig.KIJUN_PERIOD)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = talib.SMA(((high + low) / 2), timeperiod=RegimeConfig.IchimokuConfig.SENKOU_B_PERIOD)

        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close, 
            fastperiod=RegimeConfig.IndicatorConfig.MACD_FAST, 
            slowperiod=RegimeConfig.IndicatorConfig.MACD_SLOW, 
            signalperiod=RegimeConfig.IndicatorConfig.MACD_SIGNAL
        )
        
        # Calculate Bollinger Bands
        bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(
            close, 
            timeperiod=RegimeConfig.BollingerBandsConfig.PERIOD,
            nbdevup=RegimeConfig.BollingerBandsConfig.NBDEVUP,
            nbdevdn=RegimeConfig.BollingerBandsConfig.NBDEVDN,
            matype=RegimeConfig.BollingerBandsConfig.MATYPE
        )

        indicators.update({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'bollinger_upper': bollinger_upper,
            'bollinger_middle': bollinger_middle,
            'bollinger_lower': bollinger_lower
        })

        return indicators