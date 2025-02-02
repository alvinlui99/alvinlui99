import pandas as pd
import talib
from config import ModelConfig

class TACalculator():
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._calculate_ma(data)
        data = self._calculate_di(data)
        data = self._calculate_adx(data)
        data = self._calculate_rsi(data)
        data = self._calculate_atr(data)
        data = self._calculate_psar(data)
        data = self._calculate_ichimoku(data)
        data = self._calculate_macd(data)
        data = self._calculate_bbands(data)
        return data

    def _calculate_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        data['ma_short'] = talib.SMA(data['close'], timeperiod=ModelConfig.IndicatorConfig.MA_SHORT)
        data['ma_long'] = talib.SMA(data['close'], timeperiod=ModelConfig.IndicatorConfig.MA_LONG)
        return data

    def _calculate_di(self, data: pd.DataFrame) -> pd.DataFrame:
        data['plus_di'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=ModelConfig.IndicatorConfig.ADX_PERIOD)
        data['minus_di'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=ModelConfig.IndicatorConfig.ADX_PERIOD)
        return data

    def _calculate_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=ModelConfig.IndicatorConfig.ADX_PERIOD)
        return data

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        data['rsi'] = talib.RSI(data['close'], timeperiod=ModelConfig.IndicatorConfig.RSI_PERIOD)
        return data

    def _calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=ModelConfig.IndicatorConfig.ATR_PERIOD)
        return data

    def _calculate_psar(self, data: pd.DataFrame) -> pd.DataFrame:
        data['psar_fast'] = talib.SAR(data['high'], data['low'],
                                      acceleration=ModelConfig.PsarConfig.FAST_ACCELERATION,
                                      maximum=ModelConfig.PsarConfig.FAST_MAXIMUM)
        data['psar_slow'] = talib.SAR(data['high'], data['low'],
                                      acceleration=ModelConfig.PsarConfig.SLOW_ACCELERATION,
                                      maximum=ModelConfig.PsarConfig.SLOW_MAXIMUM)
        return data

    def _calculate_ichimoku(self, data: pd.DataFrame) -> pd.DataFrame:
        data['tenkan'] = talib.SMA(((data['high'] + data['low']) / 2), timeperiod=ModelConfig.IchimokuConfig.TENKAN_PERIOD)
        data['kijun'] = talib.SMA(((data['high'] + data['low']) / 2), timeperiod=ModelConfig.IchimokuConfig.KIJUN_PERIOD)
        data['senkou_a'] = (data['tenkan'] + data['kijun']) / 2
        data['senkou_b'] = talib.SMA(((data['high'] + data['low']) / 2), timeperiod=ModelConfig.IchimokuConfig.SENKOU_B_PERIOD)
        return data

    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(
            data['close'], 
            fastperiod=ModelConfig.IndicatorConfig.MACD_FAST, 
            slowperiod=ModelConfig.IndicatorConfig.MACD_SLOW, 
            signalperiod=ModelConfig.IndicatorConfig.MACD_SIGNAL
        )
        return data

    def _calculate_bbands(self, data: pd.DataFrame) -> pd.DataFrame:
        data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'] = talib.BBANDS(
            data['close'], 
            timeperiod=ModelConfig.BollingerBandsConfig.PERIOD,
            nbdevup=ModelConfig.BollingerBandsConfig.NBDEVUP,
            nbdevdn=ModelConfig.BollingerBandsConfig.NBDEVDN,
            matype=ModelConfig.BollingerBandsConfig.MATYPE
        )
        return data