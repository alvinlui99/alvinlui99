from typing import Dict
import pandas as pd
from .indicator_calculator import IndicatorCalculator
from config import RegimeConfig

class PrimaryTrendsCalculator(IndicatorCalculator):
    def calculate_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        indicator_calculators = {
            'ma_signal': MovingAverageIndicatorCalculator(),
            'ichimoku_signal': IchimokuIndicatorCalculator(),
            'dmi_signal': DmiIndicatorCalculator(),
            'psar_signal': ParabolicSARIndicatorCalculator(),
            'bbands_signal': BollingerBandsIndicatorCalculator()
        }
        signals = {
            name: calculator.calculate_indicators(indicators)
            for name, calculator in indicator_calculators.items()
        }
        weighted_signal = sum(signals[name] * RegimeConfig.PrimaryTrendConfig.WEIGHTS[name] for name in signals)
        return max(min(weighted_signal, 1.0), -1.0)

class MovingAverageIndicatorCalculator(IndicatorCalculator):
    def calculate_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate Moving Average signals with continuous strength values.
        Returns a score between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. MA Cross signal
        2. Price position relative to MAs
        3. MA slope (trend direction)
        4. MA spread (trend strength)
        """
        close = indicators['close']
        ma_short = indicators['ma_short']
        ma_long = indicators['ma_long']
        
        current_price = close.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        # 1. MA Cross Signal (-1 to 1)
        # Normalize the difference between MAs
        ma_diff = current_ma_short - current_ma_long
        ma_norm_factor = current_ma_long * RegimeConfig.MovingAverageConfig.NORM_FACTOR
        cross_signal = min(max(ma_diff / ma_norm_factor, -1.0), 1.0)
        
        # 2. Price Position Signal (-1 to 1)
        # Calculate distance from both MAs
        short_diff = (current_price - current_ma_short) / current_ma_short
        long_diff = (current_price - current_ma_long) / current_ma_long
        position_signal = min(max((short_diff + long_diff) * RegimeConfig.MovingAverageConfig.POSITION_SINGAL_AMPLIFIER, -1.0), 1.0)
        
        # 3. MA Slope Signal (-1 to 1)
        # Calculate slopes of both MAs
        short_slope = (current_ma_short - ma_short.iloc[-RegimeConfig.MovingAverageConfig.SLOPE_DURATION]) / ma_short.iloc[-RegimeConfig.MovingAverageConfig.SLOPE_DURATION]
        long_slope = (current_ma_long - ma_long.iloc[-RegimeConfig.MovingAverageConfig.SLOPE_DURATION]) / ma_long.iloc[-RegimeConfig.MovingAverageConfig.SLOPE_DURATION]
        slope_signal = min(max((short_slope + long_slope) * RegimeConfig.MovingAverageConfig.SLOPE_SINGAL_AMPLIFIER, -1.0), 1.0)
        
        # 4. MA Spread Signal (0 to 1)
        # Normalize the spread between MAs
        ma_spread = abs(ma_diff) / current_ma_long
        historical_spread = pd.Series(abs(ma_short - ma_long) / ma_long).rolling(window=RegimeConfig.MovingAverageConfig.HISTORICAL_SPREAD_DURATION).mean()
        spread_signal = min(ma_spread / historical_spread.iloc[-1], 2.0) / 2
        
        # Combine signals with weights
        weights = RegimeConfig.MovingAverageConfig.WEIGHTS
        
        composite_signal = (
            weights['cross'] * cross_signal +
            weights['position'] * position_signal +
            weights['slope'] * slope_signal +
            weights['spread'] * spread_signal * (1 if cross_signal > 0 else -1)
        )
        
        return max(min(composite_signal, 1.0), -1.0)

class IchimokuIndicatorCalculator(IndicatorCalculator):
    def calculate_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud signals with continuous strength values.
        Returns a score between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Cloud position and distance
        2. Tenkan-Kijun cross position
        3. Cloud thickness
        4. Price momentum relative to cloud
        """
        close = indicators['close']
        high = indicators['high']
        low = indicators['low']
        
        tenkan = indicators['tenkan']
        kijun = indicators['kijun']
        senkou_a = indicators['senkou_a']
        senkou_b = indicators['senkou_b']
        
        current_price = close.iloc[-1]
        current_senkou_a = senkou_a.iloc[-1]
        current_senkou_b = senkou_b.iloc[-1]
        current_tenkan = tenkan.iloc[-1]
        current_kijun = kijun.iloc[-1]
        
        # 1. Cloud position signal (-1 to 1)
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)
        cloud_thickness = cloud_top - cloud_bottom
        
        # Normalize distance from cloud
        if current_price > cloud_top:
            cloud_position = min((current_price - cloud_top) / cloud_thickness, 1.0)
        elif current_price < cloud_bottom:
            cloud_position = max((current_price - cloud_bottom) / cloud_thickness, -1.0)
        else:
            # Price inside cloud - normalize position within cloud
            cloud_position = (current_price - cloud_bottom) / cloud_thickness - 0.5
        
        # 2. Tenkan-Kijun Cross signal (-1 to 1)
        tk_diff = current_tenkan - current_kijun
        tk_signal = min(max(tk_diff / (cloud_thickness * 0.5), -1.0), 1.0)
        
        # 3. Cloud thickness signal (0 to 1)
        # Thicker cloud indicates stronger trend
        avg_price = (high.iloc[-RegimeConfig.IchimokuConfig.CLOUD_THICKNESS_DURATION:] + low.iloc[-RegimeConfig.IchimokuConfig.CLOUD_THICKNESS_DURATION:]).mean() / 2
        normalized_thickness = min(cloud_thickness / (avg_price * 0.1), 1.0)
        
        # 4. Price momentum
        price_momentum = (current_price - close.iloc[-RegimeConfig.IchimokuConfig.MOMENTUM_DURATION]) / close.iloc[-RegimeConfig.IchimokuConfig.MOMENTUM_DURATION]
        momentum_signal = min(max(price_momentum * RegimeConfig.IchimokuConfig.MOMENTUM_AMPLIFIER, -1.0), 1.0)
        
        # Combine signals with weights
        weights = RegimeConfig.IchimokuConfig.WEIGHTS
        
        composite_signal = (
            weights['cloud_position'] * cloud_position +
            weights['tk_cross'] * tk_signal +
            weights['thickness'] * normalized_thickness * (1 if cloud_position > 0 else -1) +
            weights['momentum'] * momentum_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
class DmiIndicatorCalculator(IndicatorCalculator):
    def calculate_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate DMI/ADX trend signals with continuous strength values.
        Returns a score between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. DI+ vs DI- relative strength
        2. ADX trend strength
        3. ADX slope for trend momentum
        4. DI crossover intensity
        """
        high = indicators['high']
        low = indicators['low']
        close = indicators['close']
        plus_di = indicators['plus_di']
        minus_di = indicators['minus_di']
        adx = indicators['adx']
        
        # Current values
        plus_di_current = plus_di.iloc[-1]
        minus_di_current = minus_di.iloc[-1]
        adx_current = adx.iloc[-1]
        
        # 1. DI Relative Strength (-1 to 1)
        di_sum = plus_di_current + minus_di_current
        if di_sum > 0:
            di_strength = (plus_di_current - minus_di_current) / di_sum
        else:
            di_strength = 0
        
        # 2. ADX Trend Strength (0 to 1)
        # ADX > 25 indicates strong trend
        # ADX > 50 indicates extremely strong trend
        adx_strength = min(adx_current / RegimeConfig.DMIConfig.ADX_STRENGTH_THRESHOLD, 1.0)
        
        # 3. ADX Momentum (-1 to 1)
        adx_slope = (adx_current - adx.iloc[-RegimeConfig.DMIConfig.ADX_MOMENTUM_DURATION]) / adx.iloc[-RegimeConfig.DMIConfig.ADX_MOMENTUM_DURATION]
        adx_momentum = min(max(adx_slope * RegimeConfig.DMIConfig.ADX_MOMENTUM_AMPLIFIER, -1.0), 1.0)
        
        # 4. DI Crossover Intensity (-1 to 1)
        di_diff = plus_di_current - minus_di_current
        di_diff_prev = plus_di.iloc[-2] - minus_di.iloc[-2]
        crossover_intensity = min(max((di_diff - di_diff_prev) / RegimeConfig.DMIConfig.DI_CROSSOVER_INTENSITY_AMPLIFIER, -1.0), 1.0)
        
        # Combine signals with weights
        weights = RegimeConfig.DMIConfig.WEIGHTS
        
        composite_signal = (
            weights['di_strength'] * di_strength * adx_strength +  # Scale DI strength by ADX strength
            weights['adx_strength'] * adx_strength * (1 if di_strength > 0 else -1) +
            weights['adx_momentum'] * adx_momentum * (1 if di_strength > 0 else -1) +
            weights['crossover'] * crossover_intensity
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
class ParabolicSARIndicatorCalculator(IndicatorCalculator):
    def calculate_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate Parabolic SAR signals with continuous strength values.
        Returns a score between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Position relative to price
        2. Distance from price (normalized)
        3. SAR reversal momentum
        4. Price velocity relative to SAR
        """
        close = indicators['close']
        psar_fast = indicators['psar_fast']
        psar_slow = indicators['psar_slow']
        current_price = close.iloc[-1]
        current_psar_fast = psar_fast.iloc[-1]
        current_psar_slow = psar_slow.iloc[-1]
        
        # 1. Basic position signal (-1 or 1)
        fast_position = 1 if current_price > current_psar_fast else -1
        slow_position = 1 if current_price > current_psar_slow else -1
        
        # 2. Distance signal (0 to 1)
        # Normalize distance as percentage of price
        fast_distance = abs(current_price - current_psar_fast) / current_price
        slow_distance = abs(current_price - current_psar_slow) / current_price
        
        # Cap the distance signal
        max_distance_factor = RegimeConfig.PsarConfig.MAX_DISTANCE_FACTOR
        normalized_fast_distance = min(fast_distance / max_distance_factor, 1.0)
        normalized_slow_distance = min(slow_distance / max_distance_factor, 1.0)
        
        # 3. SAR Reversal momentum (-1 to 1)
        # Check if SAR recently flipped
        fast_reversal = (
            1 if psar_fast.iloc[-2] > close.iloc[-2] and current_price > current_psar_fast
            else -1 if psar_fast.iloc[-2] < close.iloc[-2] and current_price < current_psar_fast
            else 0
        )
        
        # 4. Price velocity relative to SAR
        price_velocity = (current_price - close.iloc[-RegimeConfig.PsarConfig.VELOCITY_DURATION]) / close.iloc[-RegimeConfig.PsarConfig.VELOCITY_DURATION]
        velocity_signal = min(max(price_velocity * RegimeConfig.PsarConfig.VELOCITY_AMPLIFIER, -1.0), 1.0)
        
        # Combine signals with weights
        weights = RegimeConfig.PsarConfig.WEIGHTS
        
        composite_signal = (
            weights['fast_signal'] * fast_position * normalized_fast_distance +
            weights['slow_signal'] * slow_position * normalized_slow_distance +
            weights['reversal'] * fast_reversal +
            weights['velocity'] * velocity_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
class BollingerBandsIndicatorCalculator(IndicatorCalculator):
    def calculate_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands trend signals with continuous strength values.
        Returns a score between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Price position relative to bands
        2. Bandwidth (trend strength/volatility)
        3. Band slope (trend direction)
        4. Price velocity relative to middle band
        """
        close = indicators['close']
        upper = indicators['bollinger_upper']
        middle = indicators['bollinger_middle']
        lower = indicators['bollinger_lower']
        
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]
        
        # 1. Position Signal (-1 to 1)
        # Normalize price position within bands
        band_width = current_upper - current_lower
        relative_position = (current_close - current_lower) / band_width if band_width > 0 else 0
        position_signal = 2 * (relative_position - 0.5)  # Convert 0-1 to -1 to 1
        
        # 2. Bandwidth Signal (0 to 1)
        # Normalize bandwidth relative to middle band
        normalized_bandwidth = band_width / current_middle
        bandwidth_sma = pd.Series(normalized_bandwidth).rolling(window=RegimeConfig.BollingerBandsConfig.BANDWIDTH_SMA_PERIOD).mean().iloc[-1]
        bandwidth_signal = min(normalized_bandwidth / bandwidth_sma, 2.0) / 2  # 0 to 1
        
        # 3. Band Slope Signal (-1 to 1)
        middle_slope = (current_middle - middle.iloc[-RegimeConfig.BollingerBandsConfig.SLOPE_DURATION]) / middle.iloc[-RegimeConfig.BollingerBandsConfig.SLOPE_DURATION]
        slope_signal = min(max(middle_slope * RegimeConfig.BollingerBandsConfig.SLOPE_AMPLIFIER, -1.0), 1.0)
        
        # 4. Price Velocity Signal (-1 to 1)
        # Measure price movement relative to middle band
        price_velocity = (current_close - close.iloc[-RegimeConfig.BollingerBandsConfig.VELOCITY_DURATION]) / close.iloc[-RegimeConfig.BollingerBandsConfig.VELOCITY_DURATION]
        velocity_signal = min(max(price_velocity * RegimeConfig.BollingerBandsConfig.VELOCITY_AMPLIFIER, -1.0), 1.0)
        
        # Combine signals with weights
        weights = RegimeConfig.BollingerBandsConfig.WEIGHTS
        
        composite_signal = (
            weights['position'] * position_signal +
            weights['bandwidth'] * bandwidth_signal * (1 if position_signal > 0 else -1) +
            weights['slope'] * slope_signal +
            weights['velocity'] * velocity_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
