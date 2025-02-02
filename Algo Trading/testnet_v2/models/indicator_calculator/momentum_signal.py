from typing import Dict
import pandas as pd
import numpy as np
from .indicator_calculator import SignalCalculator
from config import RegimeConfig

class MomentumSignalCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        indicator_calculators = {
            'rsi_signal': RsiIndicatorCalculator(),
            'macd_signal': MacdIndicatorCalculator(),
            'price_momentum_signal': PriceMomentumIndicatorCalculator(),
            'smi_signal': SmiIndicatorCalculator(),
            'mfi_signal': MfiIndicatorCalculator(),
            'willr_signal': WillrIndicatorCalculator()
        }
        signals = {
            name: calculator.calculate_indicators(indicators)
            for name, calculator in indicator_calculators.items()
        }
        weighted_signal = sum(signals[name] * RegimeConfig.MomentumConfig.WEIGHTS[name] for name in signals)
        return max(min(weighted_signal, 1.0), -1.0)

class RsiIndicatorCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate RSI-based momentum signal.
        Returns a value between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Basic RSI signal (centered and normalized)
        2. RSI momentum (rate of change)
        3. Extreme levels with increased sensitivity
        """
        rsi = indicators['rsi']
        
        # Get current and previous RSI values
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # 1. Basic RSI Signal (-1 to 1)
        # Center RSI around 50 and normalize
        base_signal = (current_rsi - 50) / 50
        
        # 2. RSI Momentum Component (-1 to 1)
        rsi_momentum = (current_rsi - prev_rsi) / 100
        
        # 3. Extreme Levels Component
        extreme_signal = 0.0
        if current_rsi >= RegimeConfig.RSIConfig.OVERBOUGHT_THRESHOLD:
            # Overbought territory
            extreme_signal = ((current_rsi - RegimeConfig.RSIConfig.OVERBOUGHT_THRESHOLD) / 30) * RegimeConfig.RSIConfig.EXTREME_WEIGHT
        elif current_rsi <= RegimeConfig.RSIConfig.OVERSOLD_THRESHOLD:
            # Oversold territory
            extreme_signal = ((current_rsi - RegimeConfig.RSIConfig.OVERSOLD_THRESHOLD) / 30) * RegimeConfig.RSIConfig.EXTREME_WEIGHT
            
        # Combine signals with weights
        composite_signal = (
            RegimeConfig.RSIConfig.BASE_WEIGHT * base_signal +
            RegimeConfig.RSIConfig.MOMENTUM_WEIGHT * rsi_momentum +
            extreme_signal
        )
        
        # Ensure output is between -1 and 1
        return max(min(composite_signal, 1.0), -1.0)

class MacdIndicatorCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate MACD-based momentum signal.
        Returns a value between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Signal line crossover
        2. Histogram strength and momentum
        3. MACD line momentum
        4. Distance from zero line
        """
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_hist']
        
        # 1. Signal Line Crossover (-1 to 1)
        current_diff = macd.iloc[-1] - macd_signal.iloc[-1]
        prev_diff = macd.iloc[-2] - macd_signal.iloc[-2]
        
        # Normalize crossover signal
        crossover_signal = 0.0
        if current_diff > 0 and prev_diff <= 0:  # Bullish crossover
            crossover_signal = 1.0
        elif current_diff < 0 and prev_diff >= 0:  # Bearish crossover
            crossover_signal = -1.0
        else:  # No crossover, use normalized difference
            crossover_signal = current_diff / (abs(macd.iloc[-1]) + 1e-8)  # Avoid division by zero
            
        # 2. Histogram Analysis (-1 to 1)
        current_hist = macd_hist.iloc[-1]
        prev_hist = macd_hist.iloc[-2]
        
        # Histogram strength
        hist_strength = current_hist / (abs(macd.iloc[-1]) + 1e-8)
        hist_strength = max(min(hist_strength, 1.0), -1.0)
        
        # Histogram momentum
        hist_momentum = (current_hist - prev_hist) / (abs(current_hist) + 1e-8)
        hist_momentum = max(min(hist_momentum, 1.0), -1.0)
        
        # 3. MACD Line Momentum (-1 to 1)
        macd_momentum = (macd.iloc[-1] - macd.iloc[-2]) / (abs(macd.iloc[-1]) + 1e-8)
        macd_momentum = max(min(macd_momentum, 1.0), -1.0)
        
        # 4. Zero Line Analysis (-1 to 1)
        zero_line_distance = macd.iloc[-1] / (abs(macd.iloc[-1]) + 1e-8)
        zero_line_distance = max(min(zero_line_distance, 1.0), -1.0)
        
        # Combine signals with weights
        composite_signal = (
            RegimeConfig.MACDConfig.WEIGHTS['crossover'] * crossover_signal +
            RegimeConfig.MACDConfig.WEIGHTS['hist_strength'] * hist_strength +
            RegimeConfig.MACDConfig.WEIGHTS['hist_momentum'] * hist_momentum +
            RegimeConfig.MACDConfig.WEIGHTS['momentum'] * macd_momentum +
            RegimeConfig.MACDConfig.WEIGHTS['zero_line'] * zero_line_distance
        )
        
        return max(min(composite_signal, 1.0), -1.0)

class PriceMomentumIndicatorCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate price momentum signal using multiple timeframes and methods.
        Returns a value between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Rate of Change (RoC) across multiple timeframes
        2. Price acceleration
        3. Moving average convergence
        4. Price velocity relative to ATR
        """
        close = indicators['close']
        atr = indicators['atr']
        ma_short = indicators['ma_short']
        ma_long = indicators['ma_long']
        
        # 1. Multi-timeframe Rate of Change (-1 to 1)
        roc_signals = []
        for period in RegimeConfig.PriceMomentumConfig.PERIODS:
            roc = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
            normalized_roc = self._normalize_value(roc, RegimeConfig.PriceMomentumConfig.ROC_NORMALIZATION_FACTOR)
            roc_signals.append(normalized_roc)
        
        # Weight RoC signals with more weight to shorter timeframes
        weighted_roc = sum(
            signal * weight for signal, weight in 
            zip(roc_signals, RegimeConfig.PriceMomentumConfig.ROC_WEIGHTS)
        )
        
        # 2. Price Acceleration (-1 to 1)
        current_velocity = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        prev_velocity = (close.iloc[-2] - close.iloc[-3]) / close.iloc[-3]
        acceleration = (current_velocity - prev_velocity)
        normalized_acceleration = self._normalize_value(
            acceleration, 
            RegimeConfig.PriceMomentumConfig.ACCELERATION_NORMALIZATION_FACTOR
        )
        
        # 3. Moving Average Convergence (-1 to 1)
        ma_conv_current = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        ma_conv_prev = (ma_short.iloc[-2] - ma_long.iloc[-2]) / ma_long.iloc[-2]
        ma_convergence = self._normalize_value(
            ma_conv_current - ma_conv_prev,
            RegimeConfig.PriceMomentumConfig.MA_CONVERGENCE_NORMALIZATION_FACTOR
        )
        
        # 4. Price Velocity relative to ATR (-1 to 1)
        price_move = close.iloc[-1] - close.iloc[-RegimeConfig.PriceMomentumConfig.VELOCITY_PERIOD]
        atr_normalized_velocity = price_move / (atr.iloc[-1] * RegimeConfig.PriceMomentumConfig.VELOCITY_PERIOD)
        normalized_velocity = self._normalize_value(
            atr_normalized_velocity,
            RegimeConfig.PriceMomentumConfig.VELOCITY_NORMALIZATION_FACTOR
        )
        
        # Combine all signals
        composite_signal = (
            RegimeConfig.PriceMomentumConfig.WEIGHTS['roc'] * weighted_roc +
            RegimeConfig.PriceMomentumConfig.WEIGHTS['acceleration'] * normalized_acceleration +
            RegimeConfig.PriceMomentumConfig.WEIGHTS['ma_convergence'] * ma_convergence +
            RegimeConfig.PriceMomentumConfig.WEIGHTS['velocity'] * normalized_velocity
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _normalize_value(self, value: float, factor: float) -> float:
        """
        Normalize a value to [-1, 1] range using a scaling factor.
        Uses hyperbolic tangent for smooth normalization.
        """
        return np.tanh(value / factor)

class SmiIndicatorCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate Stochastic Momentum Index (SMI) based signal.
        Returns a value between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Basic SMI signal (normalized)
        2. SMI momentum (rate of change)
        3. Signal line crossover
        4. Extreme levels analysis
        """
        high = indicators['high']
        low = indicators['low']
        close = indicators['close']
        
        # Calculate SMI components
        smi_values = self._calculate_smi(high, low, close)
        smi = smi_values['smi']
        signal_line = smi_values['signal']
        
        # Get current and previous values
        current_smi = smi.iloc[-1]
        prev_smi = smi.iloc[-2]
        current_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        
        # 1. Basic SMI Signal (-1 to 1)
        # Center around 0 and normalize
        base_signal = current_smi / 100
        
        # 2. SMI Momentum
        smi_momentum = (current_smi - prev_smi) / 100
        
        # 3. Signal Line Crossover
        current_diff = current_smi - current_signal
        prev_diff = prev_smi - prev_signal
        
        crossover_signal = 0.0
        if current_diff > 0 and prev_diff <= 0:  # Bullish crossover
            crossover_signal = 1.0
        elif current_diff < 0 and prev_diff >= 0:  # Bearish crossover
            crossover_signal = -1.0
        else:  # No crossover, use normalized difference
            crossover_signal = current_diff / 100
            
        # 4. Extreme Levels
        extreme_signal = 0.0
        if abs(current_smi) > RegimeConfig.SMIConfig.EXTREME_THRESHOLD:
            # Add extra weight when SMI reaches extreme levels
            extreme_signal = (abs(current_smi) - RegimeConfig.SMIConfig.EXTREME_THRESHOLD) / (100 - RegimeConfig.SMIConfig.EXTREME_THRESHOLD)
            extreme_signal *= np.sign(current_smi)
        
        # Combine signals with weights
        composite_signal = (
            RegimeConfig.SMIConfig.WEIGHTS['base'] * base_signal +
            RegimeConfig.SMIConfig.WEIGHTS['momentum'] * smi_momentum +
            RegimeConfig.SMIConfig.WEIGHTS['crossover'] * crossover_signal +
            RegimeConfig.SMIConfig.WEIGHTS['extreme'] * extreme_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _calculate_smi(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate the Stochastic Momentum Index and its signal line.
        """
        # Calculate median price
        median_price = (high + low) / 2
        
        # First smoothing of price range
        highest = median_price.rolling(window=RegimeConfig.SMIConfig.PERIOD).max()
        lowest = median_price.rolling(window=RegimeConfig.SMIConfig.PERIOD).min()
        distance = close - (highest + lowest) / 2
        range_hl = highest - lowest
        
        # Double smoothing of range and distance
        smooth_distance = distance.ewm(span=RegimeConfig.SMIConfig.SMOOTH1).mean()
        double_smooth_distance = smooth_distance.ewm(span=RegimeConfig.SMIConfig.SMOOTH2).mean()
        
        smooth_range = range_hl.ewm(span=RegimeConfig.SMIConfig.SMOOTH1).mean()
        double_smooth_range = smooth_range.ewm(span=RegimeConfig.SMIConfig.SMOOTH2).mean()
        
        # Calculate SMI
        smi = 100 * (double_smooth_distance / (double_smooth_range / 2))
        
        # Calculate signal line
        signal_line = smi.ewm(span=RegimeConfig.SMIConfig.SIGNAL_PERIOD).mean()
        
        return {
            'smi': smi,
            'signal': signal_line
        }

class MfiIndicatorCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate Money Flow Index based signal.
        Returns a value between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Basic MFI signal (normalized)
        2. MFI momentum
        3. Extreme levels analysis
        4. Volume-price divergence
        """ 
        high = indicators['high']
        low = indicators['low']
        close = indicators['close']
        volume = indicators['volume']
        
        # Calculate typical price and money flow
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Get raw MFI values
        mfi_values = self._calculate_mfi(typical_price, money_flow)
        current_mfi = mfi_values.iloc[-1]
        prev_mfi = mfi_values.iloc[-2]
        
        # 1. Basic MFI Signal (-1 to 1)
        base_signal = (current_mfi - 50) / 50
        
        # 2. MFI Momentum
        mfi_momentum = (current_mfi - prev_mfi) / 100
        
        # 3. Extreme Levels Analysis
        extreme_signal = 0.0
        if current_mfi >= RegimeConfig.MFIConfig.OVERBOUGHT:
            extreme_signal = ((current_mfi - RegimeConfig.MFIConfig.OVERBOUGHT) / (100 - RegimeConfig.MFIConfig.OVERBOUGHT)) * -1  # Bearish when overbought
        elif current_mfi <= RegimeConfig.MFIConfig.OVERSOLD:
            extreme_signal = (RegimeConfig.MFIConfig.OVERSOLD - current_mfi) / RegimeConfig.MFIConfig.OVERSOLD  # Bullish when oversold
                            
        # 4. Volume-Price Divergence
        divergence_signal = self._calculate_divergence(close, mfi_values)
        
        # Combine signals with weights
        composite_signal = (
            RegimeConfig.MFIConfig.WEIGHTS['base'] * base_signal +
            RegimeConfig.MFIConfig.WEIGHTS['momentum'] * mfi_momentum +
            RegimeConfig.MFIConfig.WEIGHTS['extreme'] * extreme_signal +
            RegimeConfig.MFIConfig.WEIGHTS['divergence'] * divergence_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _calculate_mfi(self, typical_price: pd.Series, money_flow: pd.Series) -> pd.Series:
        """
        Calculate Money Flow Index values.
        """
        # Calculate positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = pd.Series(0.0, index=money_flow.index)
        negative_flow = pd.Series(0.0, index=money_flow.index)
        
        positive_flow[price_change > 0] = money_flow[price_change > 0]
        negative_flow[price_change < 0] = money_flow[price_change < 0]
        
        # Calculate money flow ratio
        period = RegimeConfig.MFIConfig.PERIOD
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def _calculate_divergence(self, close: pd.Series, mfi: pd.Series) -> float:
        """
        Calculate divergence between price and MFI.
        Returns a signal between -1 (bearish divergence) and 1 (bullish divergence).
        """
        # Get price and MFI changes over divergence period
        period = RegimeConfig.MFIConfig.DIVERGENCE_PERIOD
        price_change = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
        mfi_change = (mfi.iloc[-1] - mfi.iloc[-period]) / 100
        
        # Calculate normalized divergence
        divergence = mfi_change - price_change
        
        # Normalize to [-1, 1] range
        return max(min(divergence * RegimeConfig.MFIConfig.DIVERGENCE_FACTOR, 1.0), -1.0)

class WillrIndicatorCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate Williams %R based signal.
        Returns a value between -1 (strong bearish) and 1 (strong bullish).
        
        Signal components:
        1. Basic %R signal (normalized)
        2. Momentum and rate of change
        3. Extreme levels with confirmation
        4. Multi-timeframe analysis
        """
        high = indicators['high']
        low = indicators['low']
        close = indicators['close']
        
        # Calculate Williams %R for multiple timeframes
        willr_signals = {}
        for period in RegimeConfig.WILLRConfig.PERIODS:
            willr = self._calculate_willr(high, low, close, period)
            willr_signals[period] = willr
            
        # Get current values for primary timeframe
        primary_willr = willr_signals[RegimeConfig.WILLRConfig.PERIODS[0]]
        current_willr = primary_willr.iloc[-1]
        prev_willr = primary_willr.iloc[-2]
        
        # 1. Basic %R Signal (-1 to 1)
        # Convert from -100 to 0 scale to -1 to 1 scale
        base_signal = (current_willr + 50) / 50
        
        # 2. Momentum Component
        willr_momentum = (current_willr - prev_willr) / 100
        
        # 3. Extreme Levels Analysis
        extreme_signal = self._calculate_extreme_signal(current_willr)
        
        # 4. Multi-timeframe Confirmation
        mtf_signal = self._calculate_mtf_signal(willr_signals)
        
        # 5. Reversal Detection
        reversal_signal = self._detect_reversal(primary_willr)
        
        # Combine all signals
        composite_signal = (
            RegimeConfig.WILLRConfig.WEIGHTS['base'] * base_signal +
            RegimeConfig.WILLRConfig.WEIGHTS['momentum'] * willr_momentum +
            RegimeConfig.WILLRConfig.WEIGHTS['extreme'] * extreme_signal +
            RegimeConfig.WILLRConfig.WEIGHTS['mtf'] * mtf_signal +
            RegimeConfig.WILLRConfig.WEIGHTS['reversal'] * reversal_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _calculate_willr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """
        Calculate Williams %R for a given period.
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        willr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return willr
    
    def _calculate_extreme_signal(self, current_willr: float) -> float:
        """
        Calculate signal for extreme overbought/oversold conditions.
        """
        extreme_signal = 0.0
        
        if current_willr >= -RegimeConfig.WILLRConfig.OVERSOLD_THRESHOLD:  # Note: WillR is inverted
            # Oversold territory (closer to 0)
            extreme_signal = ((-RegimeConfig.WILLRConfig.OVERSOLD_THRESHOLD - current_willr) / 
                            RegimeConfig.WILLRConfig.OVERSOLD_THRESHOLD)
        elif current_willr <= -RegimeConfig.WILLRConfig.OVERBOUGHT_THRESHOLD:
            # Overbought territory (closer to -100)
            extreme_signal = ((current_willr + RegimeConfig.WILLRConfig.OVERBOUGHT_THRESHOLD) / 
                            (100 - RegimeConfig.WILLRConfig.OVERBOUGHT_THRESHOLD)) * -1
            
        return extreme_signal
    
    def _calculate_mtf_signal(self, willr_signals: Dict[int, pd.Series]) -> float:
        """
        Calculate multi-timeframe confirmation signal.
        """
        mtf_signals = []

        for period in RegimeConfig.WILLRConfig.PERIODS:
            current = willr_signals[period].iloc[-1]
            # Convert to -1 to 1 scale
            signal = (current + 50) / 50
            mtf_signals.append(signal)
            
        # Weight signals with more weight to shorter timeframes
        weighted_signal = sum(
            signal * weight for signal, weight in 
            zip(mtf_signals, RegimeConfig.WILLRConfig.MTF_WEIGHTS)
        )
        
        return weighted_signal
    
    def _detect_reversal(self, willr: pd.Series) -> float:
        """
        Detect potential reversals using Williams %R patterns.
        """
        lookback = RegimeConfig.WILLRConfig.REVERSAL_LOOKBACK
        
        # Get recent values
        recent_willr = willr.iloc[-lookback:]
        
        # Check for reversal patterns
        if (recent_willr.min() <= -RegimeConfig.WILLRConfig.OVERBOUGHT_THRESHOLD and 
            recent_willr.iloc[-1] > -RegimeConfig.WILLRConfig.OVERBOUGHT_THRESHOLD):
            # Potential bullish reversal from overbought
            return 1.0
        elif (recent_willr.max() >= -RegimeConfig.WILLRConfig.OVERSOLD_THRESHOLD and 
              recent_willr.iloc[-1] < -RegimeConfig.WILLRConfig.OVERSOLD_THRESHOLD):
            # Potential bearish reversal from oversold
            return -1.0
            
        return 0.0