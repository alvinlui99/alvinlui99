from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from .indicator_calculator import SignalCalculator
from config import RegimeConfig

class PatternSignalCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        indicator_calculators = {
            'candlestick': CandlestickPatternCalculator(),
            'harmonic': HarmonicPatternCalculator(),
            'chart': ChartPatternCalculator(),
            'volume': VolumePatternCalculator()
        }
        signals = {
            name: calculator.calculate_indicators(indicators)
            for name, calculator in indicator_calculators.items()
        }
        weighted_signal = sum(signals[name] * RegimeConfig.PatternSignalConfig.WEIGHTS[name] for name in signals)
        return max(min(weighted_signal, 1.0), -1.0)

class CandlestickPatternCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate candlestick pattern signals.
        Returns a value between -1 (bearish) and 1 (bullish).
        
        Signal components:
        1. Single candlestick patterns
        2. Double candlestick patterns
        3. Triple candlestick patterns
        4. Pattern confirmation
        """
        open_price = indicators['open']
        high = indicators['high']
        low = indicators['low']
        close = indicators['close']
        volume = indicators['volume']
        
        # 1. Single Candlestick Patterns (-1 to 1)
        single_signal = self._analyze_single_patterns(open_price, high, low, close)
        
        # 2. Double Candlestick Patterns (-1 to 1)
        double_signal = self._analyze_double_patterns(open_price, high, low, close)
        
        # 3. Triple Candlestick Patterns (-1 to 1)
        triple_signal = self._analyze_triple_patterns(open_price, high, low, close)
        
        # 4. Pattern Confirmation (-1 to 1)
        confirmation_signal = self._analyze_confirmation(open_price, high, low, close, volume)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.CandlestickConfig.WEIGHTS['single'] * single_signal +
            RegimeConfig.CandlestickConfig.WEIGHTS['double'] * double_signal +
            RegimeConfig.CandlestickConfig.WEIGHTS['triple'] * triple_signal +
            RegimeConfig.CandlestickConfig.WEIGHTS['confirmation'] * confirmation_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _analyze_single_patterns(self, open_price: pd.Series, high: pd.Series, 
                               low: pd.Series, close: pd.Series) -> float:
        """Analyze single candlestick patterns."""
        body_size = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        avg_body = body_size.rolling(window=RegimeConfig.CandlestickConfig.LOOKBACK).mean()
        
        # Detect hammer, shooting star, doji, etc.
        signals = []
        
        # Hammer/Hanging Man
        hammer_signal = self._detect_hammer(body_size, upper_shadow, lower_shadow, avg_body)
        signals.append(hammer_signal)
        
        # Shooting Star
        shooting_star_signal = self._detect_shooting_star(body_size, upper_shadow, lower_shadow, avg_body)
        signals.append(shooting_star_signal)
        
        # Doji
        doji_signal = self._detect_doji(body_size, avg_body)
        signals.append(doji_signal)
        
        return np.average(signals, weights=RegimeConfig.CandlestickConfig.SINGLE_WEIGHTS)
    
    def _analyze_double_patterns(self, open_price: pd.Series, high: pd.Series, 
                               low: pd.Series, close: pd.Series) -> float:
        """Analyze double candlestick patterns."""
        signals = []
        
        # Engulfing patterns
        engulfing_signal = self._detect_engulfing(open_price, close)
        signals.append(engulfing_signal)
        
        # Harami patterns
        harami_signal = self._detect_harami(open_price, close)
        signals.append(harami_signal)
        
        # Tweezer patterns
        tweezer_signal = self._detect_tweezer(high, low)
        signals.append(tweezer_signal)
        
        return np.average(signals, weights=RegimeConfig.CandlestickConfig.DOUBLE_WEIGHTS)
    
    def _detect_hammer(self, body_size: pd.Series, upper_shadow: pd.Series,
                    lower_shadow: pd.Series, avg_body: pd.Series) -> float:
        """Detect hammer and hanging man patterns."""
        # Current values
        curr_body = body_size.iloc[-1]
        curr_upper = upper_shadow.iloc[-1]
        curr_lower = lower_shadow.iloc[-1]
        curr_avg_body = avg_body.iloc[-1]
        
        # Hammer criteria
        is_small_upper = curr_upper <= curr_body * RegimeConfig.CandlestickConfig.HAMMER_UPPER_RATIO
        is_long_lower = curr_lower >= curr_body * RegimeConfig.CandlestickConfig.HAMMER_LOWER_RATIO
        is_normal_body = curr_body >= curr_avg_body * RegimeConfig.CandlestickConfig.MIN_BODY_RATIO
        
        if is_small_upper and is_long_lower and is_normal_body:
            # Determine if hammer (bullish) or hanging man (bearish)
            trend = self._get_trend(body_size)
            strength = 0.0 if curr_body == 0 else min(curr_lower / (curr_body * RegimeConfig.CandlestickConfig.HAMMER_LOWER_RATIO), 2.0)
            return strength if trend < 0 else -strength  # Hammer in downtrend, hanging man in uptrend
        
        return 0.0
    
    def _detect_shooting_star(self, body_size: pd.Series, upper_shadow: pd.Series, 
                            lower_shadow: pd.Series, avg_body: pd.Series) -> float:
        """Detect shooting star pattern."""
        curr_body = body_size.iloc[-1]
        curr_upper = upper_shadow.iloc[-1]
        curr_lower = lower_shadow.iloc[-1]
        curr_avg_body = avg_body.iloc[-1]
        
        is_long_upper = curr_upper >= curr_body * RegimeConfig.CandlestickConfig.STAR_UPPER_RATIO
        is_small_lower = curr_lower <= curr_body * RegimeConfig.CandlestickConfig.STAR_LOWER_RATIO
        is_small_body = curr_body <= curr_avg_body * RegimeConfig.CandlestickConfig.MAX_STAR_BODY_RATIO
        
        if is_long_upper and is_small_lower and is_small_body:
            trend = self._get_trend(body_size)
            strength = 0 if curr_body == 0 else min(curr_upper / (curr_body * RegimeConfig.CandlestickConfig.STAR_UPPER_RATIO), 2.0)
            return -strength if trend > 0 else 0  # Only bearish in uptrend
        
        return 0.0
    
    def _detect_doji(self, body_size: pd.Series, avg_body: pd.Series) -> float:
        """Detect doji patterns."""
        curr_body = body_size.iloc[-1]
        curr_avg_body = avg_body.iloc[-1]
        
        is_doji = curr_body <= curr_avg_body * RegimeConfig.CandlestickConfig.DOJI_RATIO
        
        if is_doji:
            trend = self._get_trend(body_size)
            strength = 1.0 if curr_avg_body == 0 else 1.0 - (curr_body / (curr_avg_body * RegimeConfig.CandlestickConfig.DOJI_RATIO))
            return strength if trend < 0 else -strength  # Bullish in downtrend, bearish in uptrend
        
        return 0.0

    def _detect_engulfing(self, open_price: pd.Series, close: pd.Series) -> float:
        """Detect bullish and bearish engulfing patterns."""
        prev_open = open_price.iloc[-2]
        prev_close = close.iloc[-2]
        curr_open = open_price.iloc[-1]
        curr_close = close.iloc[-1]
        
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        
        # Bullish engulfing
        if (prev_close < prev_open and  # Previous bearish
            curr_close > curr_open and  # Current bullish
            curr_open < prev_close and  # Opens below previous close
            curr_close > prev_open):    # Closes above previous open
            strength = min(curr_body / prev_body, 2.0)
            return strength
        
        # Bearish engulfing
        if (prev_close > prev_open and  # Previous bullish
            curr_close < curr_open and  # Current bearish
            curr_open > prev_close and  # Opens above previous close
            curr_close < prev_open):    # Closes below previous open
            strength = min(curr_body / prev_body, 2.0)
            return -strength
        
        return 0.0
    
    def _detect_harami(self, open_price: pd.Series, close: pd.Series) -> float:
        """Detect harami patterns."""
        prev_open = open_price.iloc[-2]
        prev_close = close.iloc[-2]
        curr_open = open_price.iloc[-1]
        curr_close = close.iloc[-1]
        
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        
        is_small_body = curr_body <= prev_body * RegimeConfig.CandlestickConfig.HARAMI_RATIO
        
        # Bullish harami
        if (prev_close < prev_open and  # Previous bearish
            curr_close > curr_open and  # Current bullish
            curr_open > prev_close and  # Opens inside previous body
            curr_close < prev_open and  # Closes inside previous body
            is_small_body):
            return 1.0
        
        # Bearish harami
        if (prev_close > prev_open and  # Previous bullish
            curr_close < curr_open and  # Current bearish
            curr_open < prev_close and  # Opens inside previous body
            curr_close > prev_open and  # Closes inside previous body
            is_small_body):
            return -1.0
        
        return 0.0
    
    def _detect_tweezer(self, high: pd.Series, low: pd.Series) -> float:
        """Detect tweezer tops and bottoms."""
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        curr_high = high.iloc[-1]
        curr_low = low.iloc[-1]
        
        high_diff = abs(curr_high - prev_high)
        low_diff = abs(curr_low - prev_low)
        avg_range = (high - low).rolling(window=RegimeConfig.CandlestickConfig.LOOKBACK).mean().iloc[-1]
        
        # Tweezer top
        if (high_diff <= avg_range * RegimeConfig.CandlestickConfig.TWEEZER_TOLERANCE and
            curr_high > curr_low):  # Current bar is bullish
            return -1.0
        
        # Tweezer bottom
        if (low_diff <= avg_range * RegimeConfig.CandlestickConfig.TWEEZER_TOLERANCE and
            curr_low < curr_high):  # Current bar is bearish
            return 1.0
        
        return 0.0
    
    def _get_trend(self, series: pd.Series) -> float:
        """Calculate short-term trend."""
        ma_short = series.rolling(window=RegimeConfig.CandlestickConfig.TREND_SHORT).mean()
        ma_long = series.rolling(window=RegimeConfig.CandlestickConfig.TREND_LONG).mean()
        return ma_short.iloc[-1] - ma_long.iloc[-1]

    def _analyze_confirmation(self, open_price: pd.Series, high: pd.Series, 
                            low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Analyze pattern confirmation using volume and price action."""
        # Volume confirmation
        vol_sma = volume.rolling(window=RegimeConfig.CandlestickConfig.VOL_PERIOD).mean()
        vol_signal = 0.0 if vol_sma.iloc[-1] == 0 else (volume.iloc[-1] - vol_sma.iloc[-1]) / vol_sma.iloc[-1]
        
        # Price range confirmation
        range_curr = high.iloc[-1] - low.iloc[-1]
        range_sma = (high - low).rolling(window=RegimeConfig.CandlestickConfig.RANGE_PERIOD).mean()
        range_signal = 0.0 if range_sma.iloc[-1] == 0 else (range_curr - range_sma.iloc[-1]) / range_sma.iloc[-1]
        
        # Combine signals
        return np.tanh((vol_signal + range_signal) * RegimeConfig.CandlestickConfig.CONFIRMATION_FACTOR)
    
    def _analyze_triple_patterns(self, open_price: pd.Series, high: pd.Series, 
                               low: pd.Series, close: pd.Series) -> float:
        """Analyze triple candlestick patterns."""
        signals = []
        
        # Morning/Evening Star
        star_signal = self._detect_star_pattern(open_price, high, low, close)
        signals.append(star_signal)
        
        # Three White Soldiers/Black Crows
        soldiers_signal = self._detect_three_soldiers_crows(open_price, high, low, close)
        signals.append(soldiers_signal)
        
        # Three Inside Up/Down
        inside_signal = self._detect_three_inside(open_price, high, low, close)
        signals.append(inside_signal)
        
        # Three Outside Up/Down
        outside_signal = self._detect_three_outside(open_price, high, low, close)
        signals.append(outside_signal)
        
        return np.average(signals, weights=RegimeConfig.CandlestickConfig.TRIPLE_WEIGHTS)
    
    def _detect_star_pattern(self, open_price: pd.Series, high: pd.Series, 
                           low: pd.Series, close: pd.Series) -> float:
        """Detect morning and evening star patterns."""
        # Get last three candles
        opens = open_price.iloc[-3:].values
        highs = high.iloc[-3:].values
        lows = low.iloc[-3:].values
        closes = close.iloc[-3:].values
        
        # Calculate body sizes
        bodies = abs(closes - opens)
        avg_body = bodies.mean()
        
        # Morning Star criteria
        morning_star = (
            closes[0] < opens[0] and  # First candle bearish
            bodies[1] < avg_body * RegimeConfig.CandlestickConfig.STAR_BODY_RATIO and  # Small middle body
            closes[2] > opens[2] and  # Third candle bullish
            closes[2] > (opens[0] + closes[0]) / 2  # Closes above first candle midpoint
        )
        
        # Evening Star criteria
        evening_star = (
            closes[0] > opens[0] and  # First candle bullish
            bodies[1] < avg_body * RegimeConfig.CandlestickConfig.STAR_BODY_RATIO and  # Small middle body
            closes[2] < opens[2] and  # Third candle bearish
            closes[2] < (opens[0] + closes[0]) / 2  # Closes below first candle midpoint
        )
        
        if morning_star:
            strength = min(bodies[2] / bodies[0], 2.0)
            return strength
        elif evening_star:
            strength = min(bodies[2] / bodies[0], 2.0)
            return -strength
        
        return 0.0
    
    def _detect_three_soldiers_crows(self, open_price: pd.Series, high: pd.Series, 
                                   low: pd.Series, close: pd.Series) -> float:
        """Detect three white soldiers and three black crows patterns."""
        # Get last three candles
        opens = open_price.iloc[-3:].values
        highs = high.iloc[-3:].values
        lows = low.iloc[-3:].values
        closes = close.iloc[-3:].values
        
        # Calculate body sizes and shadows
        bodies = abs(closes - opens)
        upper_shadows = highs - np.maximum(opens, closes)
        lower_shadows = np.minimum(opens, closes) - lows
        
        # Three White Soldiers criteria
        soldiers = (
            all(closes[i] > opens[i] for i in range(3)) and  # All bullish
            all(closes[i] > closes[i-1] for i in range(1, 3)) and  # Each close higher
            all(opens[i] > opens[i-1] for i in range(1, 3)) and  # Each open higher
            all(upper_shadows[i] <= bodies[i] * RegimeConfig.CandlestickConfig.SOLDIER_SHADOW_RATIO for i in range(3))  # Small upper shadows
        )
        
        # Three Black Crows criteria
        crows = (
            all(closes[i] < opens[i] for i in range(3)) and  # All bearish
            all(closes[i] < closes[i-1] for i in range(1, 3)) and  # Each close lower
            all(opens[i] < opens[i-1] for i in range(1, 3)) and  # Each open lower
            all(lower_shadows[i] <= bodies[i] * RegimeConfig.CandlestickConfig.CROW_SHADOW_RATIO for i in range(3))  # Small lower shadows
        )
        
        if soldiers:
            strength = min(sum(bodies) / (bodies[0] * 3), 2.0)
            return strength
        elif crows:
            strength = min(sum(bodies) / (bodies[0] * 3), 2.0)
            return -strength
        
        return 0.0
    
    def _detect_three_inside(self, open_price: pd.Series, high: pd.Series, 
                           low: pd.Series, close: pd.Series) -> float:
        """Detect three inside up and down patterns."""
        # Get last three candles
        opens = open_price.iloc[-3:].values
        closes = close.iloc[-3:].values
        bodies = abs(closes - opens)
        
        # Three Inside Up criteria
        inside_up = (
            closes[0] < opens[0] and  # First bearish
            closes[1] > opens[1] and  # Second bullish
            bodies[1] < bodies[0] * RegimeConfig.CandlestickConfig.INSIDE_RATIO and  # Second inside first
            closes[2] > opens[2] and  # Third bullish
            closes[2] > closes[1]  # Third confirms
        )
        
        # Three Inside Down criteria
        inside_down = (
            closes[0] > opens[0] and  # First bullish
            closes[1] < opens[1] and  # Second bearish
            bodies[1] < bodies[0] * RegimeConfig.CandlestickConfig.INSIDE_RATIO and  # Second inside first
            closes[2] < opens[2] and  # Third bearish
            closes[2] < closes[1]  # Third confirms
        )
        
        if inside_up:
            strength = min(bodies[2] / bodies[0], 2.0)
            return strength
        elif inside_down:
            strength = min(bodies[2] / bodies[0], 2.0)
            return -strength
        
        return 0.0
    
    def _detect_three_outside(self, open_price: pd.Series, high: pd.Series, 
                            low: pd.Series, close: pd.Series) -> float:
        """Detect three outside up and down patterns."""
        # Get last three candles
        opens = open_price.iloc[-3:].values
        closes = close.iloc[-3:].values
        bodies = abs(closes - opens)
        
        # Three Outside Up criteria
        outside_up = (
            closes[0] < opens[0] and  # First bearish
            closes[1] > opens[1] and  # Second bullish
            opens[1] < closes[0] and  # Second opens below first close
            closes[1] > opens[0] and  # Second closes above first open
            closes[2] > opens[2] and  # Third bullish
            closes[2] > closes[1]  # Third confirms
        )
        
        # Three Outside Down criteria
        outside_down = (
            closes[0] > opens[0] and  # First bullish
            closes[1] < opens[1] and  # Second bearish
            opens[1] > closes[0] and  # Second opens above first close
            closes[1] < opens[0] and  # Second closes below first open
            closes[2] < opens[2] and  # Third bearish
            closes[2] < closes[1]  # Third confirms
        )
        
        if outside_up:
            strength = min(bodies[2] / bodies[0], 2.0)
            return strength
        elif outside_down:
            strength = min(bodies[2] / bodies[0], 2.0)
            return -strength
        
        return 0.0

class HarmonicPatternCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate harmonic pattern signals.
        Returns a value between -1 (bearish) and 1 (bullish).
        
        Signal components:
        1. Gartley patterns
        2. Butterfly patterns
        3. Bat patterns
        4. Crab patterns
        """
        high = indicators['high']
        low = indicators['low']
        close = indicators['close']
        
        # Find potential swing points
        swing_highs, swing_lows = self._find_swing_points(close, high, low)
        
        # 1. Gartley Pattern Analysis (-1 to 1)
        gartley_signal = self._analyze_gartley(swing_highs, swing_lows)
        
        # 2. Butterfly Pattern Analysis (-1 to 1)
        butterfly_signal = self._analyze_butterfly(swing_highs, swing_lows)
        
        # 3. Bat Pattern Analysis (-1 to 1)
        bat_signal = self._analyze_bat(swing_highs, swing_lows)
        
        # 4. Crab Pattern Analysis (-1 to 1)
        crab_signal = self._analyze_crab(swing_highs, swing_lows)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.HarmonicConfig.WEIGHTS['gartley'] * gartley_signal +
            RegimeConfig.HarmonicConfig.WEIGHTS['butterfly'] * butterfly_signal +
            RegimeConfig.HarmonicConfig.WEIGHTS['bat'] * bat_signal +
            RegimeConfig.HarmonicConfig.WEIGHTS['crab'] * crab_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _find_swing_points(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Identify swing highs and lows using zigzag method."""
        window = RegimeConfig.HarmonicConfig.SWING_WINDOW
        threshold = RegimeConfig.HarmonicConfig.SWING_THRESHOLD
        
        # Initialize arrays for swing points
        swing_highs = pd.Series(index=close.index, dtype=float)
        swing_lows = pd.Series(index=close.index, dtype=float)
        
        for i in range(window, len(close) - window):
            # Check for swing high
            if all(high.iloc[i] > high.iloc[i-window:i]) and all(high.iloc[i] > high.iloc[i+1:i+window+1]):
                if high.iloc[i] > high.iloc[i-1] * (1 + threshold):
                    swing_highs.iloc[i] = high.iloc[i]
            
            # Check for swing low
            if all(low.iloc[i] < low.iloc[i-window:i]) and all(low.iloc[i] < low.iloc[i+1:i+window+1]):
                if low.iloc[i] < low.iloc[i-1] * (1 - threshold):
                    swing_lows.iloc[i] = low.iloc[i]
        
        return swing_highs.dropna(), swing_lows.dropna()
    
    def _calculate_retracement(self, point1: float, point2: float, point3: float) -> float:
        """Calculate retracement ratio between three points."""
        return abs((point3 - point2) / (point2 - point1))
    
    def _validate_pattern(self, ratios: Dict[str, float], target_ratios: Dict[str, float], 
                         tolerance: float) -> float:
        """Validate pattern ratios and return pattern strength."""
        matches = []
        for key in ratios:
            target = target_ratios[key]
            ratio = ratios[key]
            tolerance_range = tolerance * target
            
            if abs(ratio - target) <= tolerance_range:
                # Calculate how close the ratio is to target (1.0 = perfect match)
                match_quality = 1.0 - abs(ratio - target) / tolerance_range
                matches.append(match_quality)
            else:
                return 0.0  # Pattern invalid if any ratio is outside tolerance
        
        return np.mean(matches)  # Return average match quality
    
    def _analyze_gartley(self, swing_highs: pd.Series, swing_lows: pd.Series) -> float:
        """Analyze Gartley patterns (both bullish and bearish)."""
        if len(swing_highs) < 4 or len(swing_lows) < 4:
            return 0.0
        
        signals = []
        tolerance = RegimeConfig.HarmonicConfig.GARTLEY_TOLERANCE
        
        # Get recent swing points
        recent_highs = swing_highs.iloc[-4:].values
        recent_lows = swing_lows.iloc[-4:].values
        
        # Bullish Gartley
        for i in range(len(recent_lows) - 3):
            points = [recent_highs[i], recent_lows[i+1], recent_highs[i+2], recent_lows[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.GARTLEY_RATIOS, tolerance)
            if strength > 0:
                signals.append(strength)
        
        # Bearish Gartley
        for i in range(len(recent_highs) - 3):
            points = [recent_lows[i], recent_highs[i+1], recent_lows[i+2], recent_highs[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.GARTLEY_RATIOS, tolerance)
            if strength > 0:
                signals.append(-strength)
        
        return np.mean(signals) if signals else 0.0

    def _analyze_butterfly(self, swing_highs: pd.Series, swing_lows: pd.Series) -> float:
        """Analyze Butterfly patterns (both bullish and bearish)."""
        if len(swing_highs) < 4 or len(swing_lows) < 4:
            return 0.0
        
        signals = []
        tolerance = RegimeConfig.HarmonicConfig.BUTTERFLY_TOLERANCE
        
        # Get recent swing points
        recent_highs = swing_highs.iloc[-4:].values
        recent_lows = swing_lows.iloc[-4:].values
        
        # Bullish Butterfly
        for i in range(len(recent_lows) - 3):
            points = [recent_highs[i], recent_lows[i+1], recent_highs[i+2], recent_lows[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.BUTTERFLY_RATIOS, tolerance)
            if strength > 0:
                # Additional validation for Butterfly pattern
                if points[3] < points[1]:  # Point D should be below point B
                    signals.append(strength)
        
        # Bearish Butterfly
        for i in range(len(recent_highs) - 3):
            points = [recent_lows[i], recent_highs[i+1], recent_lows[i+2], recent_highs[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.BUTTERFLY_RATIOS, tolerance)
            if strength > 0:
                # Additional validation for Butterfly pattern
                if points[3] > points[1]:  # Point D should be above point B
                    signals.append(-strength)
        
        return np.mean(signals) if signals else 0.0
    
    def _analyze_bat(self, swing_highs: pd.Series, swing_lows: pd.Series) -> float:
        """Analyze Bat patterns (both bullish and bearish)."""
        if len(swing_highs) < 4 or len(swing_lows) < 4:
            return 0.0
        
        signals = []
        tolerance = RegimeConfig.HarmonicConfig.BAT_TOLERANCE
        
        # Get recent swing points
        recent_highs = swing_highs.iloc[-4:].values
        recent_lows = swing_lows.iloc[-4:].values
        
        # Bullish Bat
        for i in range(len(recent_lows) - 3):
            points = [recent_highs[i], recent_lows[i+1], recent_highs[i+2], recent_lows[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.BAT_RATIOS, tolerance)
            if strength > 0:
                # Additional validation for Bat pattern
                if points[2] < points[0]:  # Point C should be below point X
                    signals.append(strength)
        
        # Bearish Bat
        for i in range(len(recent_highs) - 3):
            points = [recent_lows[i], recent_highs[i+1], recent_lows[i+2], recent_highs[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.BAT_RATIOS, tolerance)
            if strength > 0:
                # Additional validation for Bat pattern
                if points[2] > points[0]:  # Point C should be above point X
                    signals.append(-strength)
        
        return np.mean(signals) if signals else 0.0
    
    def _analyze_crab(self, swing_highs: pd.Series, swing_lows: pd.Series) -> float:
        """Analyze Crab patterns (both bullish and bearish)."""
        if len(swing_highs) < 4 or len(swing_lows) < 4:
            return 0.0
        
        signals = []
        tolerance = RegimeConfig.HarmonicConfig.CRAB_TOLERANCE
        
        # Get recent swing points
        recent_highs = swing_highs.iloc[-4:].values
        recent_lows = swing_lows.iloc[-4:].values
        
        # Bullish Crab
        for i in range(len(recent_lows) - 3):
            points = [recent_highs[i], recent_lows[i+1], recent_highs[i+2], recent_lows[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.CRAB_RATIOS, tolerance)
            if strength > 0:
                # Additional validation for Crab pattern
                if points[3] < points[1] and points[2] < points[0]:  # Point D below B, C below X
                    signals.append(strength)
        
        # Bearish Crab
        for i in range(len(recent_highs) - 3):
            points = [recent_lows[i], recent_highs[i+1], recent_lows[i+2], recent_highs[i+3]]
            ratios = {
                'XA_BC': self._calculate_retracement(points[0], points[1], points[2]),
                'AB_CD': self._calculate_retracement(points[1], points[2], points[3]),
                'XA_AD': self._calculate_retracement(points[0], points[1], points[3])
            }
            
            strength = self._validate_pattern(ratios, RegimeConfig.HarmonicConfig.CRAB_RATIOS, tolerance)
            if strength > 0:
                # Additional validation for Crab pattern
                if points[3] > points[1] and points[2] > points[0]:  # Point D above B, C above X
                    signals.append(-strength)
        
        return np.mean(signals) if signals else 0.0

class ChartPatternCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate chart pattern signals.
        Returns a value between -1 (bearish) and 1 (bullish).
        
        Signal components:
        1. Support/Resistance
        2. Trend lines
        3. Chart formations
        4. Price channels
        """
        close = indicators['close']
        high = indicators['high']
        low = indicators['low']
        volume = indicators['volume']
        
        # 1. Support/Resistance Analysis (-1 to 1)
        sr_signal = self._analyze_support_resistance(close, high, low)
        
        # 2. Trendline Analysis (-1 to 1)
        trendline_signal = self._analyze_trendlines(close)
        
        # 3. Chart Formation Analysis (-1 to 1)
        formation_signal = self._analyze_formations(close, high, low, volume)
        
        # 4. Price Channel Analysis (-1 to 1)
        channel_signal = self._analyze_price_channels(high, low)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.ChartPatternConfig.WEIGHTS['sr'] * sr_signal +
            RegimeConfig.ChartPatternConfig.WEIGHTS['trendline'] * trendline_signal +
            RegimeConfig.ChartPatternConfig.WEIGHTS['formation'] * formation_signal +
            RegimeConfig.ChartPatternConfig.WEIGHTS['channel'] * channel_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)

    def _analyze_support_resistance(self, close: pd.Series, high: pd.Series, low: pd.Series) -> float:
        """Analyze support and resistance levels."""
        window = RegimeConfig.ChartPatternConfig.SR_WINDOW
        threshold = RegimeConfig.ChartPatternConfig.SR_THRESHOLD
        
        # Find potential support/resistance levels using price clusters
        levels = self._find_price_clusters(close, high, low, window)
        
        # Calculate current price position relative to levels
        curr_price = close.iloc[-1]
        distances = []
        strengths = []
        
        for level, strength in levels:
            distance = (curr_price - level) / level
            distances.append(distance)
            strengths.append(strength)
        
        if not distances:
            return 0.0
        
        # Weight distances by level strengths
        weighted_signal = 0.0
        total_strength = sum(strengths)
        
        if total_strength > 0:
            for distance, strength in zip(distances, strengths):
                # Closer levels have more influence
                proximity_factor = np.exp(-abs(distance) * RegimeConfig.ChartPatternConfig.PROXIMITY_FACTOR)
                weighted_signal += -np.sign(distance) * proximity_factor * (strength / total_strength)
        
        return np.tanh(weighted_signal * RegimeConfig.ChartPatternConfig.SR_FACTOR)

    def _find_price_clusters(self, close: pd.Series, high: pd.Series, low: pd.Series, 
                           window: int) -> List[Tuple[float, float]]:
        """Find price levels where multiple highs/lows cluster."""
        levels = []
        
        # Create price bins
        price_range = high.max() - low.min()
        bin_size = price_range * RegimeConfig.ChartPatternConfig.BIN_SIZE_FACTOR
        bins = np.arange(low.min(), high.max() + bin_size, bin_size)
        
        # Count touches for each price level
        for i in range(len(bins) - 1):
            level = (bins[i] + bins[i + 1]) / 2
            touches = sum((high > bins[i]) & (low < bins[i + 1]))
            
            if touches >= RegimeConfig.ChartPatternConfig.MIN_TOUCHES:
                # Calculate level strength based on number of touches and recency
                recency_weights = np.exp(-np.arange(len(close)) / RegimeConfig.ChartPatternConfig.RECENCY_DECAY)
                strength = touches * sum(recency_weights[(high > bins[i]) & (low < bins[i + 1])])
                levels.append((level, strength))
        
        return sorted(levels, key=lambda x: x[1], reverse=True)[:RegimeConfig.ChartPatternConfig.MAX_LEVELS]

    def _analyze_trendlines(self, close: pd.Series) -> float:
        """Analyze trendline breaks and tests."""
        window = RegimeConfig.ChartPatternConfig.TRENDLINE_WINDOW
        
        # Find potential trendlines using linear regression
        highs = pd.Series([max(close[max(0, i-window):i+1]) for i in range(len(close))])
        lows = pd.Series([min(close[max(0, i-window):i+1]) for i in range(len(close))])
        
        # Calculate slopes for potential trendlines
        high_slope = self._calculate_trendline_slope(highs)
        low_slope = self._calculate_trendline_slope(lows)
        
        # Analyze breaks and tests
        high_signal = self._analyze_trendline_break(close, highs, high_slope)
        low_signal = self._analyze_trendline_break(close, lows, low_slope)
        
        # Combine signals with more weight to broken trendlines
        return np.tanh((high_signal + low_signal) * RegimeConfig.ChartPatternConfig.TRENDLINE_FACTOR)

    def _calculate_trendline_slope(self, prices: pd.Series) -> float:
        """Calculate the slope of a potential trendline."""
        x = np.arange(len(prices))
        mask = ~np.isnan(prices)
        if sum(mask) < 2:
            return 0.0
        
        slope, _ = np.polyfit(x[mask], prices[mask], 1)
        return slope

    def _analyze_trendline_break(self, close: pd.Series, levels: pd.Series, slope: float) -> float:
        """Analyze if price has broken or is testing a trendline."""
        if abs(slope) < RegimeConfig.ChartPatternConfig.MIN_SLOPE:
            return 0.0
        
        # Project trendline to current bar
        curr_level = levels.iloc[-1] + slope
        curr_price = close.iloc[-1]
        
        # Calculate break/test signal
        distance = (curr_price - curr_level) / curr_level
        if abs(distance) < RegimeConfig.ChartPatternConfig.BREAK_THRESHOLD:
            return np.sign(slope) * RegimeConfig.ChartPatternConfig.TEST_STRENGTH
        else:
            return -np.sign(distance) * np.sign(slope)

    def _analyze_formations(self, close: pd.Series, high: pd.Series, low: pd.Series, 
                          volume: pd.Series) -> float:
        """Analyze chart formations (patterns)."""
        signals = []
        
        # Head and Shoulders pattern
        hs_signal = self._detect_head_shoulders(close, volume)
        signals.append(hs_signal * RegimeConfig.ChartPatternConfig.FORMATION_WEIGHTS['hs'])
        
        # Double Top/Bottom pattern
        double_signal = self._detect_double_pattern(high, low, volume)
        signals.append(double_signal * RegimeConfig.ChartPatternConfig.FORMATION_WEIGHTS['double'])
        
        # Triangle pattern
        triangle_signal = self._detect_triangle(high, low)
        signals.append(triangle_signal * RegimeConfig.ChartPatternConfig.FORMATION_WEIGHTS['triangle'])
        
        # Rectangle pattern
        rectangle_signal = self._detect_rectangle(high, low)
        signals.append(rectangle_signal * RegimeConfig.ChartPatternConfig.FORMATION_WEIGHTS['rectangle'])
        
        return np.sum(signals)

    def _detect_head_shoulders(self, close: pd.Series, volume: pd.Series) -> float:
        """
        Detect head and shoulders patterns (both regular and inverse).
        Returns: float between -1 (bearish H&S) and 1 (bullish IH&S)
        """
        window = RegimeConfig.ChartPatternConfig.HS_WINDOW
        
        # Find local peaks and troughs
        peaks = self._find_peaks(close, window)
        troughs = self._find_peaks(-close, window)
        
        if len(peaks) < 3 or len(troughs) < 2:
            return 0.0
            
        # Check for regular H&S (bearish)
        if self._validate_hs_pattern(peaks[-3:], troughs[-2:], False):
            # Calculate pattern strength based on volume confirmation
            vol_confirm = self._check_volume_confirmation(volume, peaks[-3:], False)
            return -1.0 * vol_confirm
            
        # Check for inverse H&S (bullish)
        if self._validate_hs_pattern(troughs[-3:], peaks[-2:], True):
            vol_confirm = self._check_volume_confirmation(volume, troughs[-3:], True)
            return 1.0 * vol_confirm
            
        return 0.0

    def _detect_double_pattern(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> float:
        """
        Detect double top and bottom patterns.
        Returns: float between -1 (double top) and 1 (double bottom)
        """
        window = RegimeConfig.ChartPatternConfig.DOUBLE_WINDOW
        tolerance = RegimeConfig.ChartPatternConfig.DOUBLE_TOLERANCE
        
        # Find potential double tops
        highs = self._find_peaks(high, window)
        if len(highs) >= 2:
            last_two_highs = highs[-2:]
            price_diff = abs(high.iloc[last_two_highs[0]] - high.iloc[last_two_highs[1]]) / high.iloc[last_two_highs[0]]
            
            if price_diff < tolerance:
                vol_confirm = self._check_volume_confirmation(volume, last_two_highs, False)
                return -1.0 * vol_confirm
        
        # Find potential double bottoms
        lows = self._find_peaks(-low, window)
        if len(lows) >= 2:
            last_two_lows = lows[-2:]
            price_diff = abs(low.iloc[last_two_lows[0]] - low.iloc[last_two_lows[1]]) / low.iloc[last_two_lows[0]]
            
            if price_diff < tolerance:
                vol_confirm = self._check_volume_confirmation(volume, last_two_lows, True)
                return 1.0 * vol_confirm
        
        return 0.0

    def _detect_triangle(self, high: pd.Series, low: pd.Series) -> float:
        """
        Detect triangle patterns (ascending, descending, symmetric).
        Returns: float between -1 (bearish) and 1 (bullish)
        """
        window = RegimeConfig.ChartPatternConfig.TRIANGLE_WINDOW
        min_points = RegimeConfig.ChartPatternConfig.TRIANGLE_MIN_POINTS
        
        # Get recent highs and lows
        highs = self._find_peaks(high, window)
        lows = self._find_peaks(-low, window)
        
        if len(highs) < min_points or len(lows) < min_points:
            return 0.0
        
        # Calculate slopes of upper and lower trend lines
        high_slope = self._calculate_trendline_slope(high.iloc[highs[-min_points:]])
        low_slope = self._calculate_trendline_slope(low.iloc[lows[-min_points:]])
        
        # Identify triangle type
        if abs(high_slope) < RegimeConfig.ChartPatternConfig.TRIANGLE_SLOPE_TOLERANCE:
            if low_slope > 0:  # Ascending triangle
                return 1.0
        elif abs(low_slope) < RegimeConfig.ChartPatternConfig.TRIANGLE_SLOPE_TOLERANCE:
            if high_slope < 0:  # Descending triangle
                return -1.0
        elif high_slope < 0 and low_slope > 0:  # Symmetric triangle
            return np.sign(high_slope + low_slope) * 0.5
        
        return 0.0

    def _detect_rectangle(self, high: pd.Series, low: pd.Series) -> float:
        """
        Detect rectangle patterns.
        Returns: float between -1 (bearish) and 1 (bullish)
        """
        window = RegimeConfig.ChartPatternConfig.RECTANGLE_WINDOW
        tolerance = RegimeConfig.ChartPatternConfig.RECTANGLE_TOLERANCE
        
        # Find potential resistance and support levels
        highs = high.rolling(window=window).max()
        lows = low.rolling(window=window).min()
        
        # Calculate average ranges
        high_range = abs(highs - highs.mean()) / highs.mean()
        low_range = abs(lows - lows.mean()) / lows.mean()
        
        # Check if price is moving in a channel
        if (high_range.iloc[-1] < tolerance and low_range.iloc[-1] < tolerance):
            # Determine breakout direction
            if high.iloc[-1] > highs.iloc[-2]:  # Bullish breakout
                return 1.0
            elif low.iloc[-1] < lows.iloc[-2]:  # Bearish breakout
                return -1.0
            
        return 0.0

    def _find_peaks(self, series: pd.Series, window: int) -> List[int]:
        """Find local peaks in a series using rolling window."""
        peaks = []
        for i in range(window, len(series) - window):
            if all(series.iloc[i] > series.iloc[i-window:i]) and \
               all(series.iloc[i] > series.iloc[i+1:i+window+1]):
                peaks.append(i)
        return peaks

    def _validate_hs_pattern(self, peaks: List[int], troughs: List[int], inverse: bool) -> bool:
        """Validate head and shoulders pattern formation."""
        if len(peaks) != 3 or len(troughs) != 2:
            return False
            
        # Check pattern geometry
        head = peaks[1]
        left_shoulder = peaks[0]
        right_shoulder = peaks[2]
        left_trough = troughs[0]
        right_trough = troughs[1]
        
        # Validate sequence
        if not (left_shoulder < head and right_shoulder < head):
            return False
            
        # Check symmetry
        shoulder_diff = abs(peaks[2] - peaks[0])
        if shoulder_diff > RegimeConfig.ChartPatternConfig.HS_SYMMETRY_TOLERANCE:
            return False
            
        # Check neckline
        neckline_slope = (troughs[1] - troughs[0]) / (right_trough - left_trough)
        if abs(neckline_slope) > RegimeConfig.ChartPatternConfig.HS_NECKLINE_TOLERANCE:
            return False
            
        return True

    def _check_volume_confirmation(self, volume: pd.Series, points: List[int], bullish: bool) -> float:
        """
        Check volume confirmation for pattern.
        Returns strength multiplier between 0 and 1.
        """
        # Calculate average volume during pattern formation
        pattern_vol = volume.iloc[points[0]:points[-1]].mean()
        
        # Compare with previous volume
        prev_vol = volume.iloc[points[0]-len(points):points[0]].mean()
        
        # Calculate volume ratio
        vol_ratio = pattern_vol / prev_vol if prev_vol > 0 else 1.0
        
        # Adjust ratio based on pattern direction
        if bullish:
            return min(vol_ratio, 2.0) / 2.0
        else:
            return min(2.0 / vol_ratio if vol_ratio > 0 else 2.0, 2.0) / 2.0

    def _analyze_price_channels(self, high: pd.Series, low: pd.Series) -> float:
        """Analyze price channels and breakouts."""
        window = RegimeConfig.ChartPatternConfig.CHANNEL_WINDOW
        
        # Calculate upper and lower channel boundaries
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()
        middle = (upper + lower) / 2
        
        # Calculate current position within channel
        curr_price = high.iloc[-1]
        channel_height = upper.iloc[-1] - lower.iloc[-1]
        
        if channel_height <= 0:
            return 0.0
        
        # Calculate relative position (-1 to 1)
        relative_pos = 2 * (curr_price - middle.iloc[-1]) / channel_height
        
        # Check for channel breakouts
        if curr_price > upper.iloc[-2]:
            return 1.0
        elif curr_price < lower.iloc[-2]:
            return -1.0
        
        return np.tanh(relative_pos * RegimeConfig.ChartPatternConfig.CHANNEL_FACTOR)

class VolumePatternCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate volume pattern signals.
        Returns a value between -1 (bearish) and 1 (bullish).
        
        Signal components:
        1. Volume trend analysis
        2. Price-volume relationship
        3. Volume breakouts
        4. Accumulation/Distribution
        """
        close = indicators['close']
        high = indicators['high']
        low = indicators['low']
        volume = indicators['volume']
        
        # 1. Volume Trend Analysis (-1 to 1)
        trend_signal = self._analyze_volume_trend(volume, close)
        
        # 2. Price-Volume Analysis (-1 to 1)
        pv_signal = self._analyze_price_volume(close, volume)
        
        # 3. Volume Breakout Analysis (-1 to 1)
        breakout_signal = self._analyze_volume_breakouts(close, volume)
        
        # 4. Accumulation/Distribution Analysis (-1 to 1)
        accum_signal = self._analyze_accumulation_distribution(close, high, low, volume)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.VolumePatternConfig.WEIGHTS['trend'] * trend_signal +
            RegimeConfig.VolumePatternConfig.WEIGHTS['pv'] * pv_signal +
            RegimeConfig.VolumePatternConfig.WEIGHTS['breakout'] * breakout_signal +
            RegimeConfig.VolumePatternConfig.WEIGHTS['accumulation'] * accum_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)

    def _analyze_volume_trend(self, volume: pd.Series, close: pd.Series) -> float:
        """Analyze volume trend and divergences."""
        # Calculate volume moving averages
        vol_sma_short = volume.rolling(window=RegimeConfig.VolumePatternConfig.VOL_SHORT_PERIOD).mean()
        vol_sma_long = volume.rolling(window=RegimeConfig.VolumePatternConfig.VOL_LONG_PERIOD).mean()
        
        # Calculate price moving averages
        price_sma_short = close.rolling(window=RegimeConfig.VolumePatternConfig.PRICE_SHORT_PERIOD).mean()
        price_sma_long = close.rolling(window=RegimeConfig.VolumePatternConfig.PRICE_LONG_PERIOD).mean()
        
        # Volume trend
        vol_trend = 0.0 if vol_sma_long.iloc[-1] == 0 else (vol_sma_short.iloc[-1] / vol_sma_long.iloc[-1]) - 1
        
        # Price trend
        price_trend = 0.0 if price_sma_long.iloc[-1] == 0 else (price_sma_short.iloc[-1] / price_sma_long.iloc[-1]) - 1
        
        # Check for divergences
        if abs(price_trend) > RegimeConfig.VolumePatternConfig.TREND_THRESHOLD:
            # Bullish divergence: Price down, volume up
            if price_trend < 0 and vol_trend > 0:
                return vol_trend * RegimeConfig.VolumePatternConfig.DIVERGENCE_FACTOR
            # Bearish divergence: Price up, volume down
            elif price_trend > 0 and vol_trend < 0:
                return -vol_trend * RegimeConfig.VolumePatternConfig.DIVERGENCE_FACTOR
        
        # No divergence, return volume trend signal
        return np.tanh(vol_trend * RegimeConfig.VolumePatternConfig.TREND_FACTOR)

    def _analyze_price_volume(self, close: pd.Series, volume: pd.Series) -> float:
        """Analyze price-volume relationships."""
        # Calculate price and volume changes
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        
        # Calculate moving correlations
        correlation = self._calculate_rolling_correlation(
            price_change, 
            volume_change, 
            RegimeConfig.VolumePatternConfig.CORRELATION_PERIOD
        )
        
        # Calculate volume force (price change * volume change)
        force = price_change * volume_change
        force_sma = force.rolling(window=RegimeConfig.VolumePatternConfig.FORCE_PERIOD).mean()
        
        # Combine correlation and force signals
        correlation_signal = -correlation.iloc[-1]  # Negative correlation is typically bullish
        force_signal = np.sign(force_sma.iloc[-1])
        
        return np.tanh((correlation_signal + force_signal) * RegimeConfig.VolumePatternConfig.PV_FACTOR)

    def _analyze_volume_breakouts(self, close: pd.Series, volume: pd.Series) -> float:
        """Analyze volume breakouts and climax conditions."""
        # Calculate volume statistics
        vol_mean = volume.rolling(window=RegimeConfig.VolumePatternConfig.BREAKOUT_PERIOD).mean()
        vol_std = volume.rolling(window=RegimeConfig.VolumePatternConfig.BREAKOUT_PERIOD).std()
        
        # Calculate z-score of current volume
        curr_zscore = 0.0 if vol_std.iloc[-1] == 0 else (volume.iloc[-1] - vol_mean.iloc[-1]) / vol_std.iloc[-1]
        
        # Calculate price momentum
        momentum = close.pct_change(RegimeConfig.VolumePatternConfig.MOMENTUM_PERIOD)
        
        # Identify climax conditions
        if abs(curr_zscore) > RegimeConfig.VolumePatternConfig.CLIMAX_THRESHOLD:
            # Buying climax: High volume + positive momentum
            if momentum.iloc[-1] > RegimeConfig.VolumePatternConfig.MOMENTUM_THRESHOLD:
                return -1.0  # Potentially bearish
            # Selling climax: High volume + negative momentum
            elif momentum.iloc[-1] < -RegimeConfig.VolumePatternConfig.MOMENTUM_THRESHOLD:
                return 1.0  # Potentially bullish
        
        # Normal breakout conditions
        return np.tanh(curr_zscore * np.sign(momentum.iloc[-1]) * 
                      RegimeConfig.VolumePatternConfig.BREAKOUT_FACTOR)

    def _analyze_accumulation_distribution(self, close: pd.Series, high: pd.Series, 
                                        low: pd.Series, volume: pd.Series) -> float:
        """Analyze accumulation/distribution patterns."""
        # Calculate Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Calculate Accumulation/Distribution Line
        ad_line = mf_volume.cumsum()
        
        # Calculate ADL moving averages
        adl_short = ad_line.rolling(window=RegimeConfig.VolumePatternConfig.ADL_SHORT_PERIOD).mean()
        adl_long = ad_line.rolling(window=RegimeConfig.VolumePatternConfig.ADL_LONG_PERIOD).mean()
        
        # Calculate ADL momentum
        adl_momentum = (adl_short.iloc[-1] / adl_long.iloc[-1]) - 1
        
        # Check for divergences with price
        price_momentum = (close.iloc[-1] / close.iloc[-RegimeConfig.VolumePatternConfig.ADL_LONG_PERIOD]) - 1
        
        if abs(price_momentum) > RegimeConfig.VolumePatternConfig.ADL_THRESHOLD:
            # Bullish divergence: Price down, ADL up
            if price_momentum < 0 and adl_momentum > 0:
                return adl_momentum * RegimeConfig.VolumePatternConfig.ADL_FACTOR
            # Bearish divergence: Price up, ADL down
            elif price_momentum > 0 and adl_momentum < 0:
                return adl_momentum * RegimeConfig.VolumePatternConfig.ADL_FACTOR
        
        return np.tanh(adl_momentum * RegimeConfig.VolumePatternConfig.ADL_MOMENTUM_FACTOR)

    def _calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, 
                                    window: int) -> pd.Series:
        """Calculate rolling correlation between two series."""
        return series1.rolling(window=window).corr(series2)