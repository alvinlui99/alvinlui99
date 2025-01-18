import numpy as np
import pandas as pd
from typing import Dict, Union
import talib
from config import RegimeConfig

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
        self.current_regime = 0
        self.leverage_components = {
            'regime': 0,
            'trend': 0,
            'volatility': 0,
            'confidence': 0,
            'momentum': 0,
            'drawdown': 0
        }

        self.regime_adj_factor = RegimeConfig.REGIME_ADJ_FACTOR
        self.trend_adj_factor = RegimeConfig.TREND_ADJ_FACTOR
        self.vol_adj_factor = RegimeConfig.VOL_ADJ_FACTOR
        self.momentum_adj_factor = RegimeConfig.MOMENTUM_ADJ_FACTOR

    def get_regime_leverage(self, data: Dict[str, pd.DataFrame], weights: Dict[str, float],
                            equity_curve: pd.Series) -> float:
        """
        Convert trend regime detection into a leverage multiplier with enhanced risk management.
        """
        regime_data = self.detect_trend_regime(data, weights)
        
        # Extract key metrics
        regime = regime_data['regime']
        confidence = regime_data['confidence']
        trend_score = regime_data['trend_score']
        volatility = regime_data['volatility']
        
        # Much more aggressive base leverage calculation for positive regimes
        if regime > 0:
            # Start from 2.0 base and scale up to 3.0 based on regime strength
            regime_factor = 2.0 + (self.regime_adj_factor * 2.5 * (regime + 1))  
            trend_factor = 1.0 + (self.trend_adj_factor * 2.0 * (trend_score + 1))
        else:
            # Keep conservative during negative regimes, base around 1.0
            regime_factor = 1.0 + (self.regime_adj_factor * (regime + 1))
            trend_factor = 1.0 + (self.trend_adj_factor * (trend_score + 1))
        
        base_leverage = regime_factor * trend_factor
        
        # Adjust volatility impact based on regime
        vol_percentile = self._calculate_vol_percentile(volatility)
        if regime > 0:
            vol_factor = 1.0 - (vol_percentile * self.vol_adj_factor * 0.5)  # Even less vol impact in positive regimes
        else:
            vol_factor = 1.0 - (vol_percentile * self.vol_adj_factor * 1.5)  # More vol impact in negative regimes
        
        # Enhanced drawdown protection with exponential reduction
        recent_returns = self._calculate_recent_returns(equity_curve)
        drawdown_factor = self._calculate_drawdown_factor(recent_returns)
        
        # More aggressive momentum factor during uptrends
        momentum = self._calculate_momentum(equity_curve)
        momentum_factor = self._calculate_momentum_factor(momentum, vol_percentile, confidence)
        
        self.leverage_components = {
            'regime': regime_factor,
            'trend': trend_factor,
            'volatility': vol_factor,
            'confidence': confidence,
            'momentum': momentum_factor,
            'drawdown': drawdown_factor
        }

        # Combine all factors with progressive scaling
        adjusted_leverage = (
            base_leverage * 
            vol_factor * 
            min(1.0, 1.5 * confidence) * 
            drawdown_factor * 
            momentum_factor
        )

        # Ensure leverage stays within bounds with smoother transitions
        # Force minimum leverage of 1.0 during positive regimes
        if regime > 0:
            return max(1.0, min(3.0, adjusted_leverage))
        else:
            return max(1.0, min(1.5, adjusted_leverage))
    
    def detect_trend_regime(self, data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, Union[int, float, dict]]:
        """
        Enhanced trend regime detection using multiple indicators and volatility.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Price data for multiple assets
        weights : Dict[str, float]
            Asset weights for portfolio
            
        Returns:
        --------
        dict : Contains regime classification, confidence metrics, and detailed signals
        """
        # Get weighted price data
        price_data = self.prepare_data(data, weights)
        indicators = self.calculate_indicators(price_data)
        
        # Calculate additional volatility metrics
        close_prices = price_data['close']
        volatility = close_prices.pct_change().rolling(window=24).std() * np.sqrt(365 * 24)  # Annualized volatility
        atr = talib.ATR(price_data['high'], price_data['low'], close_prices, timeperiod=14)
        
        # Initialize enhanced signals dictionary
        signals = {
            'primary_trends': {},
            'momentum_signals': {},
            'volatility_signals': {},
            'pattern_signals': {}
        }
        
        # 1. Primary Trend Signals
        ma_slope_val = talib.ROC(pd.Series(indicators['ma_long']), timeperiod=5).iloc[-1]
        ma_slope = 0 if pd.isna(ma_slope_val) else np.sign(ma_slope_val)
        signals['primary_trends'] = {
            'ma_crossover': 1 if indicators['ma_short'] > indicators['ma_long'] else -1,
            'price_vs_ma': 1 if close_prices.iloc[-1] > indicators['ma_long'] else -1,
            'ma_slope': ma_slope
        }
        
        # 2. Momentum Signals
        signals['momentum_signals'] = {
            'macd': 1 if indicators['macd'] > indicators['macd_signal'] else -1,
            'rsi': self._classify_rsi(indicators['rsi']),
            'adx_strength': self._classify_adx(indicators['adx'])
        }
        
        # 3. Volatility Signals
        signals['volatility_signals'] = {
            'volatility_regime': self._classify_volatility(volatility.iloc[-1]),
            'atr_trend': self._classify_atr_trend(atr),
            'bollinger_position': self._calculate_bollinger_position(close_prices)
        }
        
        # 4. Pattern Recognition
        signals['pattern_signals'] = {
            'trend_consistency': self._check_trend_consistency(close_prices),
            'support_resistance': self._check_support_resistance(price_data)
        }
        
        # Calculate weighted trend score with dynamic weights
        trend_score = self._calculate_weighted_score(signals)
        
        # Determine confidence level
        confidence = self._calculate_confidence(signals)
        
        # Set regime with hysteresis to prevent frequent changes
        new_regime = self._determine_regime_with_hysteresis(trend_score, confidence)
        self.current_regime = new_regime
        
        return {
            'regime': new_regime,
            'trend_score': trend_score,
            'confidence': confidence,
            'signals': signals,
            'volatility': volatility.iloc[-1]
        }
    
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

    def _classify_rsi(self, rsi: float) -> int:
        """Classify RSI with more nuanced zones"""
        if rsi > 70: return 2  # Strongly overbought
        if rsi > 60: return 1  # Mildly overbought
        if rsi < 30: return -2 # Strongly oversold
        if rsi < 40: return -1 # Mildly oversold
        return 0

    def _classify_adx(self, adx: float) -> Dict[str, Union[int, float]]:
        """Classify trend strength using ADX"""
        strength = 0
        if adx > 40: strength = 2    # Very strong trend
        elif adx > 25: strength = 1  # Strong trend
        elif adx < 15: strength = -1 # Weak trend
        return strength

    def _calculate_weighted_score(self, signals: Dict) -> float:
        """Calculate weighted trend score with dynamic weights"""
        # Adjust weights to be more sensitive to primary trends
        weights = {
            'primary_trends': 0.4,
            'momentum_signals': 0.3,
            'volatility_signals': 0.2,
            'pattern_signals': 0.1
        }
        
        # Add exponential smoothing for more responsive scoring
        scores = {
            'primary_trends': np.mean([v for v in signals['primary_trends'].values()]) * 1.2,  # Amplify primary trends
            'momentum_signals': np.mean([v if isinstance(v, (int, float)) 
                                    else v['strength'] for v in signals['momentum_signals'].values()]),
            'volatility_signals': np.mean([v for v in signals['volatility_signals'].values()]),
            'pattern_signals': np.mean([v for v in signals['pattern_signals'].values()])
        }

        return sum(scores[k] * weights[k] for k in weights)

    def _determine_regime_with_hysteresis(self, trend_score: float, confidence: float) -> int:
        """Determine regime with hysteresis to prevent frequent changes"""
        # Lower thresholds for faster regime changes
        if self.current_regime == 1:
            threshold = -0.4
        elif self.current_regime == -1:
            threshold = 0.4
        else:
            threshold = 0.3
        
        # Lower confidence requirement for strong signals
        if abs(trend_score) > 0.5:
            confidence_threshold = 0.35  # Lower confidence needed for strong trends
        else:
            confidence_threshold = 0.4
        
        if trend_score > threshold and confidence > confidence_threshold:
            return 1
        elif trend_score < -threshold and confidence > confidence_threshold:
            return -1
        return 0
    
    def _classify_volatility(self, volatility: float) -> int:
        """
        Classify volatility regime using dynamic thresholds
        
        Returns:
        --------
        int : 1 for high volatility, 0 for normal, -1 for low volatility
        """
        # Calculate rolling volatility percentiles if not already stored
        if not hasattr(self, 'vol_percentiles'):
            self.vol_percentiles = {
                'low': volatility * 0.5,    # Below 50% of typical volatility
                'high': volatility * 2.0    # Above 200% of typical volatility
            }
        
        if volatility > self.vol_percentiles['high']:
            return 1        # High volatility regime
        elif volatility < self.vol_percentiles['low']:
            return -1      # Low volatility regime
        return 0           # Normal volatility regime

    def _classify_atr_trend(self, atr: pd.Series) -> int:
        """
        Classify trend based on ATR movement
        
        Returns:
        --------
        int : 1 for expanding ranges, -1 for contracting, 0 for neutral
        """
        # Calculate ATR momentum over different periods
        atr_short = atr.rolling(window=5).mean()
        atr_long = atr.rolling(window=20).mean()
        
        # Get latest values
        current_short = atr_short.iloc[-1]
        current_long = atr_long.iloc[-1]
        
        # Calculate ATR trend
        if current_short > current_long * 1.1:
            return 1        # Expanding ranges (trending)
        elif current_short < current_long * 0.9:
            return -1      # Contracting ranges (consolidation)
        return 0           # Neutral

    def _calculate_bollinger_position(self, prices: pd.Series) -> int:
        """
        Calculate position relative to Bollinger Bands
        
        Returns:
        --------
        int : 2 (above upper), 1 (upper half), 0 (middle), 
            -1 (lower half), -2 (below lower)
        """
        # Calculate Bollinger Bands
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]
        
        # Determine position relative to bands
        if current_price > current_upper:
            return 2
        elif current_price > current_sma:
            return 1
        elif current_price < current_lower:
            return -2
        elif current_price < current_sma:
            return -1
        return 0

    def _check_trend_consistency(self, prices: pd.Series) -> int:
        """
        Check for trend consistency using multiple timeframes
        
        Returns:
        --------
        int : 1 for consistent uptrend, -1 for consistent downtrend, 
            0 for mixed signals
        """
        # Calculate SMAs for multiple timeframes
        sma_periods = [10, 20, 50]
        smas = [prices.rolling(window=period).mean().iloc[-1] 
                for period in sma_periods]
        current_price = prices.iloc[-1]
        
        # Check if price is above/below all SMAs
        above_all = all(current_price > sma for sma in smas)
        below_all = all(current_price < sma for sma in smas)
        
        # Check if SMAs are properly aligned for trend
        smas_aligned_up = all(smas[i] > smas[i+1] 
                            for i in range(len(smas)-1))
        smas_aligned_down = all(smas[i] < smas[i+1] 
                            for i in range(len(smas)-1))
        
        if above_all and smas_aligned_up:
            return 1        # Strong consistent uptrend
        elif below_all and smas_aligned_down:
            return -1      # Strong consistent downtrend
        return 0           # Mixed signals

    def _check_support_resistance(self, data: pd.DataFrame) -> int:
        """
        Analyze price action relative to support/resistance levels
        
        Returns:
        --------
        int : 1 for breakout, -1 for rejection, 0 for neutral
        """
        # Get recent price data
        high = data['high'].iloc[-20:]
        low = data['low'].iloc[-20:]
        close = data['close'].iloc[-20:]
        
        # Identify potential support/resistance levels
        resistance = high.rolling(10).max().iloc[-1]
        support = low.rolling(10).min().iloc[-1]
        
        current_close = close.iloc[-1]
        previous_close = close.iloc[-2]
        
        # Calculate price position relative to levels
        resistance_distance = (resistance - current_close) / resistance
        support_distance = (current_close - support) / current_close
        
        # Check for breakouts or rejections
        if current_close > resistance and previous_close < resistance:
            return 1    # Breakout above resistance
        elif current_close < support and previous_close > support:
            return -1   # Breakdown below support
        elif abs(resistance_distance) < 0.01 and current_close < previous_close:
            return -1   # Rejection at resistance
        elif abs(support_distance) < 0.01 and current_close > previous_close:
            return 1    # Bounce from support
        return 0        # Neutral

    def _calculate_confidence(self, signals: Dict) -> float:
        """
        Calculate confidence score based on signal agreement
        
        Returns:
        --------
        float : Confidence score between 0 and 1
        """
        # Count agreeing signals
        signal_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        total_signals = 0
        
        # Process each signal category
        for category in signals.values():
            for signal in category.values():
                total_signals += 1
                if isinstance(signal, dict):
                    value = signal.get('strength', 0)
                else:
                    value = signal
                    
                if value > 0:
                    signal_counts['bullish'] += 1
                elif value < 0:
                    signal_counts['bearish'] += 1
                else:
                    signal_counts['neutral'] += 1
        
        # Calculate agreement percentage
        max_agreement = max(signal_counts.values())
        confidence = max_agreement / total_signals if total_signals > 0 else 0
        
        # Adjust confidence based on volatility signals
        vol_signals = signals['volatility_signals']
        if abs(vol_signals.get('volatility_regime', 0)) > 0:
            confidence *= 0.8  # Reduce confidence in high volatility
        
        return min(confidence, 1.0)  # Cap at 1.0

    def _calculate_drawdown_factor(self, returns: float) -> float:
        """Calculate drawdown factor with asymmetric response"""
        if returns >= 0:
            return 1.0 + (returns * 1.0)  # Double the positive returns boost
        
        # Keep aggressive reduction for negative returns
        return np.exp(4 * returns)  # Even more aggressive reduction during drawdowns

    def _calculate_momentum_factor(self, momentum: float, vol_percentile: float, confidence: float) -> float:
        """Calculate momentum factor with more aggressive upside"""
        momentum_factor = 1.0
        
        # More aggressive conditions for positive momentum
        if momentum > 0.5:  # Even lower threshold
            if vol_percentile < 0.4 and confidence > 0.6:  # More relaxed conditions
                # More aggressive boost calculation
                boost = min(
                    2.0,  # Maximum 100% boost
                    1.0 + (
                        (momentum - 0.5) * 3.0 +     # Much stronger momentum contribution
                        (0.4 - vol_percentile) * 2.0 + # Stronger volatility contribution
                        (confidence - 0.6) * 2.0     # Stronger confidence contribution
                    )
                )
                momentum_factor = boost
        
        return momentum_factor

    def _calculate_vol_percentile(self, current_vol: float) -> float:
        """Calculate volatility percentile relative to historical range"""
        if not hasattr(self, 'vol_percentiles'):
            self.vol_percentiles = {
                'low': current_vol * 0.5,
                'high': current_vol * 2.0
            }
        
        # Return value between 0 and 1 representing current vol level
        return min(1.0, max(0.0, 
            (current_vol - self.vol_percentiles['low']) / 
            (self.vol_percentiles['high'] - self.vol_percentiles['low'])
        ))

    def _calculate_recent_returns(self, equity_curve: pd.Series) -> float:
        """
        Calculate recent portfolio returns for drawdown protection using multiple timeframes.
        
        Uses a weighted combination of:
        - Very recent returns (last period)
        - Short-term returns (last 5 periods)
        - Medium-term returns (last 20 periods)
        
        Returns:
        --------
        float : Weighted average of recent returns across timeframes
        """
        if len(equity_curve) < 20:
            return 0.0  # Not enough data
        
        # Calculate returns over different timeframes
        returns_1 = equity_curve.pct_change().iloc[-1]
        returns_5 = (equity_curve.iloc[-1] / equity_curve.iloc[-5] - 1) if len(equity_curve) >= 5 else 0
        returns_20 = (equity_curve.iloc[-1] / equity_curve.iloc[-20] - 1) if len(equity_curve) >= 20 else 0
        
        # Weight the returns (more weight on longer-term performance)
        weighted_returns = (
            0.2 * returns_1 +      # 20% weight on most recent return
            0.3 * returns_5 +      # 30% weight on 5-period return
            0.5 * returns_20       # 50% weight on 20-period return
        )
        
        return weighted_returns

    def _calculate_momentum(self, equity_curve: pd.Series) -> float:
        """Enhanced momentum calculation with faster response"""
        if len(equity_curve) < 50:
            return 0.0
        
        # Add shorter timeframes for faster response
        returns = {
            5: (equity_curve.iloc[-1] / equity_curve.iloc[-5] - 1),   # Added 5-day
            10: (equity_curve.iloc[-1] / equity_curve.iloc[-10] - 1),
            20: (equity_curve.iloc[-1] / equity_curve.iloc[-20] - 1),
            30: (equity_curve.iloc[-1] / equity_curve.iloc[-30] - 1)
        }
        
        # Add shorter moving averages
        mas = {
            5: equity_curve.rolling(5).mean().iloc[-1],    # Added 5-day MA
            10: equity_curve.rolling(10).mean().iloc[-1],
            20: equity_curve.rolling(20).mean().iloc[-1],
            30: equity_curve.rolling(30).mean().iloc[-1]   # Removed 50-day MA
        }
        
        # Updated scoring with more weight on recent data
        scores = {
            'returns': sum(ret > 0 for ret in returns.values()) / len(returns),
            'ma_alignment': sum(mas[p] > mas[q] for p, q in [(5,10), (10,20), (20,30)]) / 3,
            'price_vs_ma': sum(equity_curve.iloc[-1] > ma for ma in mas.values()) / len(mas),
            'short_term': returns[5] > 0 and returns[10] > 0  # Add short-term momentum check
        }
        
        # Weighted average with more emphasis on short-term signals
        return (
            scores['returns'] * 0.4 +
            scores['ma_alignment'] * 0.3 +
            scores['price_vs_ma'] * 0.2 +
            scores['short_term'] * 0.1    # Add short-term component
        )