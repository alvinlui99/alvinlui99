from typing import List

class RegimeConfig:
    LOOKBACK_PERIOD = 252

    # Indicator Configs
    class IndicatorConfig:
        MA_SHORT = 50
        MA_LONG = 200
        ADX_PERIOD = 14
        RSI_PERIOD = 14
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        ATR_PERIOD = 14

    class SignalBlendConfig:
        WEIGHTS = {
            'primary_trends': 0.4,
            'momentum_signals': 0.3,
            'volatility_signals': 0.2,
            'pattern_signals': 0.1
        }

    class PrimaryTrendConfig:
        WEIGHTS = {
            'ma_signal': 0.25,
            'ichimoku_signal': 0.25,
            'dmi_signal': 0.20,
            'psar_signal': 0.15,
            'bbands_signal': 0.15
        }

    class MovingAverageConfig:
        NORM_FACTOR = 0.02
        POSITION_SINGAL_AMPLIFIER = 50
        SLOPE_DURATION = 5
        SLOPE_SINGAL_AMPLIFIER = 50
        HISTORICAL_SPREAD_DURATION = 20
        WEIGHTS = {
            'cross': 0.35,
            'position': 0.25,
            'slope': 0.25,
            'spread': 0.15
        }

    class IchimokuConfig:
        TENKAN_PERIOD = 9
        KIJUN_PERIOD = 26
        SENKOU_B_PERIOD = 52
        CLOUD_THICKNESS_DURATION = 26
        MOMENTUM_DURATION = 5
        MOMENTUM_AMPLIFIER = 20
        WEIGHTS = {
            'cloud_position': 0.4,
            'tk_cross': 0.3,
            'thickness': 0.1,
            'momentum': 0.2
        }

    class DMIConfig:
        ADX_PERIOD = 14
        ADX_STRENGTH_THRESHOLD = 50
        ADX_MOMENTUM_DURATION = 5
        ADX_MOMENTUM_AMPLIFIER = 10
        DI_CROSSOVER_INTENSITY_AMPLIFIER = 5
        WEIGHTS = {
            'di_strength': 0.4,
            'adx_strength': 0.3,
            'adx_momentum': 0.15,
            'crossover': 0.15
        }

    class PsarConfig:
        FAST_ACCELERATION = 0.02
        FAST_MAXIMUM = 0.2
        SLOW_ACCELERATION = 0.01
        SLOW_MAXIMUM = 0.1
        MAX_DISTANCE_FACTOR = 0.05
        VELOCITY_DURATION = 5
        VELOCITY_AMPLIFIER = 20
        WEIGHTS = {
            'fast_signal': 0.3,
            'slow_signal': 0.3,
            'reversal': 0.2,
            'velocity': 0.2
        }

    class BollingerBandsConfig:
        PERIOD = 20
        NBDEVUP = 2
        NBDEVDN = 2
        MATYPE = 0
        BANDWIDTH_SMA_PERIOD = 20
        SLOPE_DURATION = 5
        SLOPE_AMPLIFIER = 20
        VELOCITY_DURATION = 5
        VELOCITY_AMPLIFIER = 20
        WEIGHTS = {
            'position': 0.35,
            'bandwidth': 0.25,
            'slope': 0.20,
            'velocity': 0.20
        }

    class MomentumConfig:
        WEIGHTS = {
            'rsi_signal': 0.2,
            'macd_signal': 0.2,
            'price_momentum_signal': 0.2,
            'smi_signal': 0.2,
            'mfi_signal': 0.1,
            'willr_signal': 0.1
        }

    class RSIConfig:
        OVERSOLD_THRESHOLD = 30
        OVERBOUGHT_THRESHOLD = 70
        BASE_WEIGHT = 0.6
        MOMENTUM_WEIGHT = 0.2
        EXTREME_WEIGHT = 0.2

    class MACDConfig:
        WEIGHTS = {
            'crossover': 0.3,
            'hist_strength': 0.2,
            'hist_momentum': 0.2,
            'momentum': 0.15,
            'zero_line': 0.15
        }

    class PriceMomentumConfig:
        PERIODS = [5, 10, 20, 30]  # Multiple timeframes for RoC
        ROC_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # Weights for different timeframes

        WEIGHTS = {
            'roc': 0.35,
            'acceleration': 0.25,
            'ma_convergence': 0.20,
            'velocity': 0.20
        }

        # Normalization Factors
        ROC_NORMALIZATION_FACTOR = 0.1
        ACCELERATION_NORMALIZATION_FACTOR = 0.02
        MA_CONVERGENCE_NORMALIZATION_FACTOR = 0.05
        VELOCITY_NORMALIZATION_FACTOR = 2.0

        # Other Parameters
        VELOCITY_PERIOD = 5

    class SMIConfig:
        PERIOD = 10
        SMOOTH1 = 3
        SMOOTH2 = 3
        SIGNAL_PERIOD = 10
        EXTREME_THRESHOLD = 40

        WEIGHTS = {
            'base': 0.4,
            'momentum': 0.2,
            'crossover': 0.25,
            'extreme': 0.15
        }

    class MFIConfig:
        PERIOD = 14
        OVERBOUGHT = 80
        OVERSOLD = 20
        DIVERGENCE_PERIOD = 10
        DIVERGENCE_FACTOR = 5.0

        WEIGHTS = {
            'base': 0.35,
            'momentum': 0.25,
            'extreme': 0.20,
            'divergence': 0.20
        }

    class WILLRConfig:
        # Williams %R Parameters
        PERIODS = [14, 28, 42]  # Multiple timeframes
        MTF_WEIGHTS = [0.5, 0.3, 0.2]  # Weights for different timeframes
        OVERSOLD_THRESHOLD = 20
        OVERBOUGHT_THRESHOLD = 80
        REVERSAL_LOOKBACK = 5

        WEIGHTS = {
            'base': 0.3,
            'momentum': 0.2,
            'extreme': 0.2,
            'mtf': 0.15,
            'reversal': 0.15
        }

    class VolatilitySignalConfig:
        # Main weights for combining different volatility signals
        WEIGHTS = {
            'historical': 0.30,  # Traditional volatility measure
            'garch': 0.25,      # Forward-looking volatility
            'parkinson': 0.25,  # High-low based volatility
            'yang_zhang': 0.20  # Comprehensive OHLC volatility
        }

    class HistoricalVolConfig:
        # Periods for multi-timeframe analysis (in days)
        PERIODS = [5, 10, 21, 63]  # ~1 week, 2 weeks, 1 month, 3 months
        PERIOD_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # More weight to recent periods
        
        # General parameters
        TREND_PERIOD = 21
        TREND_FACTOR = 2.0
        
        # Regime detection windows
        SHORT_WINDOW = 5
        LONG_WINDOW = 21
        REGIME_FACTOR = 2.0
        
        # Distribution analysis
        DISTRIBUTION_WINDOW = 63
        SKEW_WEIGHT = 0.4
        KURT_WEIGHT = 0.6
        
        # Term structure analysis
        TERM_STRUCTURE_PERIODS = [5, 10, 21, 63]
        TERM_STRUCTURE_FACTOR = 2.0
        
        # Component weights
        WEIGHTS = {
            'trend': 0.35,
            'regime': 0.25,
            'distribution': 0.20,
            'term_structure': 0.20
        }

    class GarchConfig:
        # GARCH(1,1) parameters
        OMEGA = 0.000025  # Long-term variance weight
        ALPHA = 0.1      # Short-term shock weight
        BETA = 0.85      # Persistence factor
        
        # General parameters
        TREND_PERIOD = 21
        TREND_FACTOR = 2.0
        LT_WINDOW = 252  # One year for long-term variance
        
        # Persistence analysis
        PERSISTENCE_LAG = 5
        PERSISTENCE_FACTOR = 2.0
        
        # Shock analysis
        SHOCK_WINDOW = 5
        SHOCK_FACTOR = 2.0
        
        # Forecast parameters
        FORECAST_WINDOW = 10
        FORECAST_FACTOR = 2.0
        
        # Component weights
        WEIGHTS = {
            'trend': 0.30,
            'persistence': 0.25,
            'shock': 0.25,
            'forecast': 0.20
        }

    class ParkinsonConfig:
        # General parameters
        TREND_PERIOD = 21
        TREND_FACTOR = 2.0
        
        # Range analysis
        RANGE_PERIOD = 10
        RANGE_FACTOR = 2.0
        
        # Multi-timeframe analysis
        MTF_PERIODS = [5, 10, 21]
        MTF_WEIGHTS = [0.5, 0.3, 0.2]
        MTF_FACTOR = 2.0
        
        # Component weights
        WEIGHTS = {
            'trend': 0.35,
            'range': 0.25,
            'relative': 0.20,
            'mtf': 0.20
        }

    class YangZhangConfig:
        # General parameters
        WINDOW = 21
        TREND_PERIOD = 21
        TREND_FACTOR = 2.0
        K = 0.34  # Weight for open-close volatility (typically 0.34)
        
        # Gap analysis
        GAP_WINDOW = 5
        GAP_FACTOR = 2.0
        
        # Intraday analysis
        INTRADAY_WINDOW = 10
        INTRADAY_FACTOR = 2.0
        
        # Component analysis
        COMPONENT_FACTOR = 2.0
        
        # Component weights
        WEIGHTS = {
            'trend': 0.30,
            'overnight': 0.25,
            'intraday': 0.25,
            'component': 0.20
        }

    class PatternSignalConfig:
        WEIGHTS = {
            'candlestick': 0.30,
            'harmonic': 0.25,
            'chart': 0.25,
            'volume': 0.20
        }

    class CandlestickConfig:
        WEIGHTS = {
            'single': 0.25,    # Single candlestick patterns
            'double': 0.30,    # Double candlestick patterns
            'triple': 0.30,    # Triple candlestick patterns
            'confirmation': 0.15  # Volume and price action confirmation
        }

        # Single pattern weights
        SINGLE_WEIGHTS = [0.35, 0.35, 0.30]  # Hammer/Shooting Star, Engulfing, Doji

        # Double pattern weights
        DOUBLE_WEIGHTS = [0.4, 0.3, 0.3]  # Engulfing, Harami, Tweezer

        # Triple pattern weights
        TRIPLE_WEIGHTS = [0.3, 0.3, 0.2, 0.2]  # Star, Soldiers/Crows, Inside, Outside

        # General parameters
        LOOKBACK = 20         # Period for average calculations
        TREND_SHORT = 5       # Short-term trend period
        TREND_LONG = 20       # Long-term trend period
        MIN_BODY_RATIO = 0.3  # Minimum body size relative to average
        
        # Hammer/Hanging Man parameters
        HAMMER_UPPER_RATIO = 0.25  # Maximum upper shadow to body ratio
        HAMMER_LOWER_RATIO = 2.0   # Minimum lower shadow to body ratio
        
        # Shooting Star parameters
        STAR_UPPER_RATIO = 2.0    # Minimum upper shadow to body ratio
        STAR_LOWER_RATIO = 0.25   # Maximum lower shadow to body ratio
        MAX_STAR_BODY_RATIO = 0.3 # Maximum body size relative to average
        
        # Doji parameters
        DOJI_RATIO = 0.1  # Maximum body size relative to average
        
        # Engulfing parameters
        ENGULFING_FACTOR = 1.1  # Minimum size factor for engulfing candle
        
        # Harami parameters
        HARAMI_RATIO = 0.6  # Maximum size ratio of harami to mother candle
        
        # Tweezer parameters
        TWEEZER_TOLERANCE = 0.001  # Maximum difference in highs/lows
        
        # Star pattern parameters
        STAR_BODY_RATIO = 0.25  # Maximum middle candle body size
        
        # Three Soldiers/Crows parameters
        SOLDIER_SHADOW_RATIO = 0.2  # Maximum upper shadow ratio for soldiers
        CROW_SHADOW_RATIO = 0.2     # Maximum lower shadow ratio for crows
        
        # Three Inside parameters
        INSIDE_RATIO = 0.7  # Maximum size ratio of inside candle
        
        # Confirmation parameters
        VOL_PERIOD = 20      # Volume moving average period
        RANGE_PERIOD = 20    # Price range moving average period
        CONFIRMATION_FACTOR = 2.0  # Scaling factor for confirmation signals

        # Pattern strength parameters
        MAX_STRENGTH = 2.0  # Maximum pattern strength multiplier

    class HarmonicConfig:
        # Main component weights
        WEIGHTS = {
            'gartley': 0.30,    # Most common pattern
            'butterfly': 0.25,   # Reliable but less common
            'bat': 0.25,        # Good reliability
            'crab': 0.20        # Most extreme pattern
        }
        
        # Swing point detection
        SWING_WINDOW = 5        # Window for swing point detection
        SWING_THRESHOLD = 0.01  # Minimum price change for swing point (1%)
        
        # Pattern tolerances
        GARTLEY_TOLERANCE = 0.05   # 5% tolerance for Gartley ratios
        BUTTERFLY_TOLERANCE = 0.05  # 5% tolerance for Butterfly ratios
        BAT_TOLERANCE = 0.05       # 5% tolerance for Bat ratios
        CRAB_TOLERANCE = 0.05      # 5% tolerance for Crab ratios
        
        # Gartley pattern ratios
        GARTLEY_RATIOS = {
            'XA_BC': 0.618,  # B retraces 61.8% of XA
            'AB_CD': 1.0,    # CD same size as AB
            'XA_AD': 0.786   # D retraces 78.6% of XA
        }
        
        # Butterfly pattern ratios
        BUTTERFLY_RATIOS = {
            'XA_BC': 0.786,  # B retraces 78.6% of XA
            'AB_CD': 1.618,  # CD extends 161.8% of AB
            'XA_AD': 1.27    # D extends 127.2% of XA
        }
        
        # Bat pattern ratios
        BAT_RATIOS = {
            'XA_BC': 0.382,  # B retraces 38.2% of XA
            'AB_CD': 1.618,  # CD extends 161.8% of AB
            'XA_AD': 0.886   # D retraces 88.6% of XA
        }
        
        # Crab pattern ratios
        CRAB_RATIOS = {
            'XA_BC': 0.382,  # B retraces 38.2% of XA
            'AB_CD': 2.618,  # CD extends 261.8% of AB
            'XA_AD': 1.618   # D extends 161.8% of XA
        }

    class ChartPatternConfig:
        # Main component weights
        WEIGHTS = {
            'sr': 0.30,        # Support/Resistance
            'trendline': 0.25, # Trendlines
            'formation': 0.25, # Chart formations
            'channel': 0.20    # Price channels
        }
        
        # Support/Resistance parameters
        SR_WINDOW = 100        # Lookback window for S/R levels
        SR_THRESHOLD = 0.02    # Minimum price difference between levels (2%)
        SR_FACTOR = 2.0        # Signal scaling factor
        PROXIMITY_FACTOR = 10  # Distance weight factor
        BIN_SIZE_FACTOR = 0.001  # Price clustering bin size
        MIN_TOUCHES = 3        # Minimum touches for valid S/R level
        MAX_LEVELS = 5         # Maximum number of S/R levels to track
        RECENCY_DECAY = 100    # Decay factor for touch recency
        
        # Trendline parameters
        TRENDLINE_WINDOW = 20   # Window for trendline calculation
        TRENDLINE_FACTOR = 2.0  # Signal scaling factor
        MIN_SLOPE = 0.0001      # Minimum valid trendline slope
        BREAK_THRESHOLD = 0.02  # Breakout threshold (2%)
        TEST_STRENGTH = 0.5     # Signal strength for trendline tests
        
        # Formation parameters
        FORMATION_WEIGHTS = {
            'hs': 0.35,         # Head and Shoulders
            'double': 0.25,     # Double Top/Bottom
            'triangle': 0.20,   # Triangle
            'rectangle': 0.20   # Rectangle
        }
        FORMATION_THRESHOLD = 0.02  # Pattern recognition threshold
        VOLUME_CONFIRM_FACTOR = 1.5  # Volume confirmation requirement
        
        # Channel parameters
        CHANNEL_WINDOW = 20     # Price channel window
        CHANNEL_FACTOR = 2.0    # Signal scaling factor

        # Head and Shoulders
        HS_WINDOW = 5
        HS_MIN_PATTERN_BARS = 15
        HS_MAX_PATTERN_BARS = 60
        HS_SYMMETRY_TOLERANCE = 0.15
        HS_NECKLINE_TOLERANCE = 0.1
        HS_HEAD_PROMINENCE = 0.03
        
        # Double Top/Bottom
        DOUBLE_WINDOW = 5
        DOUBLE_TOLERANCE = 0.015
        DOUBLE_MIN_PATTERN_BARS = 10
        DOUBLE_MAX_PATTERN_BARS = 40
        DOUBLE_PRICE_TOLERANCE = 0.02
        DOUBLE_CONFIRMATION_BARS = 3

        # Triangle Pattern
        TRIANGLE_WINDOW = 5  # Window for peak/trough detection
        TRIANGLE_MIN_POINTS = 4  # Minimum points needed to confirm triangle
        TRIANGLE_MIN_PATTERN_BARS = 15  # Minimum bars for triangle formation
        TRIANGLE_MAX_PATTERN_BARS = 60  # Maximum bars for triangle pattern
        TRIANGLE_SLOPE_TOLERANCE = 0.02  # 2% tolerance for horizontal lines
        TRIANGLE_CONVERGENCE_ANGLE = 10  # Minimum angle between trend lines (degrees)
        TRIANGLE_BREAKOUT_THRESHOLD = 0.02  # 2% breakout confirmation threshold
        
        # Rectangle Pattern
        RECTANGLE_WINDOW = 20  # Window for support/resistance detection
        RECTANGLE_MIN_TOUCHES = 3  # Minimum touches of support/resistance
        RECTANGLE_MIN_PATTERN_BARS = 20  # Minimum bars for rectangle formation
        RECTANGLE_MAX_PATTERN_BARS = 100  # Maximum bars for rectangle pattern
        RECTANGLE_HEIGHT_RATIO = 0.1  # Maximum height as % of price (10%)
        RECTANGLE_TOLERANCE = 0.02  # 2% tolerance for support/resistance levels
        RECTANGLE_BREAKOUT_THRESHOLD = 0.03  # 3% breakout confirmation threshold

    class VolumePatternConfig:
        # Main component weights
        WEIGHTS = {
            'trend': 0.30,        # Volume trend and divergences
            'pv': 0.25,           # Price-volume relationships
            'breakout': 0.25,     # Volume breakouts
            'accumulation': 0.20  # Accumulation/Distribution
        }
        
        # Volume trend parameters
        VOL_SHORT_PERIOD = 5
        VOL_LONG_PERIOD = 20
        PRICE_SHORT_PERIOD = 5
        PRICE_LONG_PERIOD = 20
        TREND_THRESHOLD = 0.02
        TREND_FACTOR = 2.0
        DIVERGENCE_FACTOR = 2.0
        
        # Price-volume relationship parameters
        CORRELATION_PERIOD = 20
        FORCE_PERIOD = 10
        PV_FACTOR = 2.0
        
        # Breakout parameters
        BREAKOUT_PERIOD = 20
        MOMENTUM_PERIOD = 5
        CLIMAX_THRESHOLD = 3.0  # Z-score threshold
        MOMENTUM_THRESHOLD = 0.02
        BREAKOUT_FACTOR = 1.5
        
        # Accumulation/Distribution parameters
        ADL_SHORT_PERIOD = 5
        ADL_LONG_PERIOD = 20
        ADL_THRESHOLD = 0.02
        ADL_FACTOR = 2.0
        ADL_MOMENTUM_FACTOR = 1.5