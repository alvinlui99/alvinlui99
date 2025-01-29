from typing import List

class FeatureConfig:
    LOOKBACK_PERIOD = 14        # Base lookback period for features
    
    # Technical Indicators
    TECHNICAL_INDICATORS = {
        'RSI_PERIOD': 14,
        'MACD_FAST': 12,
        'MACD_SLOW': 26,
        'MACD_SIGNAL': 9,
        'BB_PERIOD': 20,
        'BB_STD': 2
    }

    FEATURE_NAMES = [
        'return',
        'price',
        'high',
        'low',
        'open',
        'volume',
        'volatility',
        'momentum',
        'price_to_sma',
        'price_std',
        'rsi',
        'macd',
        'macd_signal',
        'bb_position',
        'skewness',
        'kurtosis'
    ]