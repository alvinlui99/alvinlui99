"""
Version 2 of the Statistical Arbitrage Strategy

This version implements an enhanced mean reversion strategy with:
- Dynamic z-score thresholds based on volatility
- Adaptive position sizing based on spread confidence
- Enhanced risk management with trailing stops
- Pair correlation filtering
"""

from .statistical_arbitrage import StatisticalArbitrageStrategyV2

__all__ = ['StatisticalArbitrageStrategyV2'] 