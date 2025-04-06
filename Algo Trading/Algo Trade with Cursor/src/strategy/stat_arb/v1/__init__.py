"""
Version 1 of the Statistical Arbitrage Strategy

This version implements the basic mean reversion strategy with:
- Fixed z-score thresholds for entry/exit
- Simple position sizing
- Basic risk management
"""

from .statistical_arbitrage import StatisticalArbitrageStrategyV1

__all__ = ['StatisticalArbitrageStrategyV1'] 