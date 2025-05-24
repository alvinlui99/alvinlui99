"""
Statistical Arbitrage Strategy Package

This package contains different versions of the statistical arbitrage strategy.
Each version implements improvements and optimizations over the previous one.

Versions:
- v1: Initial implementation with basic mean reversion
- v2: Enhanced version with dynamic thresholds and position sizing
"""

from .v1 import StatisticalArbitrageStrategyV1
from .v2 import StatisticalArbitrageStrategyV2

__all__ = ['StatisticalArbitrageStrategyV1', 'StatisticalArbitrageStrategyV2'] 