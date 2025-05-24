"""
Backtesting package for strategy testing and performance analysis.
"""

# Import classes directly from their modules to avoid circular imports
from .backtest import PairBacktest
from .performance import PerformanceAnalyzer
from .visualizer import BacktestVisualizer

__all__ = ['PairBacktest', 'PerformanceAnalyzer', 'BacktestVisualizer'] 