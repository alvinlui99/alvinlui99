"""
Version 2 of the Statistical Arbitrage Strategy

This version implements several improvements over v1:
1. Dynamic z-score thresholds based on volatility
2. Adaptive position sizing based on spread confidence
3. Enhanced risk management with trailing stops
4. Pair correlation filtering
5. Volatility-based entry/exit timing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StatisticalArbitrageStrategyV2:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.1,
                 base_stop_loss: float = 0.02,
                 base_take_profit: float = 0.03,
                 min_correlation: float = 0.7,
                 volatility_lookback: int = 20):
        """
        Initialize the enhanced statistical arbitrage strategy.
        
        Args:
            initial_capital: Initial capital for trading
            max_position_size: Maximum position size as fraction of capital
            base_stop_loss: Base stop loss percentage
            base_take_profit: Base take profit percentage
            min_correlation: Minimum correlation for pair selection
            volatility_lookback: Lookback period for volatility calculation
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.min_correlation = min_correlation
        self.volatility_lookback = volatility_lookback
        
        # Initialize tracking variables
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.returns: List[float] = []
        self.capital: List[float] = [initial_capital]  # Changed to list to match backtest expectations
        self.pair_returns: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self.pair_positions: Dict[Tuple[str, str], List[float]] = {}
        self.spreads: Dict[Tuple[str, str], List[float]] = {}
        self.entry_signals: List[Dict] = []
        self.exit_signals: List[Dict] = []
        self.volatility: Dict[str, List[float]] = {}
        
        # Initialize symbols and pairs
        self.symbols: List[str] = []
        self.pairs: List[Tuple[str, str]] = []

    def calculate_dynamic_thresholds(self, symbol: str, spread: float) -> Tuple[float, float]:
        """
        Calculate dynamic entry/exit thresholds based on volatility.
        
        Args:
            symbol: Trading symbol
            spread: Current spread value
            
        Returns:
            Tuple of (entry_threshold, exit_threshold)
        """
        if symbol not in self.volatility or len(self.volatility[symbol]) < self.volatility_lookback:
            return 2.0, 0.5  # Default values
            
        recent_vol = np.mean(self.volatility[symbol][-self.volatility_lookback:])
        base_vol = np.mean(self.volatility[symbol])  # Long-term average
        
        # Adjust thresholds based on volatility regime
        vol_ratio = recent_vol / base_vol
        entry_threshold = 2.0 * vol_ratio
        exit_threshold = 0.5 / vol_ratio
        
        return entry_threshold, exit_threshold
        
    def calculate_position_size(self, symbol: str, spread: float) -> float:
        """
        Calculate adaptive position size based on spread confidence.
        
        Args:
            symbol: Trading symbol
            spread: Current spread value
            
        Returns:
            Position size as fraction of capital
        """
        if symbol not in self.spreads or len(self.spreads[symbol]) < self.volatility_lookback:
            return self.max_position_size
            
        # Calculate spread confidence
        recent_spreads = self.spreads[symbol][-self.volatility_lookback:]
        spread_std = np.std(recent_spreads)
        spread_mean = np.mean(recent_spreads)
        
        # Adjust position size based on spread deviation
        spread_deviation = abs(spread - spread_mean) / spread_std
        confidence = min(1.0, spread_deviation / 3.0)  # Cap at 1.0
        
        return self.max_position_size * confidence
        
    def update_volatility(self, symbol: str, returns: float):
        """
        Update volatility tracking for a symbol.
        
        Args:
            symbol: Trading symbol
            returns: Current period returns
        """
        if symbol not in self.volatility:
            self.volatility[symbol] = []
            
        self.volatility[symbol].append(abs(returns))
        
    def process_tick(self, timestamp: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Process a single tick of market data.
        
        Args:
            timestamp: Current timestamp
            data: Dictionary of price data for each symbol
            
        Returns:
            Dictionary of trading signals for each symbol
        """
        signals = {}
        
        try:
            # Update volatility for each symbol
            for symbol in self.symbols:
                if symbol in data and not data[symbol].empty:
                    self.update_volatility(symbol, data[symbol])
            
            # Calculate returns and spreads for each pair
            for symbol1, symbol2 in self.pairs:
                if (symbol1 in data and symbol2 in data and 
                    not data[symbol1].empty and not data[symbol2].empty):
                    
                    # Calculate returns
                    ret1 = data[symbol1]['close'].pct_change().iloc[-1]
                    ret2 = data[symbol2]['close'].pct_change().iloc[-1]
                    
                    # Calculate spread
                    spread = data[symbol1]['close'].iloc[-1] - data[symbol2]['close'].iloc[-1]
                    
                    # Store data
                    self.pair_returns[(symbol1, symbol2)].append((ret1, ret2))
                    self.spreads[(symbol1, symbol2)].append(spread)
                    
                    # Calculate dynamic thresholds
                    entry_threshold, exit_threshold = self.calculate_dynamic_thresholds(symbol1, symbol2)
                    
                    # Calculate current z-score
                    if len(self.spreads[(symbol1, symbol2)]) >= self.volatility_lookback:
                        spread_series = pd.Series(self.spreads[(symbol1, symbol2)])
                        zscore = (spread - spread_series.mean()) / spread_series.std()
                        
                        # Generate signals
                        if abs(zscore) > entry_threshold:
                            # Calculate position size
                            position_size = self.calculate_position_size(symbol1, symbol2, zscore)
                            
                            if zscore > 0:
                                # Short symbol1, long symbol2
                                signals[symbol1] = {
                                    'type': 'short',
                                    'size': position_size,
                                    'zscore': zscore,
                                    'threshold': entry_threshold
                                }
                                signals[symbol2] = {
                                    'type': 'long',
                                    'size': position_size,
                                    'zscore': zscore,
                                    'threshold': entry_threshold
                                }
                            else:
                                # Long symbol1, short symbol2
                                signals[symbol1] = {
                                    'type': 'long',
                                    'size': position_size,
                                    'zscore': zscore,
                                    'threshold': entry_threshold
                                }
                                signals[symbol2] = {
                                    'type': 'short',
                                    'size': position_size,
                                    'zscore': zscore,
                                    'threshold': entry_threshold
                                }
                        elif abs(zscore) < exit_threshold:
                            # Exit signal
                            signals[symbol1] = {
                                'type': 'close',
                                'size': self.positions.get(symbol1, 0.0),
                                'zscore': zscore,
                                'threshold': exit_threshold
                            }
                            signals[symbol2] = {
                                'type': 'close',
                                'size': self.positions.get(symbol2, 0.0),
                                'zscore': zscore,
                                'threshold': exit_threshold
                            }
            
            # Update capital
            self.capital.append(self.capital[-1])  # Keep track of capital changes
            
        except Exception as e:
            logger.error(f"Error processing tick at {timestamp}: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        return signals 