import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from strategy.pair_selector import PairIndicators

class PairStrategy:
    def __init__(self, window: int = 20, zscore_threshold: float = 2.0, correlation_threshold: float = 0.8, hedge_window: int = 30):
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.correlation_threshold = correlation_threshold
        self.hedge_window = hedge_window
        self.logger = logging.getLogger(__name__)  # For debugging/analysis

    def calculate_hedge_ratio(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate rolling hedge ratio using linear regression.
        Returns both the hedge ratio (beta) and the spread.
        """
        # Align dataframes
        df1, df2 = PairIndicators.align_dataframes(df1, df2)
        
        # Initialize series for hedge ratio and spread
        hedge_ratio = pd.Series(index=df1.index, dtype=float)
        spread = pd.Series(index=df1.index, dtype=float)
        
        # Calculate rolling hedge ratio
        for i in range(self.hedge_window, len(df1)):
            window_df1 = df1['close'].iloc[i-self.hedge_window:i]
            window_df2 = df2['close'].iloc[i-self.hedge_window:i]
            
            # Calculate hedge ratio using linear regression
            beta = np.polyfit(window_df2, window_df1, 1)[0]
            hedge_ratio.iloc[i] = beta
            
            # Calculate spread using current hedge ratio
            spread.iloc[i] = df1['close'].iloc[i] - beta * df2['close'].iloc[i]
        
        return hedge_ratio, spread

    def generate_signals(self, df1: pd.DataFrame, df2: pd.DataFrame, position_state: Dict = None) -> Dict:
        """
        Generate trading signals with position state awareness.
        
        Args:
            df1: First asset DataFrame
            df2: Second asset DataFrame
            position_state: Dictionary containing current position information
                {
                    'in_position': bool,
                    'position_type': int (1 for long spread, -1 for short spread),
                    'entry_zscore': float
                }
                
        Returns:
            Dictionary containing current signal and metrics:
            {
                'signal': int (1 for long, -1 for short, 0 for no position),
                'correlation': float,
                'hedge_ratio': float,
                'zscore': float
            }
        """
        # Calculate metrics for current point
        hedge_ratio, spread = self.calculate_hedge_ratio(df1, df2)
        correlation = PairIndicators.calculate_correlation(df1, df2, self.window)

        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = (spread - spread_mean) / spread_std
        
        # Get current values
        current_correlation = correlation.iloc[-1]
        current_hedge_ratio = hedge_ratio.iloc[-1]
        current_zscore = zscore.iloc[-1]
        
        # Initialize signal
        signal = 0
        
        # If we have position state information, use it to adjust signal generation
        if position_state is not None:
            in_position = position_state.get('in_position', False)
            position_type = position_state.get('position_type', 0)
            entry_zscore = position_state.get('entry_zscore', 0)
            
            if in_position:
                # Exit conditions when in position
                if position_type == 1:  # Long spread position
                    # Exit if zscore crosses mean or correlation drops
                    if current_zscore >= 0 or current_correlation < self.correlation_threshold:
                        signal = 0
                else:  # Short spread position
                    # Exit if zscore crosses mean or correlation drops
                    if current_zscore <= 0 or current_correlation < self.correlation_threshold:
                        signal = 0
            else:
                # Entry conditions when not in position
                if current_correlation >= self.correlation_threshold:
                    if current_zscore <= -self.zscore_threshold:
                        signal = 1
                    elif current_zscore >= self.zscore_threshold:
                        signal = -1
        
        return {
            'signal': signal,
            'correlation': current_correlation,
            'hedge_ratio': current_hedge_ratio,
            'zscore': current_zscore
        }

    def analyze_signals(self, signals: pd.DataFrame) -> Dict:
        """Analyze signal generation for debugging/optimization."""
        analysis = {
            'total_signals': len(signals[signals['signal'] != 0]),
            'long_signals': len(signals[signals['signal'] == 1]),
            'short_signals': len(signals[signals['signal'] == -1]),
            'avg_correlation': signals['correlation'].mean(),
            'avg_zscore': signals['zscore'].mean(),
            'signal_clusters': self._analyze_signal_clusters(signals)
        }
        return analysis

    def _analyze_signal_clusters(self, signals: pd.DataFrame) -> Dict:
        """Analyze if signals are clustered in time (potential issue)."""
        # Implementation for analyzing signal clustering
        pass