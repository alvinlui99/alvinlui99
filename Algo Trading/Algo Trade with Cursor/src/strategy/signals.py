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

    def generate_signals(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with detailed logging for analysis."""
        hedge_ratio, spread = self.calculate_hedge_ratio(df1, df2)
        correlation = PairIndicators.calculate_correlation(df1, df2, self.window)
        # zscore = PairIndicators.calculate_pair_zscore(df1, df2, self.window)

        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = (spread - spread_mean) / spread_std
        
        signals = pd.DataFrame(index=df1.index)
        signals['correlation'] = correlation
        signals['hedge_ratio'] = hedge_ratio
        signals['zscore'] = zscore
        
        signals['signal'] = 0
        long_condition = (signals['correlation'] >= self.correlation_threshold) & (signals['zscore'] <= -self.zscore_threshold)
        short_condition = (signals['correlation'] >= self.correlation_threshold) & (signals['zscore'] >= self.zscore_threshold)
        
        signals.loc[long_condition, 'signal'] = 1
        signals.loc[short_condition, 'signal'] = -1
        
        return signals

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