import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from strategy.pair_selector import PairIndicators

class PairStrategy:
    def __init__(
        self, 
        window: int = 20, 
        zscore_threshold: float = 2.0, 
        correlation_threshold: float = 0.8, 
        hedge_window: int = 30,
        long_term_window: int = 168,  # For long-term volatility
        short_term_window: int = 24   # For short-term volatility
    ):
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.correlation_threshold = correlation_threshold
        self.hedge_window = hedge_window
        self.long_term_window = long_term_window
        self.short_term_window = short_term_window
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

    def calculate_volatility_ratio(self, zscore: pd.Series) -> float:
        """
        Calculate the ratio of long-term to short-term zscore volatility.
        Higher ratio indicates more stable market conditions.
        """
        # Only calculate std for the last window of data we need
        long_term_vol = zscore.iloc[-self.long_term_window:].std()
        short_term_vol = zscore.iloc[-self.short_term_window:].std()
        
        # Avoid division by zero
        vol_ratio = 1 if long_term_vol == 0 else short_term_vol / long_term_vol
        
        return vol_ratio

    def calculate_dynamic_threshold(self, vol_ratio: float) -> float:
        """
        Adjust zscore threshold based on volatility ratio.
        Higher vol_ratio (more stable) -> lower threshold
        Lower vol_ratio (more volatile) -> higher threshold
        """
        # Base threshold adjustment
        base_threshold = self.zscore_threshold
        
        # Adjust threshold based on vol_ratio
        # If vol_ratio > 1: market is more volatile than usual -> higher threshold
        # If vol_ratio < 1: market is more stable than usual -> lower threshold
        adjusted_threshold = base_threshold * vol_ratio
        
        # Set reasonable bounds for the threshold
        min_threshold = 1.5
        max_threshold = 3.0
        return np.clip(adjusted_threshold, min_threshold, max_threshold)

    def generate_signals(self, df1: pd.DataFrame, df2: pd.DataFrame, position_state: Dict) -> Dict:
        """
        Generate trading signals with position state awareness and dynamic thresholds.
        
        Args:
            df1: First asset DataFrame
            df2: Second asset DataFrame
            position_state: Dictionary containing current position information
                {
                    'in_position': bool,
                    'position_type': int (1 for long spread, -1 for short spread),
                    'unrealised_pnl': float
                }
                
        Returns:
            Dictionary containing current signal and metrics:
            {
                'signal': int (1 for long, -1 for short, 0 for no position),
                'correlation': float,
                'hedge_ratio': float,
                'zscore': float,
                'vol_ratio': float,
                'dynamic_threshold': float
            }
        """
        hedge_ratio, spread = self.calculate_hedge_ratio(df1, df2)
        correlation = PairIndicators.calculate_correlation(df1, df2, self.window)

        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = (spread - spread_mean) / spread_std
        
        # Calculate volatility ratio and dynamic threshold
        vol_ratio = self.calculate_volatility_ratio(zscore)
        dynamic_threshold = self.calculate_dynamic_threshold(vol_ratio)
        
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
            
            if in_position:
                # Default to holding position
                signal = position_type

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
                    if current_zscore <= -dynamic_threshold:
                        signal = 1
                    elif current_zscore >= dynamic_threshold:
                        signal = -1
        
        return {
            'signal': signal,
            'correlation': current_correlation,
            'hedge_ratio': current_hedge_ratio,
            'zscore': current_zscore,
            'vol_ratio': vol_ratio,
            'dynamic_threshold': dynamic_threshold
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