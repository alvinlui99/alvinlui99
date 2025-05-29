import logging
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
from statsmodels.tsa.stattools import adfuller
from collections import deque

from strategy.pair_selector import PairIndicators

class PairStrategy:
    def __init__(
        self, 
        window: int = 240, 
        zscore_threshold: float = 3.0, 
        adf_threshold: float = 0.05,
        hedge_window: int = 240,
        long_term_window: int = 720,
        short_term_window: int = 240
    ):
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.adf_threshold = adf_threshold
        self.hedge_window = hedge_window
        self.long_term_window = long_term_window
        self.short_term_window = short_term_window
        self.logger = logging.getLogger(__name__)
        # self.model = pickle.load(open('hmm_model.pkl', 'rb'))
        self.spread = deque(maxlen=800)

    def calculate_hedge_ratio(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[float, pd.Series]:
        """
        Calculate rolling hedge ratio using linear regression on log prices.
        Returns both the hedge ratio (beta) and the spread.
        """
        # Align dataframes
        df1, df2 = PairIndicators.align_dataframes(df1, df2)
        
        if len(df1) < self.hedge_window:
            return np.nan, pd.Series(dtype=float, index=df1.index)
        
        # Calculate the latest values using the last hedge_window rows
        window_df1 = np.log(df1['close'].iloc[-self.hedge_window:])
        window_df2 = np.log(df2['close'].iloc[-self.hedge_window:])
        
        # Calculate hedge ratio using linear regression
        beta = np.polyfit(window_df2, window_df1, 1)[0]
        
        # Calculate new spread value and append to deque
        new_spread = np.log(df1['close'].iloc[-1]) - beta * np.log(df2['close'].iloc[-1])
        self.spread.append(new_spread)
        
        # Convert deque to pandas Series for return
        spread_series = pd.Series(self.spread, index=df1.index[-len(self.spread):])
        return beta, spread_series

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
        min_threshold = 2.5
        max_threshold = 4.0
        return np.clip(adjusted_threshold, min_threshold, max_threshold)

    def calculate_adf_statistic(self, spread: pd.Series) -> float:
        """
        Calculate Augmented Dickey-Fuller test p-value for the spread.
        Returns the ADF test p-value.
        """
        # Remove NaN values
        spread = spread.dropna()
        
        if len(spread) < self.window:
            return 1.0  # Return high p-value when not enough data
            
        # Perform ADF test
        adf_result = adfuller(spread, autolag='AIC')
        return adf_result[1]  # Return the p-value

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
                'adf_pvalue': float,
                'hedge_ratio': float,
                'zscore': float,
                'vol_ratio': float,
                'dynamic_threshold': float,
                'half_life': float
            }
        """
        hedge_ratio, spread = self.calculate_hedge_ratio(df1, df2)
        adf_pvalue = self.calculate_adf_statistic(spread)

        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = (spread - spread_mean) / spread_std
        
        # Calculate volatility ratio and dynamic threshold
        vol_ratio = self.calculate_volatility_ratio(zscore)
        dynamic_threshold = self.calculate_dynamic_threshold(vol_ratio)
        
        # Get current values
        current_adf_pvalue = adf_pvalue
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
                    # Exit if zscore crosses mean
                    if current_zscore >= 0:
                        signal = 0
                else:  # Short spread position
                    # Exit if zscore crosses mean
                    if current_zscore <= 0:
                        signal = 0
            else:
                # Entry conditions when not in position
                if current_adf_pvalue <= self.adf_threshold:  # Check for cointegration (p-value <= threshold)
                    if current_zscore <= -dynamic_threshold:
                        signal = 1
                    elif current_zscore >= dynamic_threshold:
                        signal = -1
        
        return {
            'signal': signal,
            'adf_pvalue': current_adf_pvalue,
            'hedge_ratio': hedge_ratio,
            'zscore': current_zscore,
            'vol_ratio': vol_ratio,
            'dynamic_threshold': dynamic_threshold,
            'spread': spread.iloc[-1],
            'spread_mean': spread_mean.iloc[-1],
            'spread_std': spread_std.iloc[-1]
        }

    def analyze_signals(self, signals: pd.DataFrame) -> Dict:
        """Analyze signal generation for debugging/optimization."""
        analysis = {
            'total_signals': len(signals[signals['signal'] != 0]),
            'long_signals': len(signals[signals['signal'] == 1]),
            'short_signals': len(signals[signals['signal'] == -1]),
            'avg_adf': signals['adf_pvalue'].mean(),
            'avg_zscore': signals['zscore'].mean(),
            'signal_clusters': self._analyze_signal_clusters(signals)
        }
        return analysis

    def _analyze_signal_clusters(self, signals: pd.DataFrame) -> Dict:
        """Analyze if signals are clustered in time (potential issue)."""
        # Implementation for analyzing signal clustering
        pass