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
        adf_entry_threshold: float = 0.05,
        adf_exit_threshold: float = 0.10,
        hedge_window: int = 800,
        long_term_window: int = 720,
        short_term_window: int = 240
    ):
        self.window = hedge_window
        self.zscore_threshold = zscore_threshold
        self.adf_entry_threshold = adf_entry_threshold
        self.adf_exit_threshold = adf_exit_threshold
        self.hedge_window = hedge_window
        self.long_term_window = long_term_window
        self.short_term_window = hedge_window
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
        
        asset1_returns = df1['close'].pct_change()
        asset2_returns = df2['close'].pct_change()

        norm_asset1 = (1 + asset1_returns.iloc[-self.hedge_window:]).cumprod()
        norm_asset2 = (1 + asset2_returns.iloc[-self.hedge_window:]).cumprod()
        
        # Calculate hedge ratio using linear regression on normalized data
        beta = np.polyfit(norm_asset2, norm_asset1, 1)[0]

        # Calculate spread using original prices
        spread = norm_asset1 - beta * norm_asset2
        return beta, spread

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
                    if current_zscore >= 0 or current_adf_pvalue >= self.adf_exit_threshold:
                        signal = 0
                else:  # Short spread position
                    # Exit if zscore crosses mean
                    if current_zscore <= 0 or current_adf_pvalue >= self.adf_exit_threshold:
                        signal = 0
            else:
                # Entry conditions when not in position
                if current_adf_pvalue <= self.adf_entry_threshold:  # Check for cointegration (p-value <= threshold)
                    if current_zscore <= -dynamic_threshold:
                        signal = 1
                    elif current_zscore >= dynamic_threshold:
                        signal = -1
        
        return {
            'signal': signal,
            'price1': df1['close'].iloc[-1],
            'price2': df2['close'].iloc[-1],
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