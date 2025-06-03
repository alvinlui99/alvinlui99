import logging
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
from collections import deque

from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from strategy.pair_selector import PairIndicators

class PairStrategy:
    def __init__(
        self, 
        zscore_threshold: float = 2.0, 
        adf_entry_threshold: float = 0.05,
        adf_exit_threshold: float = 0.10,
        hedge_window: int = 96,
        long_term_window: int = 96,
        short_term_window: int = 24
    ):
        self.window = hedge_window
        self.zscore_threshold = zscore_threshold
        self.adf_entry_threshold = adf_entry_threshold
        self.adf_exit_threshold = adf_exit_threshold
        self.hedge_window = hedge_window
        self.long_term_window = long_term_window
        self.short_term_window = short_term_window
        self.logger = logging.getLogger(__name__)
        # self.model = pickle.load(open('hmm_model.pkl', 'rb'))
        self.spread_history = deque(maxlen=72)

    def calculate_hedge_ratio(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate dynamic hedge ratio using Kalman Filter on log prices.
        Returns both the hedge ratio (beta) and the spread.
        """
        # Align dataframes
        df1, df2 = PairIndicators.align_dataframes(df1, df2)

        beta = OLS(df1['close'].values, df2['close'].values).fit().params[0]
        spread = df1['close'] - beta * df2['close']
        pvalue = adfuller(spread)[1]
        return beta, spread.iloc[-1], pvalue

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
        hedge_ratio, spread, pvalue = self.calculate_hedge_ratio(df1, df2)
        self.spread_history.append(spread)
        if len(self.spread_history) < 72:
            return None
        else:
            spread_mean = np.mean(self.spread_history)
            spread_std = np.std(self.spread_history)
            zscore = (spread - spread_mean) / spread_std
        
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
                        if zscore >= 0 or pvalue >= self.adf_exit_threshold:
                            signal = 0
                    else:  # Short spread position
                        # Exit if zscore crosses mean
                        if zscore <= 0 or pvalue >= self.adf_exit_threshold:
                            signal = 0
                else:
                    # Entry conditions when not in position
                    if pvalue <= self.adf_entry_threshold:  # Check for cointegration (p-value <= threshold)
                        if zscore <= -self.zscore_threshold:
                            signal = 1
                        elif zscore >= self.zscore_threshold:
                            signal = -1
            
            return {
                'signal': signal,
                'price1': df1['close'].iloc[-1],
                'price2': df2['close'].iloc[-1],
                'pvalue': pvalue,
                'hedge_ratio': hedge_ratio,
                'zscore': zscore,
                'spread': spread,
                'spread_mean': spread_mean,
                'spread_std': spread_std
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