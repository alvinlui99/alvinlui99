import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from .base_strategy import BaseStrategy
from .zscore_monitor import ZScoreMonitor, ZScoreSignal
from .position_sizer import PositionSizer

logger = logging.getLogger(__name__)

class StatisticalArbitrageStrategy(BaseStrategy):
    def __init__(self,
                 client,
                 pairs: List[tuple],
                 lookback_periods: int = 100,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0,
                 timeframe: str = '15m',
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.2,
                 max_leverage: float = 2.0,
                 min_confidence: float = 0.6,
                 volatility_threshold: float = 0.02):
        """
        Initialize statistical arbitrage strategy.
        
        Args:
            client: Binance Futures API client
            pairs (List[tuple]): List of trading pairs to monitor
            lookback_periods (int): Number of periods for Z-score calculation
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
            stop_loss_threshold (float): Z-score threshold for stop loss
            timeframe (str): Timeframe for analysis
            initial_capital (float): Initial capital for position sizing
            max_position_size (float): Maximum position size as fraction of capital
            max_leverage (float): Maximum leverage allowed
            min_confidence (float): Minimum confidence required for trade
            volatility_threshold (float): Maximum allowed volatility
        """
        # Convert pairs to trading_pairs format for BaseStrategy
        trading_pairs = []
        for symbol1, symbol2 in pairs:
            trading_pairs.extend([symbol1, symbol2])
        
        super().__init__(trading_pairs=trading_pairs, timeframe=timeframe)
        
        self.client = client
        self.pairs = pairs
        self.monitor = ZScoreMonitor(
            client=client,
            pairs=pairs,
            lookback_periods=lookback_periods,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_loss_threshold,
            timeframe=timeframe
        )
        self.sizer = PositionSizer(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            max_leverage=max_leverage,
            min_confidence=min_confidence,
            volatility_threshold=volatility_threshold
        )
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with indicators
        """
        # Add any additional indicators here
        return data
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate trading signals based on Z-score monitoring.
        
        Args:
            data (Dict[str, pd.DataFrame]): Historical data for each pair
            
        Returns:
            Dict[str, Dict]: Trading signals
        """
        signals = {}
        
        # Update Z-scores and get current signals
        for pair in self.pairs:
            symbol1, symbol2 = pair
            pair_key = f"{symbol1}-{symbol2}"
            
            try:
                # Get current prices
                price1 = data[symbol1].iloc[-1]['close']  # Changed from 'Close' to 'close'
                price2 = data[symbol2].iloc[-1]['close']  # Changed from 'Close' to 'close'
                
                # Update monitor with current prices
                self.monitor.update_prices(symbol1=symbol1, symbol2=symbol2, price1=price1, price2=price2)
                
                # Get current signal
                status = self.monitor.get_pair_status(symbol1, symbol2)
                
                if status and status['signal_type'] in ['ENTRY_LONG', 'ENTRY_SHORT']:
                    # Create ZScoreSignal object
                    signal = ZScoreSignal(
                        pair=pair,
                        zscore=status['zscore'],
                        spread=status['spread'],
                        mean=status['mean'],
                        std=status['std'],
                        timestamp=status['timestamp'],
                        signal_type=status['signal_type']
                    )
                    
                    # Calculate position size
                    position = self.sizer.calculate_position_size(
                        signal=signal,
                        price1=price1,
                        price2=price2
                    )
                    
                    if position:
                        signals[pair_key] = {
                            'action': 'BUY' if status['signal_type'] == 'ENTRY_LONG' else 'SELL',
                            'size': position.size1,
                            'stop_loss': price1 * (1 - 0.02),  # 2% stop loss
                            'take_profit': price1 * (1 + 0.04)  # 4% take profit
                        }
            except Exception as e:
                logger.error(f"Error generating signal for {pair_key}: {str(e)}")
                continue
        
        return signals
        
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate trading signal.
        
        Args:
            signal (Dict): Trading signal
            
        Returns:
            bool: True if signal is valid
        """
        return True  # Add validation logic here
        
    def calculate_position_size(self, symbol: str, signal: Dict, account_balance: float) -> float:
        """
        Calculate position size based on signal and account balance.
        
        Args:
            symbol (str): Trading pair
            signal (Dict): Trading signal
            account_balance (float): Current account balance
            
        Returns:
            float: Position size
        """
        return signal['size'] 