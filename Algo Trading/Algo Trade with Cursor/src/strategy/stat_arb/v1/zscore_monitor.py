"""
Z-score monitor for statistical arbitrage strategy.

This module provides functionality for monitoring z-scores and generating trading signals.
"""

from typing import Dict, List, Tuple, NamedTuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from binance.um_futures import UMFutures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZScoreSignal(NamedTuple):
    """Trading signal based on z-score."""
    pair: Tuple[str, str]
    signal_type: str  # 'ENTRY_LONG', 'ENTRY_SHORT', 'EXIT'
    zscore: float
    timestamp: datetime

class ZScoreMonitor:
    def __init__(self,
                 client: UMFutures,
                 pairs: List[tuple],
                 lookback_periods: int = 100,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0,
                 timeframe: str = '15m'):
        """
        Initialize z-score monitor.
        
        Args:
            client: Binance Futures API client
            pairs: List of trading pairs to monitor
            lookback_periods: Number of periods for z-score calculation
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_threshold: Z-score threshold for stop loss
            timeframe: Timeframe for analysis
        """
        self.client = client
        self.pairs = pairs
        self.lookback_periods = lookback_periods
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.timeframe = timeframe
        
        # Initialize tracking variables
        self.spreads: Dict[Tuple[str, str], List[float]] = {}
        self.zscores: Dict[Tuple[str, str], List[float]] = {}
        self.signals: List[ZScoreSignal] = []
        
        # Initialize data storage
        self.spread_data: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.zscore_data: Dict[Tuple[str, str], pd.Series] = {}
        self.current_signals: Dict[Tuple[str, str], ZScoreSignal] = {}
        
        # Load initial historical data
        self._load_historical_data()
    
    def _load_historical_data(self) -> None:
        """Load historical price data for all pairs."""
        logger.info("Loading historical data for Z-score calculation...")
        
        for symbol1, symbol2 in self.pairs:
            try:
                # Get historical data for both symbols
                klines1 = self.client.klines(
                    symbol=symbol1,
                    interval=self.timeframe,
                    limit=self.lookback_periods
                )
                klines2 = self.client.klines(
                    symbol=symbol2,
                    interval=self.timeframe,
                    limit=self.lookback_periods
                )
                
                # Convert to DataFrames
                df1 = pd.DataFrame(klines1, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                df2 = pd.DataFrame(klines2, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # Convert types
                df1['close'] = pd.to_numeric(df1['close'], errors='coerce')
                df2['close'] = pd.to_numeric(df2['close'], errors='coerce')
                
                # Drop any NaN values
                df1.dropna(inplace=True)
                df2.dropna(inplace=True)
                
                # Calculate spread
                spread = df1['close'] - df2['close']
                
                # Store spread data
                self.spread_data[(symbol1, symbol2)] = spread
                
                # Calculate initial Z-score
                self._update_zscore(symbol1, symbol2)
                
                logger.info(f"Loaded historical data for {symbol1}-{symbol2}")
                logger.info(f"Spread mean: {spread.mean():.2f}, std: {spread.std():.2f}")
                logger.info(f"Current Z-score: {self.zscore_data[(symbol1, symbol2)].iloc[-1]:.2f}")
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol1}-{symbol2}: {str(e)}")
                logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_zscore(self, symbol1: str, symbol2: str) -> None:
        """Update Z-score for a pair of symbols."""
        try:
            spread = self.spread_data[(symbol1, symbol2)]
            if len(spread) < self.lookback_periods:
                logger.warning(f"Insufficient data for {symbol1}-{symbol2}: {len(spread)} < {self.lookback_periods}")
                return
            
            mean = spread.rolling(window=self.lookback_periods).mean()
            std = spread.rolling(window=self.lookback_periods).std()
            
            # Calculate Z-score
            zscore = (spread - mean) / std
            
            # Store Z-score data
            self.zscore_data[(symbol1, symbol2)] = zscore
            
            # Update current signal
            self._update_signal(symbol1, symbol2, zscore.iloc[-1], spread.iloc[-1], mean.iloc[-1], std.iloc[-1])
            
            logger.debug(f"Updated Z-score for {symbol1}-{symbol2}: {zscore.iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"Error updating Z-score for {symbol1}-{symbol2}: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_signal(
        self,
        symbol1: str,
        symbol2: str,
        zscore: float,
        spread: float,
        mean: float,
        std: float
    ) -> None:
        """Update trading signal based on Z-score."""
        try:
            pair = (symbol1, symbol2)
            current_signal = self.current_signals.get(pair)
            
            # Determine signal type
            if abs(zscore) > self.stop_loss_threshold:
                signal_type = 'EXIT'  # Stop loss
                logger.info(f"Stop loss signal for {symbol1}-{symbol2}: |{zscore:.2f}| > {self.stop_loss_threshold}")
            elif abs(zscore) > self.entry_threshold:
                if zscore > 0:
                    signal_type = 'ENTRY_SHORT'  # Short symbol1, long symbol2
                    logger.info(f"Short entry signal for {symbol1}-{symbol2}: {zscore:.2f} > {self.entry_threshold}")
                else:
                    signal_type = 'ENTRY_LONG'  # Long symbol1, short symbol2
                    logger.info(f"Long entry signal for {symbol1}-{symbol2}: {zscore:.2f} < -{self.entry_threshold}")
            elif abs(zscore) < self.exit_threshold and current_signal and current_signal.signal_type != 'NONE':
                signal_type = 'EXIT'  # Take profit
                logger.info(f"Take profit signal for {symbol1}-{symbol2}: |{zscore:.2f}| < {self.exit_threshold}")
            else:
                signal_type = 'NONE'
                logger.debug(f"No signal for {symbol1}-{symbol2}: {zscore:.2f}")
            
            # Create new signal
            self.current_signals[pair] = ZScoreSignal(
                pair=pair,
                zscore=zscore,
                signal_type=signal_type,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error updating signal for {symbol1}-{symbol2}: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def update_prices(self, symbol1: str, symbol2: str, price1: float, price2: float) -> None:
        """
        Update prices and recalculate Z-score for a pair.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            price1: Current price of first symbol
            price2: Current price of second symbol
        """
        try:
            pair = (symbol1, symbol2)
            if pair not in self.spread_data:
                logger.warning(f"No historical data for pair {symbol1}-{symbol2}")
                return
            
            # Calculate new spread
            new_spread = price1 - price2
            
            # Update spread data
            self.spread_data[pair] = pd.concat([
                self.spread_data[pair].iloc[1:],
                pd.Series([new_spread])
            ])
            
            # Update Z-score
            self._update_zscore(symbol1, symbol2)
            
            logger.debug(f"Updated prices for {symbol1}-{symbol2}: {price1:.2f}, {price2:.2f}")
        except Exception as e:
            logger.error(f"Error updating prices for {symbol1}-{symbol2}: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_signals(self) -> List[ZScoreSignal]:
        """
        Get trading signals based on z-scores.
        
        Returns:
            List of trading signals
        """
        self.signals = []
        
        for pair in self.pairs:
            if pair in self.zscores and self.zscores[pair]:
                zscore = self.zscores[pair][-1]
                
                # Generate signals based on z-score
                if zscore <= -self.entry_threshold:
                    self.signals.append(ZScoreSignal(
                        pair=pair,
                        signal_type='ENTRY_LONG',
                        zscore=zscore,
                        timestamp=datetime.now()
                    ))
                elif zscore >= self.entry_threshold:
                    self.signals.append(ZScoreSignal(
                        pair=pair,
                        signal_type='ENTRY_SHORT',
                        zscore=zscore,
                        timestamp=datetime.now()
                    ))
                elif abs(zscore) <= self.exit_threshold:
                    self.signals.append(ZScoreSignal(
                        pair=pair,
                        signal_type='EXIT',
                        zscore=zscore,
                        timestamp=datetime.now()
                    ))
                    
        return self.signals
    
    def get_pair_status(self, symbol1: str, symbol2: str) -> Optional[Dict]:
        """
        Get current status for a pair.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Dictionary containing current status or None if pair not found
        """
        pair = (symbol1, symbol2)
        if pair not in self.current_signals:
            return None
        
        signal = self.current_signals[pair]
        return {
            'zscore': signal.zscore,
            'signal_type': signal.signal_type,
            'timestamp': signal.timestamp
        }
    
    def get_all_pair_statuses(self) -> Dict[Tuple[str, str], Dict]:
        """Get current status for all pairs."""
        return {
            pair: self.get_pair_status(pair[0], pair[1])
            for pair in self.pairs
        } 