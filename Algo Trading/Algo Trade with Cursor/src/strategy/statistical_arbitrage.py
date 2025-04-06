import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from statsmodels.tsa.stattools import coint
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
                 volatility_threshold: float = 0.02,
                 coint_pvalue_threshold: float = 0.05):
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
            coint_pvalue_threshold (float): P-value threshold for cointegration test
        """
        # Convert pairs to trading_pairs format for BaseStrategy
        trading_pairs = []
        for symbol1, symbol2 in pairs:
            trading_pairs.extend([symbol1, symbol2])
        
        super().__init__(trading_pairs=trading_pairs, timeframe=timeframe)
        
        self.client = client
        self.pairs = pairs
        self.lookback_periods = lookback_periods
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.min_confidence = min_confidence
        self.volatility_threshold = volatility_threshold
        self.coint_pvalue_threshold = coint_pvalue_threshold
        
        # Initialize components
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
        
        # Initialize cointegration results
        self.cointegration_results: Dict[Tuple[str, str], Tuple[float, float]] = {}
        
    def test_cointegration(self, data1: pd.Series, data2: pd.Series) -> Tuple[float, float]:
        """
        Test for cointegration between two price series.
        
        Args:
            data1: First price series
            data2: Second price series
            
        Returns:
            Tuple[float, float]: (t-statistic, p-value)
        """
        try:
            t_stat, p_value, _ = coint(data1, data2)
            return t_stat, p_value
        except Exception as e:
            logger.error(f"Error in cointegration test: {str(e)}")
            return float('nan'), float('nan')
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added indicator columns
        """
        try:
            # Calculate volatility
            data['volatility'] = data['close'].pct_change().rolling(
                window=self.lookback_periods
            ).std()
            
            # Calculate moving averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
            data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
            
            # Drop NaN values
            data.dropna(inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate trading signals based on statistical arbitrage strategy.
        
        Args:
            data (Dict[str, pd.DataFrame]): Price data for each symbol
            
        Returns:
            Dict[str, Dict]: Trading signals for each symbol
        """
        signals = {}
        logger.info("\nGenerating signals for pairs:")
        
        for symbol1, symbol2 in self.pairs:
            try:
                # Get price data
                df1 = data[symbol1]
                df2 = data[symbol2]
                
                # Test cointegration if not already done
                if (symbol1, symbol2) not in self.cointegration_results:
                    t_stat, p_value = self.test_cointegration(
                        df1['close'].iloc[-self.lookback_periods:],
                        df2['close'].iloc[-self.lookback_periods:]
                    )
                    self.cointegration_results[(symbol1, symbol2)] = (t_stat, p_value)
                    logger.info(f"Cointegration test for {symbol1}-{symbol2}: t-stat={t_stat:.2f}, p-value={p_value:.4f}")
                
                # Get Z-score signals
                zscore_signals = self.monitor.get_signals()
                
                # Process signals for each pair
                for signal in zscore_signals:
                    if signal.pair == (symbol1, symbol2):
                        # Check cointegration
                        t_stat, p_value = self.cointegration_results[(symbol1, symbol2)]
                        if p_value > self.coint_pvalue_threshold:
                            logger.warning(f"Pair {symbol1}-{symbol2} not cointegrated (p-value={p_value:.4f})")
                            continue
                        
                        # Check volatility
                        vol1 = df1['volatility'].iloc[-1]
                        vol2 = df2['volatility'].iloc[-1]
                        if vol1 > self.volatility_threshold or vol2 > self.volatility_threshold:
                            logger.warning(f"High volatility for {symbol1}-{symbol2}: {vol1:.4f}, {vol2:.4f}")
                            continue
                        
                        # Calculate position sizes
                        confidence = 1 - abs(signal.zscore) / self.stop_loss_threshold
                        position_size = self.sizer.calculate_position_size(
                            symbol1=symbol1,
                            symbol2=symbol2,
                            confidence=confidence,
                            zscore=signal.zscore
                        )
                        
                        if position_size <= 0:
                            logger.warning(f"Zero position size for {symbol1}-{symbol2}")
                            continue
                        
                        # Generate signals
                        if signal.signal_type == 'ENTRY_LONG':
                            signals[symbol1] = {
                                'action': 'BUY',
                                'size': position_size,
                                'confidence': confidence,
                                'zscore': signal.zscore
                            }
                            signals[symbol2] = {
                                'action': 'SELL',
                                'size': position_size,
                                'confidence': confidence,
                                'zscore': signal.zscore
                            }
                        elif signal.signal_type == 'ENTRY_SHORT':
                            signals[symbol1] = {
                                'action': 'SELL',
                                'size': position_size,
                                'confidence': confidence,
                                'zscore': signal.zscore
                            }
                            signals[symbol2] = {
                                'action': 'BUY',
                                'size': position_size,
                                'confidence': confidence,
                                'zscore': signal.zscore
                            }
                        elif signal.signal_type == 'EXIT':
                            signals[symbol1] = {'action': 'CLOSE'}
                            signals[symbol2] = {'action': 'CLOSE'}
                        
                        logger.info(f"Generated signal for {symbol1}-{symbol2}: {signal.signal_type}")
                        logger.info(f"Position size: {position_size:.4f}, Confidence: {confidence:.4f}")
                        
            except Exception as e:
                logger.error(f"Error generating signals for {symbol1}-{symbol2}: {str(e)}")
                logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        return signals
        
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate trading signal.
        
        Args:
            signal (Dict): Trading signal
            
        Returns:
            bool: True if signal is valid
        """
        if not signal:
            return False
            
        # Check confidence
        if signal['confidence'] < self.min_confidence:
            logger.info(f"Signal rejected: Low confidence ({signal['confidence']:.2f} < {self.min_confidence})")
            return False
            
        # Check volatility
        if signal['vol1'] > self.volatility_threshold or signal['vol2'] > self.volatility_threshold:
            logger.info(f"Signal rejected: High volatility (vol1: {signal['vol1']:.4f}, vol2: {signal['vol2']:.4f})")
            return False
            
        # Check if z-score is within stop loss threshold
        if abs(signal['zscore']) > self.stop_loss_threshold:
            logger.info(f"Signal rejected: Z-score beyond stop loss threshold ({abs(signal['zscore']):.2f} > {self.stop_loss_threshold})")
            return False
            
        return True
        
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
        if not signal:
            return 0.0
            
        # Calculate base position size as percentage of account balance
        base_size = account_balance * self.max_position_size
        
        # Adjust size based on confidence
        confidence_factor = min(signal['confidence'], 1.0)
        position_size = base_size * confidence_factor
        
        # Adjust for volatility
        vol = signal['vol1'] if symbol == signal['symbol1'] else signal['vol2']
        vol_factor = max(0, 1 - vol/self.volatility_threshold)
        position_size *= vol_factor
        
        # Get current price
        price = signal['price1'] if symbol == signal['symbol1'] else signal['price2']
        
        # Convert to quantity
        quantity = position_size / price
        
        logger.info(f"Position size for {symbol}: {quantity:.6f} (${position_size:.2f})")
        return quantity 