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
        self.capital: List[float] = [initial_capital]
        self.pair_returns: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self.pair_positions: Dict[Tuple[str, str], List[float]] = {}
        self.spreads: Dict[Tuple[str, str], List[float]] = {}
        self.entry_signals: List[Dict] = []
        self.exit_signals: List[Dict] = []
        self.volatility: Dict[str, List[float]] = {}
        
        # Initialize symbols and pairs
        self.symbols: List[str] = []
        self.pairs: List[Tuple[str, str]] = []
        
        # Add warmup tracking
        self.warmup_periods = volatility_lookback * 2  # Double the lookback for warmup
        self.periods_processed = 0
        
        # Add spread thresholds - adjusted for better sensitivity
        self.min_spread_std = 0.001  # Reduced threshold for normalized spread (percentage)
        self.min_spread = 0.001  # Minimum spread value (percentage)
        self.spread_multiplier = 1000  # Multiplier to increase spread sensitivity
        self.volatility_adjustment_factor = 0.5  # Factor to adjust thresholds based on volatility

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
        
    def calculate_position_size(self, symbol1: str, symbol2: str, zscore: float) -> float:
        """
        Calculate adaptive position size based on spread confidence and asset prices.
        
        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            zscore: Current z-score of the spread
            
        Returns:
            Position size in base currency (USDT)
        """
        pair = (symbol1, symbol2)
        if pair not in self.spreads or len(self.spreads[pair]) < self.volatility_lookback:
            return 0.0  # Return 0 if we don't have enough data
            
        # Calculate spread confidence based on z-score
        confidence = min(1.0, abs(zscore) / 2.0)  # Changed to 2.0 for more responsiveness
        confidence = confidence ** 1.5  # Less aggressive squaring
        
        # Calculate base position value in USDT
        base_position_value = self.capital[-1] * self.max_position_size * confidence
        
        # Adjust position value based on volatility
        if symbol1 in self.volatility and symbol2 in self.volatility:
            vol1 = np.mean(self.volatility[symbol1][-self.volatility_lookback:])
            vol2 = np.mean(self.volatility[symbol2][-self.volatility_lookback:])
            vol_ratio = min(vol1, vol2) / max(vol1, vol2)
            # Use a less severe volatility adjustment
            base_position_value *= (0.5 + 0.5 * vol_ratio)  # Linear interpolation between 0.5 and 1.0
            
        # Set reasonable minimum and maximum position values
        min_position_value = self.capital[-1] * 0.01  # 1% of capital
        max_position_value = self.capital[-1] * 0.05  # 5% of capital
        
        # Ensure position value is within bounds
        position_value = max(min(base_position_value, max_position_value), min_position_value)
        
        # Add detailed logging
        logger.info(f"Position size calculation for {symbol1}-{symbol2}:")
        logger.info(f"Z-score: {zscore:.4f}")
        logger.info(f"Confidence: {confidence:.4f}")
        logger.info(f"Volatility ratio: {vol_ratio:.4f}")
        logger.info(f"Base position value: ${base_position_value:.2f}")
        logger.info(f"Final position value: ${position_value:.2f}")
            
        return position_value
        
    def update_volatility(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Update volatility tracking for a symbol.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with price data
        """
        if symbol not in self.volatility:
            self.volatility[symbol] = []
            
        # Calculate returns
        returns = data['close'].pct_change()
        if not returns.empty:
            # Only append if we have valid returns
            if not pd.isna(returns.iloc[-1]):
                # Calculate volatility on rolling window of returns
                if len(returns) >= self.volatility_lookback:
                    # Get the last volatility_lookback periods of returns
                    recent_returns = returns.iloc[-self.volatility_lookback:]
                    current_volatility = recent_returns.std()
                    
                    # Add detailed logging for SHIB and DOGE
                    if symbol in ["1000SHIBUSDT", "DOGEUSDT"]:
                        logger.info(f"\nUpdating volatility for {symbol}:")
                        logger.info(f"Recent returns (last 5): {recent_returns.iloc[-5:].tolist()}")
                        logger.info(f"Current volatility (std): {current_volatility:.8f}")
                    
                    # Update volatility history
                    self.volatility[symbol].append(current_volatility)
                    
                    # Keep only the last volatility_lookback periods
                    if len(self.volatility[symbol]) > self.volatility_lookback:
                        self.volatility[symbol] = self.volatility[symbol][-self.volatility_lookback:]
                        if symbol in ["1000SHIBUSDT", "DOGEUSDT"]:
                            logger.info(f"Volatility history length: {len(self.volatility[symbol])}")

    def initialize_pairs(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize trading pairs based on available data.
        
        Args:
            data: Dictionary of price data for each symbol
        """
        # Get available symbols
        self.symbols = list(data.keys())
        
        # Initialize pairs
        self.pairs = []
        for i in range(len(self.symbols)):
            for j in range(i + 1, len(self.symbols)):
                symbol1 = self.symbols[i]
                symbol2 = self.symbols[j]
                
                # Initialize tracking for this pair
                self.pair_returns[(symbol1, symbol2)] = []
                self.pair_positions[(symbol1, symbol2)] = []
                self.spreads[(symbol1, symbol2)] = []
                
                # Add to pairs list
                self.pairs.append((symbol1, symbol2))
                
        logger.info(f"Initialized {len(self.pairs)} trading pairs")

    def calculate_spread(self, price1: float, price2: float) -> float:
        """
        Calculate normalized spread between two prices using log returns.
        
        Args:
            price1: Price of first asset
            price2: Price of second asset
            
        Returns:
            Normalized spread value
        """
        # Calculate log returns to normalize price scales
        log_price1 = np.log(price1)
        log_price2 = np.log(price2)
        
        # Calculate spread as difference in log prices
        spread = log_price1 - log_price2
        
        return spread

    def process_tick(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> Dict[str, Dict]:
        """
        Process a single tick of market data.
        
        Args:
            data: Dictionary of price data for each symbol
            timestamp: Current timestamp
            
        Returns:
            Dictionary of trading signals for each symbol
        """
        signals = {}
        skipped_pairs = 0
        total_pairs = 0
        skip_reasons = {
            'invalid_returns': 0,
            'insufficient_data': 0,
            'small_std': 0,
            'small_spread': 0,
            'nan_zscore': 0,
            'infinite_zscore': 0,
            'other': 0
        }
        
        try:
            # Initialize pairs if not done yet
            if not self.pairs:
                self.initialize_pairs(data)
            
            # Update volatility for each symbol
            for symbol in self.symbols:
                if symbol in data and not data[symbol].empty:
                    self.update_volatility(symbol, data[symbol])
            
            # Increment periods processed
            self.periods_processed += 1
            
            # Log price scales for debugging
            if self.periods_processed == 0:
                logger.info("\nPrice scales for each symbol:")
                for symbol in self.symbols:
                    if symbol in data and not data[symbol].empty:
                        price = data[symbol]['close'].iloc[-1]
                        logger.info(f"{symbol}: {price:.8f}")
            
            # Calculate returns and spreads for each pair
            for symbol1, symbol2 in self.pairs:
                total_pairs += 1
                
                if (symbol1 in data and symbol2 in data and 
                    not data[symbol1].empty and not data[symbol2].empty):
                    
                    # Get price data
                    current_idx1 = data[symbol1].index.get_loc(timestamp)
                    current_idx2 = data[symbol2].index.get_loc(timestamp)
                    
                    price1 = data[symbol1]['close'].iloc[current_idx1]
                    price2 = data[symbol2]['close'].iloc[current_idx2]
                    
                    prev_price1 = data[symbol1]['close'].iloc[current_idx1 - 1] if current_idx1 > 0 else price1
                    prev_price2 = data[symbol2]['close'].iloc[current_idx2 - 1] if current_idx2 > 0 else price2
                    
                    # Log detailed price information for debugging
                    if symbol1 == "1000SHIBUSDT" and symbol2 == "DOGEUSDT":
                        logger.info(f"\nDetailed price stats for {symbol1}-{symbol2} at {timestamp}:")
                        logger.info(f"Current prices: {price1:.8f} vs {price2:.8f}")
                        logger.info(f"Previous prices: {prev_price1:.8f} vs {prev_price2:.8f}")
                        logger.info(f"Price changes: {(price1 - prev_price1):.8f} vs {(price2 - prev_price2):.8f}")
                        logger.info(f"Price returns: {((price1 - prev_price1) / prev_price1):.8f} vs {((price2 - prev_price2) / prev_price2):.8f}")
                    
                    # Calculate returns
                    ret1 = (price1 - prev_price1) / prev_price1
                    ret2 = (price2 - prev_price2) / prev_price2
                    
                    # Skip if we have invalid returns
                    if pd.isna(ret1) or pd.isna(ret2):
                        logger.debug(f"Skipping {symbol1}-{symbol2}: Invalid returns (ret1={ret1}, ret2={ret2})")
                        skipped_pairs += 1
                        skip_reasons['invalid_returns'] += 1
                        continue
                    
                    # Calculate normalized spread
                    spread = self.calculate_spread(price1, price2)
                    
                    # Skip if spread is too small
                    if abs(spread) < self.min_spread:
                        logger.debug(f"Skipping {symbol1}-{symbol2}: Spread too small ({spread:.2e})")
                        skipped_pairs += 1
                        skip_reasons['small_spread'] += 1
                        continue
                    
                    # Store data
                    self.pair_returns[(symbol1, symbol2)].append((ret1, ret2))
                    
                    # Initialize spread list if not exists
                    if (symbol1, symbol2) not in self.spreads:
                        self.spreads[(symbol1, symbol2)] = []
                    
                    # Store current spread
                    self.spreads[(symbol1, symbol2)].append(spread)
                    
                    # Keep only the last volatility_lookback periods
                    if len(self.spreads[(symbol1, symbol2)]) > self.volatility_lookback:
                        self.spreads[(symbol1, symbol2)] = self.spreads[(symbol1, symbol2)][-self.volatility_lookback:]
                    
                    # Skip trading during warmup
                    if self.periods_processed < self.warmup_periods:
                        logger.debug(f"Accumulating data for {symbol1}-{symbol2} during warmup (period {self.periods_processed}/{self.warmup_periods})")
                        continue
                    
                    # Calculate dynamic thresholds
                    entry_threshold, exit_threshold = self.calculate_dynamic_thresholds(symbol1, spread)
                    
                    # Calculate current z-score
                    if len(self.spreads[(symbol1, symbol2)]) >= self.volatility_lookback:
                        spread_series = pd.Series(self.spreads[(symbol1, symbol2)])
                        
                        # Skip if we don't have enough valid data points
                        valid_points = spread_series.count()
                        if valid_points < self.volatility_lookback:
                            logger.debug(f"Skipping {symbol1}-{symbol2}: Insufficient valid data points ({valid_points} < {self.volatility_lookback})")
                            skipped_pairs += 1
                            skip_reasons['insufficient_data'] += 1
                            continue
                            
                        # Calculate z-score with proper error handling
                        try:
                            spread_mean = spread_series.mean()
                            spread_std = spread_series.std()
                            
                            # Calculate dynamic minimum std threshold based on pair volatility
                            pair_volatility = min(
                                np.mean(self.volatility[symbol1][-self.volatility_lookback:]) if symbol1 in self.volatility else 0,
                                np.mean(self.volatility[symbol2][-self.volatility_lookback:]) if symbol2 in self.volatility else 0
                            )
                            dynamic_min_std = self.min_spread_std * (1 + self.volatility_adjustment_factor * pair_volatility)
                            
                            # Skip if standard deviation is too small
                            if spread_std < dynamic_min_std:
                                logger.debug(f"Skipping {symbol1}-{symbol2}: Standard deviation too small ({spread_std:.2e} < {dynamic_min_std:.2e})")
                                logger.debug(f"Pair volatility: {pair_volatility:.2e}, Dynamic threshold: {dynamic_min_std:.2e}")
                                skipped_pairs += 1
                                skip_reasons['small_std'] += 1
                                continue
                                
                            zscore = (spread - spread_mean) / spread_std
                            
                            # Calculate volatility ratio for logging
                            vol1 = np.mean(self.volatility[symbol1][-self.volatility_lookback:]) if symbol1 in self.volatility else 0
                            vol2 = np.mean(self.volatility[symbol2][-self.volatility_lookback:]) if symbol2 in self.volatility else 0
                            
                            # Add detailed volatility logging
                            if symbol1 == "1000SHIBUSDT" and symbol2 == "DOGEUSDT":
                                logger.info(f"\nDetailed volatility stats for {symbol1}-{symbol2}:")
                                logger.info(f"Recent volatility {symbol1}: {vol1:.8f}")
                                logger.info(f"Recent volatility {symbol2}: {vol2:.8f}")
                                logger.info(f"Volatility ratio: {min(vol1, vol2) / max(vol1, vol2) if max(vol1, vol2) > 0 else 0:.4f}")
                            
                            vol_ratio = min(vol1, vol2) / max(vol1, vol2) if max(vol1, vol2) > 0 else 0
                            
                            # Log detailed spread statistics for debugging
                            if symbol1 == "1000SHIBUSDT" and symbol2 == "DOGEUSDT":
                                logger.info(f"\nDetailed spread stats for {symbol1}-{symbol2}:")
                                logger.info(f"Current spread: {spread:.8f}")
                                logger.info(f"Spread mean: {spread_mean:.8f}")
                                logger.info(f"Spread std: {spread_std:.8f}")
                                logger.info(f"Entry threshold: {entry_threshold:.4f}")
                                logger.info(f"Exit threshold: {exit_threshold:.4f}")
                                logger.info(f"Z-score: {zscore:.4f}")
                            
                            # Skip if z-score is invalid
                            if pd.isna(zscore):
                                logger.debug(f"Skipping {symbol1}-{symbol2}: NaN z-score")
                                skipped_pairs += 1
                                skip_reasons['nan_zscore'] += 1
                                continue
                            elif not np.isfinite(zscore):
                                logger.debug(f"Skipping {symbol1}-{symbol2}: Infinite z-score ({zscore})")
                                skipped_pairs += 1
                                skip_reasons['infinite_zscore'] += 1
                                continue
                            
                            # Generate signals
                            if abs(zscore) > entry_threshold:
                                # Calculate position size
                                position_size = self.calculate_position_size(symbol1, symbol2, zscore)
                                
                                # Add debug logging for trade execution
                                logger.info(f"\nTrade execution for {symbol1}-{symbol2}:")
                                logger.info(f"Z-score: {zscore:.4f}")
                                logger.info(f"Position size: {position_size:.4f}")
                                logger.info(f"Current capital: {self.capital[-1]:.2f}")
                                
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
                                
                                # Log trade details
                                logger.info(f"Trade signals generated:")
                                logger.info(f"{symbol1}: {signals[symbol1]}")
                                logger.info(f"{symbol2}: {signals[symbol2]}")
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
                        except Exception as e:
                            logger.warning(f"Error calculating z-score for {symbol1}-{symbol2}: {str(e)}")
                            skipped_pairs += 1
                            skip_reasons['other'] += 1
                            continue
            
            # Log skipped pairs statistics
            if skipped_pairs > 0:
                logger.info(f"Skipped {skipped_pairs}/{total_pairs} pairs at {timestamp}")
                logger.info(f"Skip reasons: {skip_reasons}")
            
            # Update capital
            self.capital.append(self.capital[-1])  # Keep track of capital changes
            
        except Exception as e:
            logger.error(f"Error processing tick at {timestamp}: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        return signals 