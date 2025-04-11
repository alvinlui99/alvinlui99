"""
Enhanced backtest engine with improved reliability and robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self,
                 strategy: BaseStrategy,
                 initial_capital: float = 10000,
                 commission: float = 0.0000,
                 max_position_size: float = 0.1,
                 risk_per_trade: float = 0.02):
        """
        Initialize backtesting framework with enhanced parameters.
        
        Args:
            strategy (BaseStrategy): Trading strategy to test
            initial_capital (float): Initial capital in USDT
            commission (float): Commission rate per trade
            max_position_size (float): Maximum position size as fraction of capital
            risk_per_trade (float): Maximum risk per trade as fraction of capital
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        
        # Initialize tracking variables
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.peak_equity: float = initial_capital
        
        # Initialize performance metrics
        self.metrics: Dict = {}
        
        # Initialize error tracking
        self.errors: List[Dict] = []
        
    def _validate_data(self, data: dict) -> Tuple[bool, str]:
        """
        Validate input data for backtesting.
        
        Args:
            data: Dictionary of DataFrames for each symbol
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not data:
            return False, "No data provided"
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for symbol, df in data.items():
            if not all(col in df.columns for col in required_columns):
                return False, f"Missing required columns for {symbol}"
                
            # Check for NaN values
            if df[required_columns].isna().any().any():
                return False, f"NaN values found in {symbol} data"
                
            # Check for infinite values
            if np.isinf(df[required_columns]).any().any():
                return False, f"Infinite values found in {symbol} data"
                
        return True, ""
        
    def _synchronize_data(self, data: dict) -> dict:
        """
        Synchronize data across all symbols.
        
        Args:
            data: Dictionary of DataFrames for each symbol
            
        Returns:
            Synchronized data dictionary
        """
        # Get common index across all symbols
        common_index = None
        for symbol, df in data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
                
        if common_index is None or len(common_index) == 0:
            raise ValueError("No common data points found across symbols")
            
        # Filter data to common index
        synchronized_data = {}
        for symbol, df in data.items():
            synchronized_data[symbol] = df.loc[common_index]
            
        return synchronized_data
        
    def _update_positions(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> float:
        """
        Update positions and calculate PnL with enhanced error handling.
        
        Args:
            data: Historical data
            timestamp: Current timestamp
            
        Returns:
            Updated capital
        """
        current_capital = self.equity_curve[-1]
        
        for symbol, position in list(self.positions.items()):
            try:
                if position['size'] != 0 and symbol in data:
                    # Get the price using the correct column name
                    if 'close' in data[symbol].columns:
                        current_price = data[symbol].loc[timestamp, 'close']
                    else:
                        logger.error(f"Cannot find price column for {symbol}")
                        continue
                        
                    # Calculate PnL with proper error handling
                    try:
                        pnl = (current_price - position['entry_price']) * position['size']
                        if position['side'] == 'SELL':
                            pnl = -pnl
                            
                        # Update position value
                        position['current_value'] = position['size'] * current_price
                        position['unrealized_pnl'] = pnl
                        
                        # Check stop loss and take profit
                        if position['side'] == 'BUY':
                            if current_price <= position['stop_loss']:
                                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                                self._close_position(symbol, current_price, timestamp)
                            elif current_price >= position['take_profit']:
                                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                                self._close_position(symbol, current_price, timestamp)
                        else:  # SELL
                            if current_price >= position['stop_loss']:
                                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                                self._close_position(symbol, current_price, timestamp)
                            elif current_price <= position['take_profit']:
                                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                                self._close_position(symbol, current_price, timestamp)
                                
                    except Exception as e:
                        logger.error(f"Error calculating PnL for {symbol}: {str(e)}")
                        self.errors.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'error': str(e),
                            'type': 'pnl_calculation'
                        })
                        
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {str(e)}")
                self.errors.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'error': str(e),
                    'type': 'position_update'
                })
                
        return current_capital
        
    def _close_position(self, symbol: str, price: float, timestamp: datetime) -> None:
        """
        Close a position with proper error handling.
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Current timestamp
        """
        try:
            position = self.positions[symbol]
            pnl = (price - position['entry_price']) * position['size']
            if position['side'] == 'SELL':
                pnl = -pnl
                
            # Record trade
            self._record_trade(symbol, 'CLOSE', position['size'], price, pnl, timestamp)
            
            # Update capital
            self.equity_curve[-1] += pnl - (abs(position['size']) * price * self.commission)
            
            # Remove position
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            self.errors.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'error': str(e),
                'type': 'position_close'
            })
            
    def _calculate_enhanced_metrics(self) -> Dict:
        """
        Calculate enhanced performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_pnl': 0.0,
                'return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'trades': [],
                'equity_curve': [],
                'drawdown_curve': [],
                'errors': self.errors
            }
            
        # Calculate basic metrics
        final_capital = self.equity_curve[-1]
        total_pnl = final_capital - self.initial_capital
        returns = (final_capital - self.initial_capital) / self.initial_capital
        
        # Calculate win rate
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # Calculate maximum drawdown
        peak = self.initial_capital
        max_drawdown = 0.0
        for capital in self.equity_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        daily_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 else 0
        
        # Calculate additional metrics
        avg_win = np.mean([trade['pnl'] for trade in self.trades if trade['pnl'] > 0]) if any(trade['pnl'] > 0 for trade in self.trades) else 0
        avg_loss = np.mean([trade['pnl'] for trade in self.trades if trade['pnl'] < 0]) if any(trade['pnl'] < 0 for trade in self.trades) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate Calmar ratio
        calmar_ratio = returns / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Calculate Sortino ratio
        downside_returns = [r for r in daily_returns if r < 0]
        sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'return': returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'errors': self.errors
        }
        
    def run(self, data: dict) -> dict:
        """
        Run the backtest on historical data with enhanced error handling.
        
        Args:
            data: Dictionary of DataFrames for each symbol
            
        Returns:
            dict: Backtest results
        """
        # Validate data
        is_valid, error_message = self._validate_data(data)
        if not is_valid:
            logger.error(f"Data validation failed: {error_message}")
            return {
                'error': error_message,
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_pnl': 0.0,
                'return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'trades': [],
                'equity_curve': [],
                'drawdown_curve': [],
                'errors': self.errors
            }
            
        try:
            # Synchronize data
            data = self._synchronize_data(data)
            
            # Initialize positions and trades
            self.positions = {}
            self.trades = []
            self.equity_curve = [self.initial_capital]
            self.drawdown_curve = [0.0]
            self.peak_equity = self.initial_capital
            
            # Calculate indicators for each symbol
            for symbol, df in data.items():
                try:
                    data[symbol] = self.strategy.calculate_indicators(df)
                except Exception as e:
                    logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                    self.errors.append({
                        'timestamp': None,
                        'symbol': symbol,
                        'error': str(e),
                        'type': 'indicator_calculation'
                    })
                    continue
                    
            # Get common index across all symbols
            common_index = None
            for symbol, df in data.items():
                if common_index is None:
                    common_index = df.index
                else:
                    common_index = common_index.intersection(df.index)
                    
            if common_index is None or len(common_index) == 0:
                logger.error("No common data points found across symbols")
                return {
                    'error': "No common data points found across symbols",
                    'initial_capital': self.initial_capital,
                    'final_capital': self.initial_capital,
                    'total_pnl': 0.0,
                    'return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'trades': [],
                    'equity_curve': [],
                    'drawdown_curve': [],
                    'errors': self.errors
                }
                
            logger.info(f"Running backtest with {len(common_index)} data points")
            
            # Run backtest for each time step
            for timestamp in common_index:
                try:
                    current_data = {}
                    for symbol, df in data.items():
                        current_data[symbol] = df.loc[timestamp:timestamp]
                        
                    # Generate signals and execute trades
                    signals = self.strategy.generate_signals(current_data)
                    
                    # Execute trades based on signals
                    for symbol, signal in signals.items():
                        if signal != 0:  # Non-zero signal indicates a trade
                            self._execute_trade(symbol, signal, current_data[symbol])
                            
                    # Update positions and calculate PnL
                    current_capital = self._update_positions(current_data, timestamp)
                    
                    # Update equity curve
                    self.equity_curve.append(current_capital)
                    
                    # Update drawdown curve
                    if current_capital > self.peak_equity:
                        self.peak_equity = current_capital
                    drawdown = (self.peak_equity - current_capital) / self.peak_equity
                    self.drawdown_curve.append(drawdown)
                    
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {str(e)}")
                    self.errors.append({
                        'timestamp': timestamp,
                        'symbol': None,
                        'error': str(e),
                        'type': 'timestamp_processing'
                    })
                    continue
                    
            # Calculate performance metrics
            results = self._calculate_enhanced_metrics()
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {
                'error': str(e),
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_pnl': 0.0,
                'return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'trades': [],
                'equity_curve': [],
                'drawdown_curve': [],
                'errors': self.errors
            }
    
    def _execute_trade(self, symbol: str, signal: float, data: pd.DataFrame) -> None:
        """
        Execute a trade based on a signal.
        
        Args:
            symbol (str): Trading pair
            signal (float): Trade signal
            data (pd.DataFrame): Historical data for the symbol
        """
        if signal != 0:  # Non-zero signal indicates a trade
            if signal > 0:
                self._open_position(symbol, 'BUY', data.loc[data.index[-1], 'close'], data)
            else:
                self._open_position(symbol, 'SELL', data.loc[data.index[-1], 'close'], data)
    
    def _open_position(self, symbol: str, side: str, price: float, data: pd.DataFrame) -> None:
        """
        Open a new position.
        
        Args:
            symbol (str): Trading pair
            side (str): Position side ('BUY' or 'SELL')
            price (float): Trade price
            data (pd.DataFrame): Historical data for the symbol
        """
        # Calculate position size based on risk per trade
        size = self.max_position_size * self.initial_capital
        
        # Calculate stop loss and take profit levels
        stop_loss_distance = self.risk_per_trade * price
        take_profit_distance = 2 * stop_loss_distance
        
        # Open new position
        self.positions[symbol] = {
            'size': size,
            'side': side,
            'entry_price': price,
            'stop_loss': price - stop_loss_distance,
            'take_profit': price + take_profit_distance
        }
        self._record_trade(symbol, side, size, price, 0, data.index[-1])
    
    def _record_trade(self, symbol: str, action: str, size: float, price: float, pnl: float, timestamp: datetime) -> None:
        """
        Record a trade.
        
        Args:
            symbol (str): Trading pair
            action (str): Trade action (BUY/SELL/CLOSE)
            size (float): Position size
            price (float): Trade price
            pnl (float): Profit/loss
            timestamp (datetime): Trade timestamp
        """
        # For entry trades, store raw position size without commission
        if action in ['BUY', 'SELL']:
            raw_size = size / (1 + self.commission)
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'size': raw_size,
                'price': price,
                'pnl': pnl
            })
        else:
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'size': size,
                'price': price,
                'pnl': pnl
            })
    
    def _execute_trades(self, signals: Dict[str, Dict], data: Dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """
        Execute trades based on signals.
        
        Args:
            signals (Dict[str, Dict]): Trading signals
            data (Dict[str, pd.DataFrame]): Historical data
            timestamp (datetime): Current timestamp
        """
        current_capital = self.equity_curve[-1]
        
        for pair_key, signal in signals.items():
            logger.info(f"\nProcessing signal for pair {pair_key} at {timestamp}")
            logger.info(f"Signal action: {signal['action']}")
            
            # Extract symbols from pair key
            symbol1, symbol2 = signal['symbol1'], signal['symbol2']
            
            # Validate data availability
            if symbol1 not in data or symbol2 not in data:
                logger.info(f"Skipping {pair_key}: Missing data")
                continue
            
            # Validate signal
            if not self.strategy.validate_signal(signal):
                logger.info(f"Skipping {pair_key}: Invalid signal")
                continue
            
            # Get current prices
            price1 = signal['price1']
            price2 = signal['price2']
            
            # Calculate position sizes for both symbols
            size1 = self.strategy.calculate_position_size(symbol1, signal, current_capital/2)  # Split capital between pairs
            size2 = self.strategy.calculate_position_size(symbol2, signal, current_capital/2)
            
            if size1 == 0 or size2 == 0:
                logger.info(f"Skipping {pair_key}: Zero position size")
                continue
            
            # Calculate total cost including commission
            total_cost = (size1 * price1 + size2 * price2) * (1 + self.commission)
            
            if total_cost > current_capital:
                logger.info(f"Skipping {pair_key}: Insufficient capital (need {total_cost:.2f}, have {current_capital:.2f})")
                continue
            
            # Calculate stop loss and take profit levels
            stop_loss_distance = signal['std'] * 2  # 2 standard deviations
            take_profit_distance = signal['std'] * 3  # 3 standard deviations
            
            if signal['action'] == 'BUY':
                # Long symbol1, short symbol2
                # Close existing positions if any
                for symbol, pos_size in [(symbol1, size1), (symbol2, -size2)]:
                    current_position = self.positions.get(symbol, {}).get('size', 0)
                    if current_position != 0:
                        current_price = data[symbol].loc[timestamp, 'close']
                        pnl = (current_price - self.positions[symbol]['entry_price']) * current_position
                        if self.positions[symbol]['side'] == 'SELL':
                            pnl = -pnl
                        current_capital += pnl - (abs(current_position) * current_price * self.commission)
                        self._record_trade(symbol, 'CLOSE', current_position, current_price, pnl, timestamp)
                        logger.info(f"Closed position for {symbol}: PnL = {pnl:.2f}")
                
                # Open new positions
                self.positions[symbol1] = {
                    'size': size1,
                    'side': 'BUY',
                    'entry_price': price1,
                    'stop_loss': price1 - stop_loss_distance,
                    'take_profit': price1 + take_profit_distance
                }
                self._record_trade(symbol1, 'BUY', size1, price1, 0, timestamp)
                
                self.positions[symbol2] = {
                    'size': -size2,
                    'side': 'SELL',
                    'entry_price': price2,
                    'stop_loss': price2 + stop_loss_distance,
                    'take_profit': price2 - take_profit_distance
                }
                self._record_trade(symbol2, 'SELL', size2, price2, 0, timestamp)
                
                logger.info(f"Opened long {size1:.6f} {symbol1} @ {price1:.2f}")
                logger.info(f"Opened short {size2:.6f} {symbol2} @ {price2:.2f}")
                
            elif signal['action'] == 'SELL':
                # Short symbol1, long symbol2
                # Close existing positions if any
                for symbol, pos_size in [(symbol1, -size1), (symbol2, size2)]:
                    current_position = self.positions.get(symbol, {}).get('size', 0)
                    if current_position != 0:
                        current_price = data[symbol].loc[timestamp, 'close']
                        pnl = (current_price - self.positions[symbol]['entry_price']) * current_position
                        if self.positions[symbol]['side'] == 'SELL':
                            pnl = -pnl
                        current_capital += pnl - (abs(current_position) * current_price * self.commission)
                        self._record_trade(symbol, 'CLOSE', current_position, current_price, pnl, timestamp)
                        logger.info(f"Closed position for {symbol}: PnL = {pnl:.2f}")
                
                # Open new positions
                self.positions[symbol1] = {
                    'size': -size1,
                    'side': 'SELL',
                    'entry_price': price1,
                    'stop_loss': price1 + stop_loss_distance,
                    'take_profit': price1 - take_profit_distance
                }
                self._record_trade(symbol1, 'SELL', size1, price1, 0, timestamp)
                
                self.positions[symbol2] = {
                    'size': size2,
                    'side': 'BUY',
                    'entry_price': price2,
                    'stop_loss': price2 - stop_loss_distance,
                    'take_profit': price2 + take_profit_distance
                }
                self._record_trade(symbol2, 'BUY', size2, price2, 0, timestamp)
                
                logger.info(f"Opened short {size1:.6f} {symbol1} @ {price1:.2f}")
                logger.info(f"Opened long {size2:.6f} {symbol2} @ {price2:.2f}")
            
            # Update capital
            current_capital -= total_cost
            self.equity_curve[-1] = current_capital
    
    def _update_equity_curve(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Update equity curve based on current positions and prices.
        
        Args:
            data (Dict[str, pd.DataFrame]): Historical data
        """
        current_capital = self.equity_curve[-1]
        
        for symbol, position in self.positions.items():
            if position['size'] != 0 and symbol in data:
                current_price = data[symbol].loc[data[symbol].index[-1], 'close']
                current_capital += position['size'] * current_price
        
        self.equity_curve.append(current_capital)
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            Dict: Performance metrics
        """
        if not self.trades:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_pnl': 0.0,
                'return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'trades': [],
                'equity_curve': []
            }
        
        # Calculate basic metrics
        final_capital = self.equity_curve[-1]
        total_pnl = final_capital - self.initial_capital
        returns = (final_capital - self.initial_capital) / self.initial_capital
        
        # Calculate win rate
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # Calculate maximum drawdown
        peak = self.initial_capital
        max_drawdown = 0.0
        for capital in self.equity_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        daily_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'return': returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'trades': self.trades,
            'equity_curve': self.equity_curve
        } 