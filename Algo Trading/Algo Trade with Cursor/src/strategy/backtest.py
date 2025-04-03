import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self,
                 strategy: BaseStrategy,
                 initial_capital: float = 10000,
                 commission: float = 0.0004):
        """
        Initialize backtesting framework.
        
        Args:
            strategy (BaseStrategy): Trading strategy to test
            initial_capital (float): Initial capital in USDT
            commission (float): Commission rate per trade (default: 0.04%)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data (Dict[str, pd.DataFrame]): Historical data for each trading pair
            
        Returns:
            Dict: Backtest results
        """
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        
        # Ensure all dataframes have the same index
        common_index = None
        for symbol, df in data.items():
            # Calculate indicators for each symbol
            data[symbol] = self.strategy.calculate_indicators(df)
            
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        # Log number of data points
        logger.info(f"Running backtest with {len(common_index)} data points")
        
        # Run backtest for each timestamp
        for i, timestamp in enumerate(common_index):
            # Skip the first few data points to allow indicators to initialize
            if i < 30:
                continue
                
            # Get data for current timestamp
            current_data = {}
            for symbol, df in data.items():
                current_data[symbol] = df.loc[:timestamp]
            
            # Update positions and calculate PnL
            current_capital = self._update_positions(current_data, timestamp)
            
            # Generate signals for current timestamp
            signals = self.strategy.generate_signals(current_data)
            
            # Execute trades based on signals
            self._execute_trades(signals, current_data, timestamp)
            
            # Record equity
            self.equity_curve.append(current_capital)
            
            # Log portfolio status every day (24 periods for 1h timeframe)
            if i % 24 == 0:
                logger.info(f"Portfolio Status at {timestamp}:")
                logger.info(f"  Capital: {current_capital:.2f} USDT")
                for symbol, pos in self.positions.items():
                    if pos['size'] != 0:
                        current_price = current_data[symbol].iloc[-1]['Close']
                        unrealized_pnl = (current_price - pos['entry_price']) * pos['size']
                        if pos['side'] == 'SELL':
                            unrealized_pnl = -unrealized_pnl
                        logger.info(f"  {symbol}: {pos['size']:.3f} @ {pos['entry_price']:.2f} ({pos['side']}) - PnL: {unrealized_pnl:.2f} USDT")
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        return results
    
    def _update_positions(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> float:
        """
        Update positions and calculate PnL.
        
        Args:
            data (Dict[str, pd.DataFrame]): Historical data
            timestamp (datetime): Current timestamp
            
        Returns:
            float: Updated capital
        """
        current_capital = self.equity_curve[-1]
        
        for symbol, position in list(self.positions.items()):
            if position['size'] != 0 and symbol in data:
                # Get the price using the correct column name
                if 'Close' in data[symbol].columns:
                    current_price = data[symbol].loc[timestamp, 'Close']
                else:
                    logger.error(f"Cannot find price column for {symbol}")
                    continue
                
                pnl = (current_price - position['entry_price']) * position['size']
                if position['side'] == 'SELL':
                    pnl = -pnl
                
                # Check stop loss and take profit
                if position['side'] == 'BUY':
                    # Long position
                    if current_price <= position['stop_loss']:
                        # Close position at stop loss
                        current_capital += pnl - (abs(position['size']) * current_price * self.commission)
                        self._record_trade(symbol, 'CLOSE', position['size'], current_price, pnl, timestamp)
                        self.positions[symbol] = {'size': 0, 'side': None, 'entry_price': 0, 'stop_loss': 0, 'take_profit': 0}
                    elif current_price >= position['take_profit']:
                        # Close position at take profit
                        current_capital += pnl - (abs(position['size']) * current_price * self.commission)
                        self._record_trade(symbol, 'CLOSE', position['size'], current_price, pnl, timestamp)
                        self.positions[symbol] = {'size': 0, 'side': None, 'entry_price': 0, 'stop_loss': 0, 'take_profit': 0}
                elif position['side'] == 'SELL':
                    # Short position
                    if current_price >= position['stop_loss']:
                        # Close position at stop loss
                        current_capital += pnl - (abs(position['size']) * current_price * self.commission)
                        self._record_trade(symbol, 'CLOSE', position['size'], current_price, pnl, timestamp)
                        self.positions[symbol] = {'size': 0, 'side': None, 'entry_price': 0, 'stop_loss': 0, 'take_profit': 0}
                    elif current_price <= position['take_profit']:
                        # Close position at take profit
                        current_capital += pnl - (abs(position['size']) * current_price * self.commission)
                        self._record_trade(symbol, 'CLOSE', position['size'], current_price, pnl, timestamp)
                        self.positions[symbol] = {'size': 0, 'side': None, 'entry_price': 0, 'stop_loss': 0, 'take_profit': 0}
        
        return current_capital
    
    def _execute_trades(self, signals: Dict[str, Dict], data: Dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """
        Execute trades based on signals.
        
        Args:
            signals (Dict[str, Dict]): Trading signals
            data (Dict[str, pd.DataFrame]): Historical data
            timestamp (datetime): Current timestamp
        """
        current_capital = self.equity_curve[-1]
        
        for symbol, signal in signals.items():
            logger.info(f"\nProcessing signal for {symbol} at {timestamp}")
            logger.info(f"Signal action: {signal['action']}")
            
            if symbol not in data or not self.strategy.validate_signal(signal):
                logger.info(f"Skipping {symbol}: Invalid signal or missing data")
                continue
                
            # Get the price using the correct column name
            if 'Close' in data[symbol].columns:
                current_price = data[symbol].loc[timestamp, 'Close']
            else:
                logger.error(f"Cannot find price column for {symbol}")
                continue
            
            # Calculate position size
            position_size = self.strategy.calculate_position_size(symbol, signal, current_capital)
            if position_size == 0:
                logger.info(f"Skipping {symbol}: Position size is 0")
                continue
                
            current_position = self.positions.get(symbol, {}).get('size', 0)
            logger.info(f"Current position for {symbol}: {current_position}")
            
            # Handle position changes
            if signal['action'] == 'BUY':
                # Close any existing short position first
                if current_position < 0:
                    pnl = (self.positions[symbol]['entry_price'] - current_price) * abs(current_position)
                    current_capital += pnl - (abs(current_position) * current_price * self.commission)
                    self._record_trade(symbol, 'CLOSE', current_position, current_price, pnl, timestamp)
                    logger.info(f"Closed short position for {symbol}: PnL = {pnl:.2f}")
                
                # Open long position
                cost = position_size * current_price * (1 + self.commission)
                if cost <= current_capital:
                    # Adjust position size to account for commission
                    adjusted_size = position_size / (1 + self.commission)
                    self.positions[symbol] = {
                        'size': adjusted_size,
                        'side': 'BUY',
                        'entry_price': current_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit']
                    }
                    current_capital -= cost
                    self._record_trade(symbol, 'BUY', adjusted_size, current_price, -cost, timestamp)
                    logger.info(f"Opened long position for {symbol}: Size = {adjusted_size:.3f}, Cost = {cost:.2f}")
                else:
                    logger.info(f"Skipping {symbol}: Insufficient capital for trade")
                    
            elif signal['action'] == 'SELL':
                # Close any existing long position first
                if current_position > 0:
                    pnl = (current_price - self.positions[symbol]['entry_price']) * current_position
                    current_capital += pnl - (current_position * current_price * self.commission)
                    self._record_trade(symbol, 'CLOSE', current_position, current_price, pnl, timestamp)
                    logger.info(f"Closed long position for {symbol}: PnL = {pnl:.2f}")
                
                # Open short position
                cost = position_size * current_price * (1 + self.commission)
                if cost <= current_capital:
                    # Adjust position size to account for commission
                    adjusted_size = position_size / (1 + self.commission)
                    self.positions[symbol] = {
                        'size': -adjusted_size,
                        'side': 'SELL',
                        'entry_price': current_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit']
                    }
                    current_capital -= cost
                    self._record_trade(symbol, 'SELL', -adjusted_size, current_price, -cost, timestamp)
                    logger.info(f"Opened short position for {symbol}: Size = {adjusted_size:.3f}, Cost = {cost:.2f}")
                else:
                    logger.info(f"Skipping {symbol}: Insufficient capital for trade")
    
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