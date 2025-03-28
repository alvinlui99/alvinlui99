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
            if symbol not in data or not self.strategy.validate_signal(signal):
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
                logger.info(f"Skipping trade: Position size is 0")
                continue
                
            current_position = self.positions.get(symbol, {}).get('size', 0)
            
            # Handle position changes
            if signal['action'] == 'BUY':
                # Close any existing short position first
                if current_position < 0:
                    pnl = (self.positions[symbol]['entry_price'] - current_price) * abs(current_position)
                    current_capital += pnl - (abs(current_position) * current_price * self.commission)
                    self._record_trade(symbol, 'CLOSE', current_position, current_price, pnl, timestamp)
                
                # Open long position
                cost = position_size * current_price * (1 + self.commission)
                logger.info(f"Attempting BUY: Size: {position_size:.3f} BTC, Price: {current_price:.2f}, Cost: {cost:.2f} USDT, Capital: {current_capital:.2f} USDT")
                if cost <= current_capital:
                    self.positions[symbol] = {
                        'size': position_size,
                        'side': 'BUY',
                        'entry_price': current_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit']
                    }
                    current_capital -= cost
                    self._record_trade(symbol, 'BUY', position_size, current_price, -cost, timestamp)
                    # logger.info(f"BUY executed: New capital: {current_capital:.2f} USDT")
                # else:
                    # logger.info(f"BUY rejected: Insufficient capital (Need: {cost:.2f} USDT, Have: {current_capital:.2f} USDT)")
                    
            elif signal['action'] == 'SELL':
                # Close any existing long position first
                if current_position > 0:
                    pnl = (current_price - self.positions[symbol]['entry_price']) * current_position
                    current_capital += pnl - (current_position * current_price * self.commission)
                    self._record_trade(symbol, 'CLOSE', current_position, current_price, pnl, timestamp)
                
                # Open short position
                cost = position_size * current_price * (1 + self.commission)
                logger.info(f"Attempting SELL: Size: {position_size:.3f} BTC, Price: {current_price:.2f}, Cost: {cost:.2f} USDT, Capital: {current_capital:.2f} USDT")
                if cost <= current_capital:
                    self.positions[symbol] = {
                        'size': -position_size,
                        'side': 'SELL',
                        'entry_price': current_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit']
                    }
                    current_capital -= cost
                    self._record_trade(symbol, 'SELL', -position_size, current_price, -cost, timestamp)
                    logger.info(f"SELL executed: New capital: {current_capital:.2f} USDT")
                else:
                    logger.info(f"SELL rejected: Insufficient capital (Need: {cost:.2f} USDT, Have: {current_capital:.2f} USDT)")
    
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
        Calculate performance metrics based on completed trades.
        
        Returns:
            Dict: Performance metrics
        """
        equity_curve = pd.Series(self.equity_curve)
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(returns)) - 1
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min() * 100
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Initialize metrics
        completed_trades = []
        
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # Process trades to pair entries with exits
            for symbol in trades_df['symbol'].unique():
                symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
                
                i = 0
                while i < len(symbol_trades):
                    current_trade = symbol_trades.iloc[i]
                    
                    if current_trade['action'] in ['BUY', 'SELL']:
                        # Look for the closing trade
                        next_trades = symbol_trades.iloc[i+1:]
                        close_trade = None
                        
                        if len(next_trades) > 0:
                            if current_trade['action'] == 'BUY':
                                close_idx = next_trades[next_trades['action'].isin(['CLOSE', 'SELL'])].index.min()
                            else:  # SELL
                                close_idx = next_trades[next_trades['action'].isin(['CLOSE', 'BUY'])].index.min()
                            
                            if pd.notna(close_idx):
                                close_trade = trades_df.loc[close_idx]
                                
                                # Calculate trade metrics
                                entry_price = current_trade['price']
                                exit_price = close_trade['price']
                                position_size = abs(current_trade['size'])
                                
                                if current_trade['action'] == 'BUY':
                                    pnl = (exit_price - entry_price) * position_size
                                    trade_type = 'Long'
                                else:  # SELL
                                    pnl = (entry_price - exit_price) * position_size
                                    trade_type = 'Short'
                                
                                # Account for commission
                                commission = (entry_price + exit_price) * position_size * self.commission
                                net_pnl = pnl - commission
                                
                                completed_trades.append({
                                    'symbol': symbol,
                                    'type': trade_type,
                                    'entry_time': current_trade['timestamp'],
                                    'exit_time': close_trade['timestamp'],
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'size': position_size,
                                    'pnl': net_pnl,
                                    'commission': commission,
                                    'duration': (close_trade['timestamp'] - current_trade['timestamp']).total_seconds() / 3600
                                })
                                
                                # Skip the closing trade
                                i = trades_df.index.get_loc(close_idx) + 1
                                continue
                    
                    i += 1
            
            # Convert completed trades to DataFrame
            completed_df = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()
            
            if not completed_df.empty:
                winning_trades = completed_df[completed_df['pnl'] > 0]
                losing_trades = completed_df[completed_df['pnl'] <= 0]
                
                win_rate = len(winning_trades) / len(completed_df) * 100
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            completed_df = pd.DataFrame()
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(completed_df),
            'total_commission': completed_df['commission'].sum() if not completed_df.empty else 0,
            'equity_curve': equity_curve,
            'trades': completed_trades  # Now returning the paired trades instead of raw trades
        } 