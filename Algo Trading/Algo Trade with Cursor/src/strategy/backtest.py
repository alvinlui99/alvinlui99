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
            
            # Log indicator values every 10 periods
            if i % 10 == 0 or i >= len(common_index) - 5:
                for symbol in self.strategy.trading_pairs:
                    if symbol in data:
                        df = data[symbol]
                        current_row = df.loc[timestamp]
                        
                        # Safe access to indicators
                        macd_val = current_row.get('macd', 'N/A')
                        signal_val = current_row.get('signal', 'N/A')
                        rsi_val = current_row.get('rsi', 'N/A')
                        price = current_row.get('Close', current_row.get('close', 'N/A'))
                        
                        # Log values and signal
                        signal_action = signals.get(symbol, {}).get('action', 'NONE')
                        logger.info(f"Time: {timestamp}, Price: {price}, "
                                   f"MACD: {macd_val}, Signal: {signal_val}, "
                                   f"RSI: {rsi_val}, Action: {signal_action}")
                        
                        # Log why no trade if MACD and RSI look good
                        if signal_action == 'HOLD':
                            if isinstance(macd_val, (int, float)) and isinstance(signal_val, (int, float)) and isinstance(rsi_val, (int, float)):
                                if macd_val > signal_val and rsi_val < self.strategy.rsi_oversold:
                                    logger.info(f"Potential BUY signal but no trade: MACD({macd_val:.4f}) > Signal({signal_val:.4f}) and RSI({rsi_val:.2f}) < {self.strategy.rsi_oversold}")
                                elif macd_val < signal_val and rsi_val > self.strategy.rsi_overbought:
                                    logger.info(f"Potential SELL signal but no trade: MACD({macd_val:.4f}) < Signal({signal_val:.4f}) and RSI({rsi_val:.2f}) > {self.strategy.rsi_overbought}")
            
            # Execute trades based on signals
            self._execute_trades(signals, current_data, timestamp)
            
            # Record equity
            self.equity_curve.append(current_capital)
        
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
                    logger.info(f"BUY executed: New capital: {current_capital:.2f} USDT")
                else:
                    logger.info(f"BUY rejected: Insufficient capital (Need: {cost:.2f} USDT, Have: {current_capital:.2f} USDT)")
                    
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
        Calculate performance metrics.
        
        Returns:
            Dict: Performance metrics
        """
        equity_curve = pd.Series(self.equity_curve)
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(returns)) - 1
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min() * 100
        
        # Calculate trade statistics
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean()
            avg_loss = losing_trades['pnl'].mean()
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
        else:
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
            'total_trades': len(self.trades),
            'equity_curve': equity_curve,
            'trades': self.trades
        } 