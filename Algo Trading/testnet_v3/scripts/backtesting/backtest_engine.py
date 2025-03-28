import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from pathlib import Path

class BacktestEngine:
    def __init__(self, strategy, initial_capital: float = 10000.0):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: Trading strategy instance
            initial_capital: Starting capital for the backtest
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}  # Current positions for each symbol
        self.trades: List[Dict] = []  # List of all trades
        self.logger = logging.getLogger(__name__)
        
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run the backtest on historical data.
        
        Args:
            data: Dictionary of DataFrames with historical data for each symbol
            
        Returns:
            Dictionary containing backtest results and metrics
        """
        self.logger.info("Starting backtest...")
        
        # Initialize results tracking
        results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }
        
        # Get common timestamps from all symbols
        all_timestamps = set(next(iter(data.values())).index)
        for df in data.values():
            all_timestamps = all_timestamps.intersection(df.index)
        timestamps = sorted(list(all_timestamps))
        
        if len(timestamps) == 0:
            self.logger.error("No common timestamps found across symbols")
            return results
            
        self.logger.info(f"Running backtest on {len(timestamps)} timestamps")
        
        # Process each timestamp
        for i, timestamp in enumerate(timestamps):
            # Get historical data up to current timestamp for each symbol
            current_data = {}
            for symbol, df in data.items():
                # Get data up to current timestamp
                symbol_data = df[df.index <= timestamp].copy()
                if not symbol_data.empty:
                    current_data[symbol] = symbol_data
            
            if not current_data:
                self.logger.warning(f"No data available at timestamp {timestamp}")
                continue
                
            # Get trading signals
            try:
                signals = self.strategy.get_signals(current_data)
                
                # Process signals and update positions
                for symbol, signal in signals.items():
                    if signal['side'] != 'NONE':
                        self._execute_trade(symbol, signal, timestamp)
                
                # Update equity curve
                current_equity = self._calculate_current_equity(current_data, timestamp)
                results['equity_curve'].append({
                    'timestamp': timestamp,
                    'equity': current_equity
                })
                
                # Log progress every 100 timestamps
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(timestamps)} timestamps")
                    
            except Exception as e:
                self.logger.error(f"Error processing timestamp {timestamp}: {str(e)}")
                continue
        
        # Calculate final metrics
        if results['equity_curve']:
            results['metrics'] = self._calculate_metrics(results['equity_curve'])
            self.logger.info(f"Completed backtest with {len(self.trades)} trades")
        else:
            self.logger.warning("No equity curve data generated during backtest")
            results['metrics'] = {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0,
                'final_equity': self.initial_capital
            }
        
        return results
    
    def _execute_trade(self, symbol: str, signal: Dict, timestamp: datetime):
        """
        Execute a trade based on the signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            timestamp: Current timestamp
        """
        price = signal['price']
        quantity = signal['quantity']
        side = signal['side']
        
        # Calculate trade value
        trade_value = price * quantity
        
        # Update position
        if side == 'BUY':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            self.current_capital -= trade_value
        else:  # SELL
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            self.current_capital += trade_value
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'value': trade_value
        })
    
    def _calculate_current_equity(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> float:
        """
        Calculate current portfolio equity.
        
        Args:
            data: Historical data
            timestamp: Current timestamp
            
        Returns:
            Current portfolio value
        """
        equity = self.current_capital
        
        for symbol, position in self.positions.items():
            if position != 0:
                current_price = data[symbol].loc[timestamp, 'Close']
                equity += position * current_price
        
        return equity
    
    def _calculate_metrics(self, equity_curve: List[Dict]) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: List of equity values over time
            
        Returns:
            Dictionary of performance metrics
        """
        equity_values = [point['equity'] for point in equity_curve]
        
        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Calculate metrics
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate trade metrics
        winning_trades = [t for t in self.trades if t['value'] > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_equity': equity_values[-1]
        } 