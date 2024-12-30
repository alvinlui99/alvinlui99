from typing import List, Dict, Callable
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from portfolio import Portfolio
from config import (TradingConfig, DATA_PATH, DATA_TIMEFRAME)

@dataclass
class TradeStats:
    """Store trade statistics and performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

class Backtester:
    def __init__(
        self,
        symbols: List[str],
        strategy: Callable,
        initial_capital: float = 10000,
        commission: float = TradingConfig.COMMISSION_RATE
    ):
        self.symbols = symbols
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = Portfolio(symbols, initial_capital)
        self.trade_history = []
        self.equity_curve = []
        self.price_history = pd.DataFrame()  # Store full price history
        
    def load_data(self, timeframe: str = DATA_TIMEFRAME) -> Dict[str, pd.DataFrame]:
        """Load historical data for all symbols"""
        data = {}
        for symbol in self.symbols:
            df = pd.read_csv(f"{DATA_PATH}/{symbol}.csv")
            df['datetime'] = pd.to_datetime(df['index'])
            df.set_index('datetime', inplace=True)
            data[symbol] = df
        return data
    
    def run(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run backtest and return performance metrics"""
        if start_date:
            self.start_date = pd.Timestamp(start_date)
        if end_date:
            self.end_date = pd.Timestamp(end_date)
        
        # Initialize tracking variables
        self.equity_curve = pd.DataFrame(columns=['timestamp', 'equity'])
        self.trade_history = []
        timestep = 0
        
        # Get historical data
        historical_data = self._load_historical_data()
        
        # Run backtest
        for timestamp, prices in historical_data.items():
            print(f"\n=== Timestep {timestep} ===")
            print(f"Timestamp: {timestamp}")
            print(f"Current Portfolio Value: ${self.portfolio.get_total_value():.2f}")
            print(f"Current Cash: ${self.portfolio.cash:.2f}")
            
            # Update portfolio prices
            for symbol in self.symbols:
                if symbol in prices:
                    self.portfolio.update_price(symbol, prices[symbol])
            
            # Get strategy signals
            signals = self.strategy(self.portfolio, prices, timestep)
            
            if signals:
                print("\nStrategy Signals:")
                for symbol, target_pos in signals.items():
                    current_pos = self.portfolio.portfolio_df.loc[symbol, 'position']
                    if isinstance(current_pos, pd.Series):
                        current_pos = current_pos.iloc[0]
                    print(f"{symbol}: {float(current_pos):.6f} -> {target_pos:.6f}")
            
            # Execute trades
            if signals:
                self._execute_trades(signals, prices, timestamp)
            
            # Track equity
            self.equity_curve = pd.concat([
                self.equity_curve,
                pd.DataFrame({
                    'timestamp': [timestamp],
                    'equity': [self.portfolio.get_total_value()]
                })
            ], ignore_index=True)
            
            timestep += 1
        
        # Convert trade history to DataFrame for analysis
        trades_df = pd.DataFrame(self.trade_history)
        
        # Calculate performance metrics
        returns = self.equity_curve['equity'].pct_change().fillna(0)
        
        metrics = {
            'total_return': self._calculate_total_return(),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'volatility': returns.std() * np.sqrt(TradingConfig.ANNUALIZATION_FACTOR),
            'trades': len(self.trade_history)
        }
        
        # Add trade statistics
        trade_stats = self._calculate_win_rate(trades_df)
        metrics.update(trade_stats)
        
        return metrics
    
    def _execute_trades(self, signals: Dict[str, float], current_prices: Dict[str, dict], timestamp: pd.Timestamp) -> None:
        """Execute trades based on strategy signals"""
        print("\n=== Trade Execution ===")
        print("\nPortfolio DataFrame Before Trades:")
        print(self.portfolio.portfolio_df)
        print(f"Portfolio Value Before Trades: ${self.portfolio.get_total_value(current_prices):.2f}")
        print(f"Cash Before Trades: ${self.portfolio.cash:.2f}")
        
        for symbol, target_position in signals.items():
            current_position = self.portfolio.portfolio_df.loc[symbol, 'position']
            if isinstance(current_position, pd.Series):
                current_position = current_position.iloc[0]
            
            # Convert to float to ensure numeric comparison
            target_position = float(target_position)
            current_position = float(current_position)
            
            if not np.isclose(target_position, current_position, rtol=1e-5):
                print(f"\nProcessing {symbol}:")
                print(f"  Current Position: {current_position:.6f}")
                print(f"  Target Position: {target_position:.6f}")
                
                # Calculate trade size
                trade_size = target_position - current_position
                
                # Get current price
                if symbol not in current_prices:
                    print(f"  Skipping: No price data")
                    continue
                price = float(current_prices[symbol]['markPrice'])
                print(f"  Price: ${price:.2f}")
                
                # Calculate trade value and commission
                trade_value = abs(trade_size * price)
                commission = trade_value * self.commission
                print(f"  Trade Size: {trade_size:.6f}")
                print(f"  Trade Value: ${trade_value:.2f}")
                print(f"  Commission: ${commission:.2f}")
                
                # Check if we have enough cash for the trade
                if trade_size > 0:  # Buying
                    if self.portfolio.cash < (trade_value + commission):
                        print(f"  Skipping: Insufficient cash")
                        continue
                
                # Update position and cash
                self.portfolio.portfolio_df.at[symbol, 'position'] = target_position
                self.portfolio.cash -= (trade_value * np.sign(trade_size) + commission)
                print(f"  New Position: {target_position:.6f}")
                print(f"  Remaining Cash: ${self.portfolio.cash:.2f}")
                
                # Record trade with unified format
                self.trade_history.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'size': trade_size,  # consistent naming for position change
                    'price': price,
                    'commission': commission,
                    'trade_cost': trade_value,
                    'type': 'execution'
                })
        
        print("\nPortfolio DataFrame After Trades:")
        print(self.portfolio.portfolio_df)
        print("\nPortfolio Value After Trades: ${:.2f}".format(self.portfolio.get_total_value(current_prices)))
        print("=== End Trade Execution ===\n")
    
    def _place_trade(self, symbol: str, size: float, price: float, timestamp: pd.Timestamp = None) -> None:
        """Record a trade with commission"""
        trade_cost = abs(size * price)  # Use abs to match _execute_trades format
        commission_cost = trade_cost * self.commission
        
        if self.portfolio.cash - trade_cost - commission_cost < TradingConfig.MIN_CASH:
            return
        
        self.portfolio.cash -= (trade_cost + commission_cost)
        
        # Record trade with unified format
        self.trade_history.append({
            'timestamp': timestamp if timestamp is not None else datetime.now(),
            'symbol': symbol,
            'size': size,  # consistent with _execute_trades
            'price': price,
            'commission': commission_cost,
            'trade_cost': trade_cost,
            'type': 'placement'
        })
        
        self.portfolio.portfolio_df.loc[symbol, 'position'] += size
    
    def calculate_total_equity(self, current_prices: Dict[str, dict]) -> float:
        """Calculate total portfolio value including cash and positions"""
        equity = self.portfolio.cash
        
        for symbol, row in self.portfolio.portfolio_df.iterrows():
            position = row['position']
            price = current_prices[symbol]['markPrice']
            position_value = position * price
            equity += position_value
        
        return equity
    
    def _calculate_stats(self) -> TradeStats:
        """Calculate performance statistics"""
        equity_df = pd.DataFrame(self.equity_curve)
        returns = equity_df['equity'].pct_change().dropna()
        
        stats = TradeStats(
            total_return=(equity_df['equity'].iloc[-1] / self.initial_capital - 1) * 100,
            sharpe_ratio=np.sqrt(TradingConfig.ANNUALIZATION_FACTOR) * returns.mean() / returns.std() if len(returns) > 1 else 0,
            max_drawdown=self._calculate_max_drawdown(equity_df['equity']),
            win_rate=self._calculate_win_rate(),
            profit_factor=self._calculate_profit_factor()
        )
        return stats
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if self.equity_curve.empty:
            return 0.0
        
        # Get equity series
        equity = self.equity_curve['equity']
        
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max * 100
        
        # Get maximum drawdown
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown
    
    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate win rate and other trade statistics"""
        if trades_df.empty:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
        
        # Calculate trade PnL
        trades_df['trade_value'] = trades_df['size'] * trades_df['price']
        trades_df['pnl'] = trades_df.groupby('symbol')['trade_value'].diff()
        
        # Count profitable trades
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        total_trades = len(trades_df)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average win and loss
        winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
        losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']
        
        avg_win = winning_trades.mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 0
        
        # Calculate profit factor
        total_wins = winning_trades.sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades.sum()) if not losing_trades.empty else 0
        profit_factor = total_wins / total_losses if total_losses != 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': total_trades
        }
    
    def _calculate_profit_factor(self) -> float:
        """Calculate ratio of gross profits to gross losses"""
        if not self.trade_history:
            return 0.0
        trades_df = pd.DataFrame(self.trade_history)
        gross_profit = trades_df[trades_df['size'] * trades_df['price'] > 0]['size'].sum()
        gross_loss = abs(trades_df[trades_df['size'] * trades_df['price'] < 0]['size'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _load_historical_data(self) -> Dict[str, Dict]:
        """Load and prepare historical data for backtesting"""
        data = {}
        all_prices = []
        
        for symbol in self.symbols:
            df = pd.read_csv(f"{DATA_PATH}/{symbol}.csv")
            df['datetime'] = pd.to_datetime(df['index'])
            df.set_index('datetime', inplace=True)
            
            # Filter by date range if specified
            if hasattr(self, 'start_date'):
                df = df[df.index >= self.start_date]
            if hasattr(self, 'end_date'):
                df = df[df.index <= self.end_date]
            
            # Store full price history
            price_df = pd.DataFrame(df['Close'])
            price_df.columns = [f'{symbol}_price']
            all_prices.append(price_df)
            
            # Store data in dictionary format for backtesting
            for timestamp in df.index:
                if timestamp not in data:
                    data[timestamp] = {}
                data[timestamp][symbol] = {
                    'symbol': symbol,
                    'markPrice': f"{df.loc[timestamp, 'Close']:.8f}",
                    'indexPrice': f"{df.loc[timestamp, 'Close']:.8f}",  # Using Close as a proxy
                    'estimatedSettlePrice': f"{df.loc[timestamp, 'Close']:.8f}",  # Using Close as a proxy
                    'lastFundingRate': '0.00010000',  # Default value
                    'interestRate': '0.00010000',  # Default value
                    'nextFundingTime': int((timestamp + pd.Timedelta(hours=8)).timestamp() * 1000),  # 8 hours from current time
                    'time': int(timestamp.timestamp() * 1000)
                }
        
        # Combine all price histories
        self.price_history = pd.concat(all_prices, axis=1)
        
        # Update portfolio with price history
        for symbol in self.symbols:
            price_history = self.price_history[f'{symbol}_price'].to_dict()
            self.portfolio.portfolio_df.loc[symbol, 'asset'].set_price_history(price_history)
        
        # Sort by timestamp
        return dict(sorted(data.items()))
    
    def _calculate_total_return(self) -> float:
        """Calculate total return percentage"""
        if self.equity_curve.empty:
            return 0.0
        
        initial_equity = self.initial_capital
        final_equity = self.equity_curve['equity'].iloc[-1]
        
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
        return total_return
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if returns.empty:
            return 0.0
        
        # Calculate annualized metrics
        risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
        
        excess_returns = returns - (risk_free_rate / TradingConfig.ANNUALIZATION_FACTOR)
        
        # Annualize returns and volatility
        annualized_returns = excess_returns.mean() * TradingConfig.ANNUALIZATION_FACTOR
        annualized_vol = returns.std() * np.sqrt(TradingConfig.ANNUALIZATION_FACTOR)
        
        # Handle zero volatility case
        if annualized_vol == 0:
            return 0.0
            
        sharpe_ratio = annualized_returns / annualized_vol
        
        # Handle invalid values
        if not np.isfinite(sharpe_ratio):
            return 0.0
            
        return sharpe_ratio