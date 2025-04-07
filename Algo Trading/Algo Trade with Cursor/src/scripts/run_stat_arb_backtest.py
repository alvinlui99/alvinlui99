import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.strategy.stat_arb.v1 import StatisticalArbitrageStrategyV1
from src.strategy.stat_arb.v2 import StatisticalArbitrageStrategyV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_from_csv(symbols: list, timeframe: str, data_dir: str, test_duration_hours: int = None) -> dict:
    """
    Load historical data from CSV files.
    
    Args:
        symbols: List of trading symbols
        timeframe: Data timeframe
        data_dir: Directory containing CSV files
        test_duration_hours: Optional duration in hours to limit the test period.
                            If specified, only the most recent N hours of data will be used.
                            This is useful for quick testing and debugging.
                            Example: test_duration_hours=1 will use only the last hour of data.
        
    Returns:
        dict: Dictionary of DataFrames for each symbol
    """
    data = {}
    for symbol in symbols:
        try:
            # Find the most recent CSV file for this symbol and timeframe
            csv_files = [f for f in os.listdir(data_dir) 
                        if f.startswith(f"{symbol}_{timeframe}_") and f.endswith('.csv')]
            if not csv_files:
                logger.warning(f"No CSV files found for {symbol} {timeframe}")
                continue
                
            # Sort by date and get the most recent
            csv_files.sort()
            latest_file = csv_files[-1]
            
            # Load data
            df = pd.read_csv(os.path.join(data_dir, latest_file))
            
            # Convert column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Convert timestamp to datetime and set as index
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            # If test duration is specified, limit the data
            if test_duration_hours is not None:
                end_time = df.index[-1]
                start_time = end_time - pd.Timedelta(hours=test_duration_hours)
                df = df[start_time:end_time]
                logger.info(f"Limited data for {symbol} to last {test_duration_hours} hours")
                logger.info(f"Test period: {start_time} to {end_time}")
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in {latest_file}")
                continue
                
            # Convert numeric columns
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Drop rows with NaN values
            df.dropna(inplace=True)
            
            # Sort index
            df.sort_index(inplace=True)
            
            data[symbol] = df
            logger.info(f"Loaded {len(df)} rows for {symbol} from {latest_file}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            continue
            
    if not data:
        logger.error("No data was successfully loaded. Please check the CSV files and their column names.")
        raise ValueError("No data was successfully loaded. Please check the CSV files and their column names.")
            
    return data

def split_data(data: dict) -> tuple:
    """
    Split data into training, testing, and validation sets.
    
    Args:
        data: Dictionary of DataFrames for each symbol
        
    Returns:
        tuple: (train_data, test_data, val_data)
    """
    # Find common date range
    start_date = max(df.index[0] for df in data.values())
    end_date = min(df.index[-1] for df in data.values())
    
    # Split into three equal periods
    total_days = (end_date - start_date).days
    train_end = start_date + timedelta(days=total_days // 3)
    test_end = train_end + timedelta(days=total_days // 3)
    
    # Split data
    train_data = {}
    test_data = {}
    val_data = {}
    
    for symbol, df in data.items():
        train_data[symbol] = df[start_date:train_end]
        test_data[symbol] = df[train_end:test_end]
        val_data[symbol] = df[test_end:end_date]
        
    return train_data, test_data, val_data

def backtest_strategy(data: dict, 
                     strategy_version: str = 'v1',
                     initial_capital: float = 10000.0,
                     max_position_size: float = 0.1,
                     stop_loss: float = 0.02,
                     take_profit: float = 0.03) -> dict:
    """
    Backtest the statistical arbitrage strategy.
    
    Args:
        data: Dictionary of DataFrames for each symbol
        strategy_version: Strategy version to use ('v1' or 'v2')
        initial_capital: Initial capital for backtesting
        max_position_size: Maximum position size as fraction of capital
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        
    Returns:
        dict: Backtest results
    """
    # Initialize strategy
    if strategy_version == 'v1':
        strategy = StatisticalArbitrageStrategyV1(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    else:
        strategy = StatisticalArbitrageStrategyV2(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            base_stop_loss=stop_loss,
            base_take_profit=take_profit
        )
        
    # Find common timestamps across all pairs
    common_timestamps = None
    for symbol, df in data.items():
        if common_timestamps is None:
            common_timestamps = set(df.index)
        else:
            common_timestamps = common_timestamps.intersection(set(df.index))
            
    if not common_timestamps:
        logger.error("No common timestamps found across all pairs")
        logger.error("Available date ranges:")
        for symbol, df in data.items():
            logger.error(f"{symbol}: {df.index[0]} to {df.index[-1]}")
        raise ValueError("No common timestamps found across all pairs")
        
    # Convert to sorted list
    common_timestamps = sorted(list(common_timestamps))
    logger.info(f"Found {len(common_timestamps)} common timestamps")
    logger.info(f"Date range: {common_timestamps[0]} to {common_timestamps[-1]}")
    
    # Track active pairs
    active_pairs = {}  # (symbol1, symbol2) -> {'entry_prices': (price1, price2), 'sizes': (size1, size2)}
    
    # Initialize results tracking
    timestamps = []
    capital = []
    trades = []
    spreads = {}
    prices = {}
    entry_signals = {}
    exit_signals = {}
    returns = []
    
    # Initialize capital
    current_capital = initial_capital
    capital.append(current_capital)
    
    # Run backtest
    for timestamp in common_timestamps:
        timestamps.append(timestamp)
        signals = strategy.process_tick(data, timestamp)
        
        # Group signals by pair
        pair_signals = {}
        for symbol, signal in signals.items():
            # Find the pair this symbol belongs to
            for pair in strategy.pairs:
                if symbol in pair:
                    other_symbol = pair[0] if pair[1] == symbol else pair[1]
                    if other_symbol in signals:
                        pair_key = tuple(sorted(pair))
                        if pair_key not in pair_signals:
                            pair_signals[pair_key] = {}
                        pair_signals[pair_key][symbol] = signal
                        pair_signals[pair_key][other_symbol] = signals[other_symbol]
                        break
        
        # Execute trades for each pair
        for pair_key, pair_signal in pair_signals.items():
            symbol1, symbol2 = pair_key
            
            # Store prices for plotting
            if symbol1 not in prices:
                prices[symbol1] = []
            if symbol2 not in prices:
                prices[symbol2] = []
            prices[symbol1].append(data[symbol1].loc[timestamp, 'close'])
            prices[symbol2].append(data[symbol2].loc[timestamp, 'close'])
            
            # Store spreads for plotting
            if pair_key not in spreads:
                spreads[pair_key] = []
            spreads[pair_key].append(pair_signal[symbol1].get('spread', 0.0))
            
            if pair_signal[symbol1]['type'] == 'close' and pair_signal[symbol2]['type'] == 'close':
                # Close position
                if pair_key in active_pairs:
                    # Get current prices
                    price1 = data[symbol1].loc[timestamp, 'close']
                    price2 = data[symbol2].loc[timestamp, 'close']
                    
                    # Get entry prices and sizes
                    entry_price1, entry_price2 = active_pairs[pair_key]['entry_prices']
                    size1, size2 = active_pairs[pair_key]['sizes']
                    
                    # Calculate PnL for both sides
                    pnl1 = (price1 - entry_price1) * size1
                    pnl2 = (price2 - entry_price2) * size2
                    total_pnl = pnl1 + pnl2
                    
                    # Calculate return
                    if current_capital > 0:
                        returns.append(total_pnl / current_capital)
                    else:
                        returns.append(0.0)
                    
                    # Update capital
                    current_capital += total_pnl
                    
                    # Record trades
                    trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol1,
                        'type': 'close',
                        'price': price1,
                        'size': size1,
                        'pnl': pnl1,
                        'spread': pair_signal[symbol1].get('spread', 0.0)
                    })
                    trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol2,
                        'type': 'close',
                        'price': price2,
                        'size': size2,
                        'pnl': pnl2,
                        'spread': pair_signal[symbol2].get('spread', 0.0)
                    })
                    
                    # Store exit signals
                    if symbol1 not in exit_signals:
                        exit_signals[symbol1] = []
                    if symbol2 not in exit_signals:
                        exit_signals[symbol2] = []
                    exit_signals[symbol1].append({'timestamp': timestamp, 'price': price1})
                    exit_signals[symbol2].append({'timestamp': timestamp, 'price': price2})
                    
                    # Clear position
                    del active_pairs[pair_key]
            else:
                # Open position
                price1 = data[symbol1].loc[timestamp, 'close']
                price2 = data[symbol2].loc[timestamp, 'close']
                
                # Calculate position sizes
                size1 = pair_signal[symbol1]['size']
                size2 = pair_signal[symbol2]['size']
                
                # Record trades
                trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol1,
                    'type': 'open',
                    'direction': pair_signal[symbol1]['type'],
                    'price': price1,
                    'size': size1,
                    'pnl': 0.0,
                    'spread': pair_signal[symbol1].get('spread', 0.0)
                })
                trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol2,
                    'type': 'open',
                    'direction': pair_signal[symbol2]['type'],
                    'price': price2,
                    'size': size2,
                    'pnl': 0.0,
                    'spread': pair_signal[symbol2].get('spread', 0.0)
                })
                
                # Store entry signals
                if symbol1 not in entry_signals:
                    entry_signals[symbol1] = []
                if symbol2 not in entry_signals:
                    entry_signals[symbol2] = []
                entry_signals[symbol1].append({'timestamp': timestamp, 'price': price1})
                entry_signals[symbol2].append({'timestamp': timestamp, 'price': price2})
                
                # Store position
                active_pairs[pair_key] = {
                    'entry_prices': (price1, price2),
                    'sizes': (size1, size2)
                }
                
        # Update capital curve
        capital.append(current_capital)
        
    # Calculate performance metrics
    results = {
        'timestamps': timestamps,
        'capital': capital,
        'trades': trades,
        'returns': returns,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'spreads': spreads,
        'prices': prices,
        'entry_signals': entry_signals,
        'exit_signals': exit_signals
    }
    
    if results['returns']:
        returns_series = pd.Series(results['returns'])
        results['sharpe_ratio'] = returns_series.mean() / returns_series.std() * np.sqrt(252)
        results['max_drawdown'] = (returns_series.cumsum().expanding().max() - 
                                 returns_series.cumsum()).max()
                                 
    return results

class BacktestResults:
    def __init__(self, run_id: str = None):
        """Initialize backtest results with a unique run ID."""
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.trades_dir = self.results_dir / "trades"
        self.trades_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.config = {}
        
    def add_config(self, config: dict):
        """Add configuration parameters for this backtest run."""
        self.config = config
        with open(self.results_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
            
    def add_result(self, result: dict, period: str):
        """Add a backtest result for a specific period."""
        result['period'] = period
        self.results.append(result)
        
        # Save detailed metrics
        metrics = self._calculate_metrics(result)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.metrics_dir / f"{period}_metrics.csv", index=False)
        
        # Save trades
        trades_df = pd.DataFrame(result['trades'])
        if not trades_df.empty:
            trades_df.to_csv(self.trades_dir / f"{period}_trades.csv", index=False)
            
    def _calculate_metrics(self, result: dict) -> dict:
        """Calculate detailed performance metrics."""
        returns = pd.Series(result['returns'])
        capital = pd.Series(result['capital'])
        
        metrics = {
            'period': result['period'],
            'initial_capital': result['capital'][0],
            'final_capital': result['capital'][-1],
            'total_return': (result['capital'][-1] / result['capital'][0] - 1) * 100,
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'] * 100,
            'num_trades': len(result['trades']),
            'avg_trade_return': returns.mean() * 100 if not returns.empty else 0,
            'std_trade_return': returns.std() * 100 if not returns.empty else 0,
            'win_rate': self._calculate_win_rate(result['trades']),
            'profit_factor': self._calculate_profit_factor(result['trades']),
            'avg_holding_period': self._calculate_avg_holding_period(result['trades']),
            'max_consecutive_wins': self._calculate_max_consecutive_wins(result['trades']),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(result['trades'])
        }
        return metrics
        
    def _calculate_win_rate(self, trades: list) -> float:
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade['type'] == 'close' and trade['size'] > 0)
        return winning_trades / len(trades)
        
    def _calculate_profit_factor(self, trades: list) -> float:
        if not trades:
            return 0.0
        profits = sum(trade['size'] for trade in trades if trade['type'] == 'close' and trade['size'] > 0)
        losses = abs(sum(trade['size'] for trade in trades if trade['type'] == 'close' and trade['size'] < 0))
        return profits / losses if losses != 0 else float('inf')
        
    def _calculate_avg_holding_period(self, trades: list) -> float:
        if not trades:
            return 0.0
        holding_periods = []
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                holding_periods.append((trades[i+1]['timestamp'] - trades[i]['timestamp']).total_seconds() / 3600)
        return np.mean(holding_periods) if holding_periods else 0.0
        
    def _calculate_max_consecutive_wins(self, trades: list) -> int:
        if not trades:
            return 0
        current_streak = 0
        max_streak = 0
        for trade in trades:
            if trade['type'] == 'close' and trade['size'] > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
        
    def _calculate_max_consecutive_losses(self, trades: list) -> int:
        if not trades:
            return 0
        current_streak = 0
        max_streak = 0
        for trade in trades:
            if trade['type'] == 'close' and trade['size'] < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
        
    def analyze_trades(self):
        """Analyze trade results and generate plots."""
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load all trade files
        trades = {}
        for period in ['train', 'test', 'val']:
            trades_file = self.trades_dir / f"{period}_trades.csv"
            if trades_file.exists():
                trades[period] = pd.read_csv(trades_file)
                logger.info(f"Loaded {len(trades[period])} trades from {period} period")
        
        if not trades:
            logger.error("No trade files found")
            return
            
        # Create plots directory if it doesn't exist
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot capital curve
        plt.figure(figsize=(12, 6))
        for period, df in trades.items():
            if 'timestamp' in df.columns and 'capital' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                plt.plot(df['timestamp'], df['capital'], label=period.upper())
        plt.title('Capital Curve')
        plt.xlabel('Time')
        plt.ylabel('Capital')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'capital_curve.png')
        plt.close()
        
        # Plot trade returns distribution
        plt.figure(figsize=(12, 6))
        for period, df in trades.items():
            if 'return' in df.columns:
                plt.hist(df['return'], bins=50, alpha=0.5, label=period.upper())
        plt.title('Trade Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'returns_distribution.png')
        plt.close()
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        for period, df in trades.items():
            if 'capital' in df.columns:
                capital = pd.Series(df['capital'])
                drawdown = (capital.expanding().max() - capital) / capital.expanding().max()
                plt.plot(df['timestamp'], drawdown, label=period.upper())
        plt.title('Drawdown Over Time')
        plt.xlabel('Time')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'drawdown.png')
        plt.close()
        
        # Calculate and print key metrics
        logger.info("\nTrade Analysis Summary:")
        logger.info("=" * 50)
        for period, df in trades.items():
            if 'return' in df.columns and 'capital' in df.columns:
                total_return = (df['capital'].iloc[-1] / df['capital'].iloc[0] - 1) * 100
                sharpe_ratio = df['return'].mean() / df['return'].std() * np.sqrt(252)
                max_drawdown = (df['capital'].expanding().max() - df['capital']) / df['capital'].expanding().max()
                max_drawdown = max_drawdown.max() * 100
                
                logger.info(f"\n{period.upper()} Period:")
                logger.info(f"Total Return: {total_return:.2f}%")
                logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
                logger.info(f"Number of Trades: {len(df)}")
                logger.info(f"Average Return: {df['return'].mean() * 100:.2f}%")
                logger.info(f"Win Rate: {(df['return'] > 0).mean() * 100:.2f}%")
                
    def plot_results(self):
        """Plot backtest results."""
        self.analyze_trades()  # Use the new analysis function
        for result in self.results:
            # Plot capital curve
            plt.figure(figsize=(12, 6))
            if 'timestamps' in result and 'capital' in result:
                plt.plot(result['timestamps'], result['capital'], label='Capital')
                plt.title('Capital Curve')
                plt.xlabel('Time')
                plt.ylabel('Capital')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.results_dir, 'capital_curve.png'))
                plt.close()
            else:
                logger.warning("Missing timestamps or capital data for plotting")
            
            # Plot trade signals
            if 'entry_signals' in result and 'prices' in result:
                for symbol in result['entry_signals']:
                    if isinstance(symbol, tuple):  # Handle pair symbols
                        symbol_str = f"{symbol[0]}-{symbol[1]}"
                    else:
                        symbol_str = symbol
                        
                    plt.figure(figsize=(12, 6))
                    if symbol in result['prices']:
                        plt.plot(result['timestamps'], result['prices'][symbol], label='Price')
                        
                        # Plot entry signals
                        if symbol in result['entry_signals']:
                            entry_times = [signal['timestamp'] for signal in result['entry_signals'][symbol]]
                            entry_prices = [result['prices'][symbol][result['timestamps'].index(t)] for t in entry_times]
                            plt.scatter(entry_times, entry_prices, color='green', label='Entry', marker='^')
                        
                        # Plot exit signals
                        if symbol in result['exit_signals']:
                            exit_times = [signal['timestamp'] for signal in result['exit_signals'][symbol]]
                            exit_prices = [result['prices'][symbol][result['timestamps'].index(t)] for t in exit_times]
                            plt.scatter(exit_times, exit_prices, color='red', label='Exit', marker='v')
                        
                        plt.title(f'Trade Signals for {symbol_str}')
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(os.path.join(self.results_dir, f'trade_signals_{symbol_str}.png'))
                        plt.close()
                    else:
                        logger.warning(f"Missing price data for {symbol_str}")
            else:
                logger.warning("Missing entry_signals or prices data for plotting")
                
            # Plot spreads
            if 'spreads' in result:
                for pair, spread_data in result['spreads'].items():
                    plt.figure(figsize=(12, 6))
                    plt.plot(result['timestamps'], spread_data, label='Spread')
                    plt.title(f'Spread for {pair[0]}-{pair[1]}')
                    plt.xlabel('Time')
                    plt.ylabel('Spread')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(self.results_dir, f'spread_{pair[0]}_{pair[1]}.png'))
                    plt.close()
                    
            # Plot returns
            if 'returns' in result:
                plt.figure(figsize=(12, 6))
                plt.plot(result['timestamps'], result['returns'], label='Returns')
                plt.title('Returns Over Time')
                plt.xlabel('Time')
                plt.ylabel('Returns')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.results_dir, 'returns.png'))
                plt.close()
                
            # Plot drawdown
            if 'capital' in result:
                capital_series = pd.Series(result['capital'])
                drawdown = (capital_series.expanding().max() - capital_series) / capital_series.expanding().max()
                plt.figure(figsize=(12, 6))
                plt.plot(result['timestamps'], drawdown, label='Drawdown')
                plt.title('Drawdown Over Time')
                plt.xlabel('Time')
                plt.ylabel('Drawdown')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.results_dir, 'drawdown.png'))
                plt.close()

    def save_summary(self):
        """Save a summary of all results."""
        summary = []
        for result in self.results:
            metrics = self._calculate_metrics(result)
            summary.append(metrics)
            
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.results_dir / "summary.csv", index=False)
        
        # Print summary to console
        logger.info("\nBacktest Results Summary:")
        logger.info("=" * 50)
        for metrics in summary:
            logger.info(f"\n{metrics['period'].upper()} Period:")
            logger.info(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
            logger.info(f"Final Capital: ${metrics['final_capital']:,.2f}")
            logger.info(f"Total Return: {metrics['total_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            logger.info(f"Number of Trades: {metrics['num_trades']}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"Avg Holding Period: {metrics['avg_holding_period']:.2f} hours")
            logger.info(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")
            logger.info(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration
    config = {
        'strategy_version': 'v2',  # or 'v1' for original strategy
        'symbols': [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'LINKUSDT', 'NEARUSDT', 'WIFUSDT', 'AVAXUSDT', '1000SHIBUSDT',
            'DOGEUSDT', '1000PEPEUSDT', 'WLDUSDT'
        ],
        'timeframe': '15m',
        'data_dir': 'data',
        'initial_capital': 10000.0,
        'max_position_size': 0.1,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        'test_duration_hours': 24  # Increased to 24 hours to ensure enough data for signal generation
    }
    
    # Initialize results manager
    results_manager = BacktestResults()
    results_manager.add_config(config)
    
    # Load data
    logger.info("Loading data...")
    data = load_data_from_csv(
        config['symbols'], 
        config['timeframe'], 
        config['data_dir'],
        test_duration_hours=config['test_duration_hours']
    )
    
    if not data:
        logger.error("No data loaded. Exiting.")
        return
        
    # Log data statistics
    logger.info("\nData Statistics:")
    for symbol, df in data.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"Number of data points: {len(df)}")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Price range: {df['close'].min():.8f} to {df['close'].max():.8f}")
        logger.info(f"Average price: {df['close'].mean():.8f}")
        logger.info(f"Price volatility: {df['close'].std():.8f}")
        
    # Split data
    logger.info("\nSplitting data...")
    train_data, test_data, val_data = split_data(data)
    
    if not train_data or not test_data or not val_data:
        logger.error("Data splitting failed. Exiting.")
        return
        
    # Log split statistics
    logger.info("\nSplit Statistics:")
    for period, period_data in [('train', train_data), ('test', test_data), ('val', val_data)]:
        logger.info(f"\n{period.upper()} Period:")
        for symbol, df in period_data.items():
            logger.info(f"{symbol}: {len(df)} data points")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        
    # Run backtests
    for period, period_data in [('train', train_data), ('test', test_data), ('val', val_data)]:
        logger.info(f"\nRunning {period} period backtest...")
        result = backtest_strategy(
            period_data,
            strategy_version=config['strategy_version'],
            initial_capital=config['initial_capital'],
            max_position_size=config['max_position_size'],
            stop_loss=config['stop_loss'],
            take_profit=config['take_profit']
        )
        results_manager.add_result(result, period)
        
    # Save and display results
    results_manager.plot_results()
    results_manager.save_summary()

if __name__ == "__main__":
    main() 