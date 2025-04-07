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

def load_data_from_csv(symbols: list, timeframe: str, data_dir: str) -> dict:
    """
    Load historical data from CSV files.
    
    Args:
        symbols: List of trading symbols
        timeframe: Data timeframe
        data_dir: Directory containing CSV files
        
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
            
            # Print actual columns for debugging
            logger.info(f"Columns in {latest_file}: {df.columns.tolist()}")
            
            # Convert timestamp to datetime and set as index
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in {latest_file}")
                logger.error(f"Required columns: {required_columns}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                continue
                
            # Convert numeric columns
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Drop rows with NaN values
            df.dropna(inplace=True)
            
            # Sort index
            df.sort_index(inplace=True)
            
            # Validate date range - only check if data is too old
            if df.index[0].year < 2020:
                logger.error(f"Data too old for {symbol}: {df.index[0]} to {df.index[-1]}")
                continue
                
            data[symbol] = df
            logger.info(f"Loaded {len(df)} rows for {symbol} from {latest_file}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
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
    
    # Run backtest
    for timestamp in common_timestamps:
        signals = strategy.process_tick(data, timestamp)
        
        # Execute trades
        for symbol, signal in signals.items():
            if signal['type'] == 'close':
                # Close position
                strategy.positions[symbol] = 0.0
                strategy.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'type': 'close',
                    'price': data[symbol].loc[timestamp, 'close'],
                    'size': signal['size'],
                    'spread': signal.get('spread', 0.0)
                })
            else:
                # Open position
                strategy.positions[symbol] = signal['size']
                strategy.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'type': 'open',
                    'direction': signal['type'],
                    'price': data[symbol].loc[timestamp, 'close'],
                    'size': signal['size'],
                    'spread': signal.get('spread', 0.0)
                })
                
    # Calculate performance metrics
    results = {
        'capital': strategy.capital,
        'positions': strategy.positions,
        'trades': strategy.trades,
        'returns': strategy.returns,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'pair_returns': strategy.pair_returns,
        'pair_positions': strategy.pair_positions,
        'spreads': strategy.spreads,
        'entry_signals': strategy.entry_signals,
        'exit_signals': strategy.exit_signals
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
        
    def plot_results(self):
        """Plot backtest results."""
        for result in self.results:
            # Plot capital curve
            plt.figure(figsize=(12, 6))
            if 'timestamps' in result and 'capital' in result:
                plt.plot(result['timestamps'], result['capital'])
                plt.title('Capital Curve')
                plt.xlabel('Time')
                plt.ylabel('Capital')
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
        'take_profit': 0.03
    }
    
    # Initialize results manager
    results_manager = BacktestResults()
    results_manager.add_config(config)
    
    # Load data
    logger.info("Loading data...")
    data = load_data_from_csv(config['symbols'], config['timeframe'], config['data_dir'])
    
    if not data:
        logger.error("No data loaded. Exiting.")
        return
        
    # Split data
    logger.info("Splitting data...")
    train_data, test_data, val_data = split_data(data)
    
    if not train_data or not test_data or not val_data:
        logger.error("Data splitting failed. Exiting.")
        return
        
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