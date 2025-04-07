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
        try:
            result['period'] = period
            self.results.append(result)
            
            # Calculate and save detailed metrics
            metrics = self._calculate_metrics(result)
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(self.metrics_dir / f"{period}_metrics.csv", index=False)
            logger.info(f"Saved metrics to {period}_metrics.csv")
            
            # Save trades with additional analysis
            trades_df = pd.DataFrame(result['trades'])
            if not trades_df.empty:
                try:
                    # Convert timestamps and sort
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                    trades_df.sort_values('timestamp', inplace=True)
                    
                    # Calculate trade-level metrics
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                    trades_df['cumulative_return'] = (trades_df['cumulative_pnl'] / result['capital'][0]) * 100
                    trades_df['drawdown'] = self._calculate_drawdown(trades_df['cumulative_pnl'])
                    trades_df['drawdown_pct'] = trades_df['drawdown'] * 100
                    
                    # Calculate rolling metrics
                    trades_df['rolling_avg_pnl_10'] = trades_df['pnl'].rolling(window=10).mean()
                    trades_df['rolling_std_pnl_10'] = trades_df['pnl'].rolling(window=10).std()
                    trades_df['rolling_sharpe_10'] = trades_df['rolling_avg_pnl_10'] / trades_df['rolling_std_pnl_10']
                    
                    # Save enhanced trade data
                    trades_df.to_csv(self.trades_dir / f"{period}_trades_enhanced.csv", index=False)
                    logger.info(f"Saved enhanced trade data to {period}_trades_enhanced.csv")
                    
                    # Calculate and save daily aggregates
                    try:
                        daily_df = trades_df.set_index('timestamp').resample('D').agg({
                            'pnl': 'sum',
                            'cumulative_pnl': 'last',
                            'drawdown': 'max'
                        }).reset_index()
                        
                        daily_df.columns = ['date', 'daily_pnl', 'cumulative_pnl', 'max_drawdown']
                        daily_df.to_csv(self.trades_dir / f"{period}_daily.csv", index=False)
                        logger.info(f"Saved daily aggregates to {period}_daily.csv")
                    except Exception as e:
                        logger.error(f"Error calculating daily aggregates: {str(e)}")
                    
                    # Calculate and save trade statistics
                    try:
                        trade_stats = {
                            'period': period,
                            'start_date': trades_df['timestamp'].min(),
                            'end_date': trades_df['timestamp'].max(),
                            'total_trades': len(trades_df) // 2,
                            'winning_trades': (trades_df['pnl'] > 0).sum(),
                            'losing_trades': (trades_df['pnl'] < 0).sum(),
                            'win_rate': (trades_df['pnl'] > 0).mean(),
                            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
                            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean(),
                            'largest_win': trades_df['pnl'].max(),
                            'largest_loss': trades_df['pnl'].min(),
                            'total_pnl': trades_df['pnl'].sum(),
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'max_drawdown': metrics['max_drawdown_pct'],
                            'profit_factor': metrics['profit_factor']
                        }
                        
                        pd.DataFrame([trade_stats]).to_csv(self.trades_dir / f"{period}_statistics.csv", index=False)
                        logger.info(f"Saved trade statistics to {period}_statistics.csv")
                    except Exception as e:
                        logger.error(f"Error calculating trade statistics: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"Error processing trade data: {str(e)}")
                    # Save original trade data as fallback
                    trades_df.to_csv(self.trades_dir / f"{period}_trades.csv", index=False)
            else:
                logger.warning(f"No trades found for {period} period")
                
        except Exception as e:
            logger.error(f"Error in add_result: {str(e)}")
            # Try to save at least the original trades
            if 'trades' in result:
                pd.DataFrame(result['trades']).to_csv(self.trades_dir / f"{period}_trades.csv", index=False)
        
    def _calculate_metrics(self, result: dict) -> dict:
        """Calculate detailed performance metrics."""
        trades_df = pd.DataFrame(result['trades'])
        if trades_df.empty:
            return self._get_empty_metrics(result)
            
        # Convert timestamps
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df.sort_values('timestamp', inplace=True)
        
        # Calculate PnL metrics
        total_pnl = trades_df['pnl'].sum()
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        # Calculate streaks
        trades_df['is_win'] = trades_df['pnl'] > 0
        win_streaks = self._calculate_streaks(trades_df['is_win'])
        loss_streaks = self._calculate_streaks(~trades_df['is_win'])
        
        # Calculate drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        drawdown_series = self._calculate_drawdown(cumulative_pnl)
        
        # Calculate trade durations
        # Group trades by symbol
        grouped = trades_df.groupby('symbol')
        trade_durations = []
        
        for symbol, group in grouped:
            # Reset index for easier pairing
            group = group.reset_index(drop=True)
            # Find open and close trades
            opens = group[group['type'] == 'open']
            closes = group[group['type'] == 'close']
            
            # Calculate durations if we have both opens and closes
            if not opens.empty and not closes.empty:
                for i in range(min(len(opens), len(closes))):
                    duration = closes.iloc[i]['timestamp'] - opens.iloc[i]['timestamp']
                    trade_durations.append(duration.total_seconds() / 3600)  # Convert to hours
        
        avg_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Calculate daily statistics
        trades_df.set_index('timestamp', inplace=True)
        daily_returns = trades_df['pnl'].resample('D').sum()
        
        metrics = {
            'period': result['period'],
            'initial_capital': result['capital'][0],
            'final_capital': result['capital'][-1],
            'total_return_pct': (result['capital'][-1] / result['capital'][0] - 1) * 100,
            'total_pnl': total_pnl,
            'num_trades': len(trades_df) // 2,  # Divide by 2 since each trade has open and close
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
            'avg_trade_return': trades_df['pnl'].mean(),
            'std_trade_return': trades_df['pnl'].std(),
            'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0,
            'max_drawdown_pct': drawdown_series.max() * 100,
            'avg_drawdown_pct': drawdown_series.mean() * 100,
            'max_drawdown_duration': self._calculate_max_drawdown_duration(drawdown_series),
            'avg_trade_duration_hours': avg_duration,
            'max_consecutive_wins': win_streaks.max() if len(win_streaks) > 0 else 0,
            'max_consecutive_losses': loss_streaks.max() if len(loss_streaks) > 0 else 0,
            'avg_consecutive_wins': win_streaks.mean() if len(win_streaks) > 0 else 0,
            'avg_consecutive_losses': loss_streaks.mean() if len(loss_streaks) > 0 else 0,
            'trades_per_day': len(trades_df) / len(daily_returns) if len(daily_returns) > 0 else 0,
            'daily_win_rate': (daily_returns > 0).mean() if len(daily_returns) > 0 else 0,
            'risk_reward_ratio': abs(winning_trades['pnl'].mean() / losing_trades['pnl'].mean()) if len(losing_trades) > 0 and len(winning_trades) > 0 else 0
        }
        return metrics
        
    def _get_empty_metrics(self, result: dict) -> dict:
        """Return empty metrics when no trades are available."""
        return {
            'period': result['period'],
            'initial_capital': result['capital'][0],
            'final_capital': result['capital'][-1],
            'total_return_pct': 0,
            'total_pnl': 0,
            'num_trades': 0,
            'num_winning_trades': 0,
            'num_losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'avg_trade_return': 0,
            'std_trade_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'avg_drawdown_pct': 0,
            'max_drawdown_duration': 0,
            'avg_trade_duration_hours': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_consecutive_wins': 0,
            'avg_consecutive_losses': 0,
            'trades_per_day': 0,
            'daily_win_rate': 0,
            'risk_reward_ratio': 0
        }
        
    def _calculate_streaks(self, series: pd.Series) -> pd.Series:
        """Calculate consecutive True values in a boolean series."""
        # Reset streak counter when False
        streaks = series.groupby((series != series.shift()).cumsum()).cumcount() + 1
        # Only keep streaks for True values
        return streaks[series]
        
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series from cumulative returns."""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdown)
        
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate the longest drawdown duration in days."""
        is_drawdown = drawdown_series > 0
        if not is_drawdown.any():
            return 0
            
        # Group consecutive drawdown periods
        drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
        # Calculate duration for each group
        durations = drawdown_groups[is_drawdown].value_counts()
        return durations.max() if not durations.empty else 0
        
    def analyze_trades(self):
        """Analyze trade results and generate plots."""
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
            
        # Create plots directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot cumulative PnL
        plt.figure(figsize=(15, 7))
        for period, df in trades.items():
            if 'timestamp' in df.columns and 'cumulative_pnl' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                plt.plot(df['timestamp'], df['cumulative_pnl'], label=f"{period.upper()} Cumulative PnL")
        plt.title('Cumulative PnL Over Time')
        plt.xlabel('Time')
        plt.ylabel('Cumulative PnL')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'cumulative_pnl.png')
        plt.close()
        
        # Plot drawdown
        plt.figure(figsize=(15, 7))
        for period, df in trades.items():
            if 'timestamp' in df.columns and 'drawdown' in df.columns:
                plt.plot(df['timestamp'], df['drawdown'] * 100, label=f"{period.upper()} Drawdown")
        plt.title('Drawdown Over Time')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'drawdown.png')
        plt.close()
        
        # Plot daily returns distribution
        plt.figure(figsize=(15, 7))
        for period, df in trades.items():
            if 'timestamp' in df.columns and 'pnl' in df.columns:
                daily_returns = df.set_index('timestamp')['pnl'].resample('D').sum()
                plt.hist(daily_returns, bins=50, alpha=0.5, label=f"{period.upper()} Daily Returns")
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'daily_returns_dist.png')
        plt.close()
        
        # Plot trade duration distribution
        plt.figure(figsize=(15, 7))
        for period, df in trades.items():
            if 'trade_duration' in df.columns:
                durations_hours = pd.to_timedelta(df['trade_duration']).dt.total_seconds() / 3600
                plt.hist(durations_hours, bins=50, alpha=0.5, label=f"{period.upper()} Trade Durations")
        plt.title('Trade Duration Distribution')
        plt.xlabel('Duration (hours)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'trade_duration_dist.png')
        plt.close()
        
    def save_summary(self):
        """Save a summary of all results."""
        summary = []
        for result in self.results:
            metrics = self._calculate_metrics(result)
            summary.append(metrics)
            
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.results_dir / "summary.csv", index=False)
        
        # Print detailed summary to console
        logger.info("\nBacktest Results Summary:")
        logger.info("=" * 80)
        for metrics in summary:
            logger.info(f"\n{metrics['period'].upper()} Period Performance Metrics:")
            logger.info("-" * 40)
            logger.info(f"Returns and Capital:")
            logger.info(f"  Initial Capital: ${metrics['initial_capital']:,.2f}")
            logger.info(f"  Final Capital: ${metrics['final_capital']:,.2f}")
            logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
            logger.info(f"  Total PnL: ${metrics['total_pnl']:,.2f}")
            
            logger.info(f"\nTrade Statistics:")
            logger.info(f"  Number of Trades: {metrics['num_trades']}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"  Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
            
            logger.info(f"\nProfitability Metrics:")
            logger.info(f"  Average Win: ${metrics['avg_win']:,.2f}")
            logger.info(f"  Average Loss: ${metrics['avg_loss']:,.2f}")
            logger.info(f"  Largest Win: ${metrics['largest_win']:,.2f}")
            logger.info(f"  Largest Loss: ${metrics['largest_loss']:,.2f}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            
            logger.info(f"\nDrawdown Analysis:")
            logger.info(f"  Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            logger.info(f"  Average Drawdown: {metrics['avg_drawdown_pct']:.2f}%")
            logger.info(f"  Max Drawdown Duration: {metrics['max_drawdown_duration']} days")
            
            logger.info(f"\nTrade Patterns:")
            logger.info(f"  Average Trade Duration: {metrics['avg_trade_duration_hours']:.2f} hours")
            logger.info(f"  Max Consecutive Wins: {metrics['max_consecutive_wins']}")
            logger.info(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")
            logger.info(f"  Trades per Day: {metrics['trades_per_day']:.2f}")
            logger.info(f"  Daily Win Rate: {metrics['daily_win_rate']:.2%}")
            
            logger.info("\n" + "=" * 80)

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
                     take_profit: float = 0.03,
                     results_manager: BacktestResults = None) -> dict:
    """
    Backtest the statistical arbitrage strategy.
    
    Args:
        data: Dictionary of DataFrames for each symbol
        strategy_version: Strategy version to use ('v1' or 'v2')
        initial_capital: Initial capital for backtesting
        max_position_size: Maximum position size as fraction of capital
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        results_manager: BacktestResults instance for saving results
        
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
    
    # Initialize tracking data
    tracking_data = []
    
    # Initialize capital
    current_capital = initial_capital
    capital.append(current_capital)
    
    # Run backtest
    for timestamp in common_timestamps:
        timestamps.append(timestamp)
        signals = strategy.process_tick(data, timestamp)
        
        # Initialize tracking for this timestamp
        timestamp_data = {
            'timestamp': timestamp,
            'capital': current_capital,
            'total_pnl': 0.0,
            'active_positions': len(active_pairs),
            'spreads': {},
            'prices': {}
        }
        
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
            
            # Store prices for plotting and tracking
            if symbol1 not in prices:
                prices[symbol1] = []
            if symbol2 not in prices:
                prices[symbol2] = []
            price1 = data[symbol1].loc[timestamp, 'close']
            price2 = data[symbol2].loc[timestamp, 'close']
            prices[symbol1].append(price1)
            prices[symbol2].append(price2)
            
            # Store prices in tracking data
            timestamp_data['prices'][f"{symbol1}_price"] = price1
            timestamp_data['prices'][f"{symbol2}_price"] = price2
            
            # Store spreads for plotting and tracking
            if pair_key not in spreads:
                spreads[pair_key] = []
            spread = pair_signal[symbol1]['spread'] if 'spread' in pair_signal[symbol1] else 0.0
            spreads[pair_key].append(spread)
            timestamp_data['spreads'][f"{symbol1}_{symbol2}_spread"] = spread
            
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
                    
                    # Update tracking data
                    timestamp_data['total_pnl'] += total_pnl
                    timestamp_data['trades'] = {
                        'type': 'close',
                        'pair': f"{symbol1}_{symbol2}",
                        'pnl1': pnl1,
                        'pnl2': pnl2,
                        'total_pnl': total_pnl
                    }
                    
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
                        'spread': spread
                    })
                    trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol2,
                        'type': 'close',
                        'price': price2,
                        'size': size2,
                        'pnl': pnl2,
                        'spread': spread
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
                
                # Update tracking data
                timestamp_data['trades'] = {
                    'type': 'open',
                    'pair': f"{symbol1}_{symbol2}",
                    'direction1': pair_signal[symbol1]['type'],
                    'direction2': pair_signal[symbol2]['type'],
                    'size1': size1,
                    'size2': size2
                }
                
                # Record trades
                trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol1,
                    'type': 'open',
                    'direction': pair_signal[symbol1]['type'],
                    'price': price1,
                    'size': size1,
                    'pnl': 0.0,
                    'spread': spread
                })
                trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol2,
                    'type': 'open',
                    'direction': pair_signal[symbol2]['type'],
                    'price': price2,
                    'size': size2,
                    'pnl': 0.0,
                    'spread': spread
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
        
        # Add tracking data for this timestamp
        tracking_data.append(timestamp_data)
        
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
        'exit_signals': exit_signals,
        'tracking_data': tracking_data
    }
    
    if results['returns']:
        returns_series = pd.Series(results['returns'])
        results['sharpe_ratio'] = returns_series.mean() / returns_series.std() * np.sqrt(252)
        results['max_drawdown'] = (returns_series.cumsum().expanding().max() - 
                                 returns_series.cumsum()).max()
                                 
    # Save tracking data and create plots if results_manager is provided
    if results_manager:
        # Save tracking data to CSV
        tracking_df = pd.DataFrame(tracking_data)
        tracking_df.to_csv(results_manager.metrics_dir / 'tracking.csv', index=False)
        
        # Create plots from tracking data
        plt.figure(figsize=(12, 6))
        plt.plot(tracking_df['timestamp'], tracking_df['capital'], label='Capital')
        plt.title('Capital Curve')
        plt.xlabel('Time')
        plt.ylabel('Capital')
        plt.legend()
        plt.grid(True)
        plt.savefig(results_manager.plots_dir / 'capital_curve.png')
        plt.close()
        
        # Plot PnL
        plt.figure(figsize=(12, 6))
        plt.plot(tracking_df['timestamp'], tracking_df['total_pnl'].cumsum(), label='Cumulative PnL')
        plt.title('Cumulative PnL')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        plt.savefig(results_manager.plots_dir / 'cumulative_pnl.png')
        plt.close()
        
        # Plot active positions
        plt.figure(figsize=(12, 6))
        plt.plot(tracking_df['timestamp'], tracking_df['active_positions'], label='Active Positions')
        plt.title('Active Positions Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Positions')
        plt.legend()
        plt.grid(True)
        plt.savefig(results_manager.plots_dir / 'active_positions.png')
        plt.close()
        
        # Plot spreads for each pair
        spread_columns = [col for col in tracking_df.columns if 'spread' in col]
        if spread_columns:
            plt.figure(figsize=(12, 6))
            for col in spread_columns:
                # Convert to numeric and handle NaN values
                spread_data = pd.to_numeric(tracking_df[col], errors='coerce')
                if not spread_data.isna().all():  # Only plot if there's valid data
                    plt.plot(tracking_df['timestamp'], spread_data, label=col.replace('_spread', ''))
            plt.title('Spreads Over Time')
            plt.xlabel('Time')
            plt.ylabel('Spread')
            plt.legend()
            plt.grid(True)
            plt.savefig(results_manager.plots_dir / 'spreads.png')
            plt.close()
            
        # Plot prices for each symbol
        price_columns = [col for col in tracking_df.columns if 'price' in col]
        if price_columns:
            plt.figure(figsize=(12, 6))
            for col in price_columns:
                # Convert to numeric and handle NaN values
                price_data = pd.to_numeric(tracking_df[col], errors='coerce')
                if not price_data.isna().all():  # Only plot if there's valid data
                    plt.plot(tracking_df['timestamp'], price_data, label=col.replace('_price', ''))
            plt.title('Prices Over Time')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.savefig(results_manager.plots_dir / 'prices.png')
            plt.close()
    
    return results

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
        # 'test_duration_hours': 24
        'test_duration_hours': None  # Remove time limit for complete test
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
            take_profit=config['take_profit'],
            results_manager=results_manager
        )
        
        results_manager.add_result(result, period)
        
    # Save and display results
    results_manager.save_summary()
    # results_manager.plot_results()

if __name__ == "__main__":
    main() 