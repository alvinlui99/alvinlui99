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
import csv

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
        self.capital_updates = []  # Store capital updates
        
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
        total_commission = trades_df['commission'].sum() if 'commission' in trades_df.columns else 0
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
            'total_commission': total_commission,
            'net_pnl': total_pnl - total_commission,
            'commission_ratio': total_commission / total_pnl if total_pnl != 0 else 0,
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
            'total_commission': 0,
            'net_pnl': 0,
            'commission_ratio': 0,
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

    def add_capital_update(self, update_data: dict):
        """Add a capital update to the results.
        
        Args:
            update_data: Dictionary containing capital update information
        """
        self.capital_updates.append(update_data)
        
        # Save capital updates to CSV after each update
        capital_updates_df = pd.DataFrame(self.capital_updates)
        capital_updates_df.to_csv(self.results_dir / "capital_updates.csv", index=False)

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
                     commission: float = 0.0004,
                     results_manager: BacktestResults = None) -> dict:
    """
    Run backtest for statistical arbitrage strategy.
    """
    # Create capital tracking file in the results manager's directory
    if results_manager:
        capital_tracking_file = results_manager.results_dir / "capital_tracking.csv"
    else:
        capital_tracking_file = Path("results/capital_tracking.csv")
        capital_tracking_file.parent.mkdir(exist_ok=True)
    
    with open(capital_tracking_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'capital', 'pnl', 'commission', 'trade_type', 'symbol1', 'symbol2', 'price1', 'price2', 'size1', 'size2'])
    
    def write_capital_update(timestamp, capital, pnl, commission, trade_type=None, symbol1=None, symbol2=None, price1=None, price2=None, size1=None, size2=None):
        with open(capital_tracking_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, capital, pnl, commission, trade_type, symbol1, symbol2, price1, price2, size1, size2])
            
        # Also store in results manager if available
        if results_manager:
            trade_data = {
                'timestamp': timestamp,
                'capital': capital,
                'pnl': pnl,
                'commission': commission,
                'trade_type': trade_type,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'price1': price1,
                'price2': price2,
                'size1': size1,
                'size2': size2
            }
            if trade_type == 'summary':
                results_manager.add_capital_update(trade_data)
    
    # Initialize strategy
    if strategy_version == 'v1':
        strategy = StatisticalArbitrageStrategyV1(
            client=None,  # Not needed for backtesting
            pairs=[],  # Will be populated from data
            lookback_periods=100,
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_loss_threshold=stop_loss,
            timeframe='4h',
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            max_leverage=2.0,
            min_confidence=0.6,
            volatility_threshold=0.02,
            coint_pvalue_threshold=0.05
        )
    else:  # v2
        strategy = StatisticalArbitrageStrategyV2(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            base_stop_loss=stop_loss,
            base_take_profit=take_profit,
            min_correlation=0.7,
            volatility_lookback=20
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
        raise ValueError("No common timestamps found across all pairs")
        
    # Convert to sorted list
    common_timestamps = sorted(list(common_timestamps))
    
    # Track active pairs and recorded trades
    active_pairs = {}  # (symbol1, symbol2) -> {'entry_prices': (price1, price2), 'sizes': (size1, size2)}
    recorded_trades = set()  # Track unique trades to prevent duplicates
    
    # Initialize results tracking
    timestamps = []
    capital = [initial_capital]  # Initialize with initial capital
    trades = []
    spreads = {}
    prices = {}
    entry_signals = {}
    exit_signals = {}
    returns = []
    tracking_data = []
    total_commission = 0.0  # Track total commissions for this period only
    
    # Add initial logging
    logger.info(f"\nStarting backtest with initial capital: ${initial_capital:,.2f}")
    logger.info(f"Commission rate: {commission:.4%}")
    logger.info(f"Period start time: {common_timestamps[0]}")
    logger.info(f"Period end time: {common_timestamps[-1]}")
    
    # Run backtest
    for timestamp in common_timestamps:
        timestamps.append(timestamp)
        signals = strategy.process_tick(data, timestamp)
        
        # Initialize timestamp data
        timestamp_data = {
            'capital': capital[-1],
            'total_pnl': 0.0,
            'timestamp_commission': 0.0,  # Rename to be clear this is just current timestamp's commission
            'active_positions': len(active_pairs),
            'spreads': {},
            'prices': {},
            'trades': []
        }

        # Track timestamp's total commission separately
        timestamp_commission = 0.0
        
        # Log start of timestamp
        logger.info(f"\nProcessing timestamp: {timestamp}")
        logger.info(f"Starting capital: ${capital[-1]:,.2f}")
        logger.info(f"Accumulated commission so far: ${total_commission:,.2f}")
        
        # Group signals by pair
        pair_signals = {}
        for symbol, signal in signals.items():
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
                    
                    # Calculate commission
                    commission1 = price1 * size1 * commission
                    commission2 = price2 * size2 * commission
                    trade_commission = commission1 + commission2
                    
                    # Update total commission
                    total_commission += trade_commission
                    timestamp_commission += trade_commission  # Add to timestamp's commission
                    timestamp_data['timestamp_commission'] = timestamp_commission  # Store with clearer name
                    
                    # Update tracking data
                    timestamp_data['total_pnl'] += total_pnl
                    
                    # Write capital update for closing trade
                    write_capital_update(
                        timestamp=timestamp,
                        capital=capital[-1] + total_pnl - trade_commission,
                        pnl=total_pnl,
                        commission=trade_commission,
                        trade_type='close',
                        symbol1=symbol1,
                        symbol2=symbol2,
                        price1=price1,
                        price2=price2,
                        size1=size1,
                        size2=size2
                    )
                    
                    # Log trade details
                    logger.info(f"\nClosing position for {symbol1} and {symbol2}:")
                    logger.info(f"Entry prices: {entry_price1:.2f}, {entry_price2:.2f}")
                    logger.info(f"Exit prices: {price1:.2f}, {price2:.2f}")
                    logger.info(f"Position sizes: {size1:.8f}, {size2:.8f}")
                    logger.info(f"PnL: ${total_pnl:,.2f} (${pnl1:,.2f} + ${pnl2:,.2f})")
                    logger.info(f"Commission: ${trade_commission:,.2f} (${commission1:,.2f} + ${commission2:,.2f})")
                    logger.info(f"Accumulated commission: ${timestamp_commission:,.2f}")
                    
                    # Create trade keys to check for duplicates
                    trade_key1 = (timestamp, symbol1, 'close')
                    trade_key2 = (timestamp, symbol2, 'close')
                    
                    # Only record trades if they haven't been recorded before
                    if trade_key1 not in recorded_trades:
                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol1,
                            'type': 'close',
                            'entry_price': entry_price1,
                            'exit_price': price1,
                            'size': size1,
                            'pnl': pnl1,
                            'commission': trade_commission,  # Store total commission for both sides
                            'spread': spread
                        })
                        recorded_trades.add(trade_key1)
                    
                    if trade_key2 not in recorded_trades:
                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol2,
                            'type': 'close',
                            'entry_price': entry_price2,
                            'exit_price': price2,
                            'size': size2,
                            'pnl': pnl2,
                            'commission': trade_commission,  # Store total commission for both sides
                            'spread': spread
                        })
                        recorded_trades.add(trade_key2)
                    
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
                position_value = pair_signal[symbol1]['size']  # This is now in USDT
                size1 = position_value / price1  # Convert to quantity for symbol1
                size2 = position_value / price2  # Convert to quantity for symbol2
                
                # Round to appropriate decimal places based on symbol
                if '1000' in symbol1:  # For tokens like 1000PEPE, 1000SHIB
                    size1 = round(size1, 0)  # Round to whole numbers
                else:
                    size1 = round(size1, 3)  # Round to 3 decimal places for other tokens
                    
                if '1000' in symbol2:
                    size2 = round(size2, 0)
                else:
                    size2 = round(size2, 3)
                
                # Calculate commission
                commission1 = price1 * size1 * commission
                commission2 = price2 * size2 * commission
                trade_commission = commission1 + commission2
                
                # Update total commission
                total_commission += trade_commission
                timestamp_commission += trade_commission  # Add to timestamp's commission
                timestamp_data['timestamp_commission'] = timestamp_commission  # Store with clearer name
                
                # Write capital update for opening trade
                write_capital_update(
                    timestamp=timestamp,
                    capital=capital[-1] - trade_commission,
                    pnl=0.0,
                    commission=trade_commission,  # Store total commission for both sides
                    trade_type='open',
                    symbol1=symbol1,
                    symbol2=symbol2,
                    price1=price1,
                    price2=price2,
                    size1=size1,
                    size2=size2
                )
                
                # Log trade details
                logger.info(f"\nOpening position for {symbol1} and {symbol2}:")
                logger.info(f"Entry prices: {price1:.2f}, {price2:.2f}")
                logger.info(f"Position sizes: {size1:.8f}, {size2:.8f}")
                logger.info(f"Commission: ${trade_commission:,.2f} (${commission1:,.2f} + ${commission2:,.2f})")
                logger.info(f"Accumulated commission: ${timestamp_commission:,.2f}")
                
                # Create trade keys to check for duplicates
                trade_key1 = (timestamp, symbol1, 'open')
                trade_key2 = (timestamp, symbol2, 'open')
                
                # Only record trades if they haven't been recorded before
                if trade_key1 not in recorded_trades:
                    trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol1,
                        'type': 'open',
                        'direction': pair_signal[symbol1]['type'],
                        'entry_price': price1,
                        'size': size1,
                        'pnl': 0.0,
                        'commission': trade_commission,  # Store total commission for both sides
                        'spread': spread
                    })
                    recorded_trades.add(trade_key1)
                
                if trade_key2 not in recorded_trades:
                    trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol2,
                        'type': 'open',
                        'direction': pair_signal[symbol2]['type'],
                        'entry_price': price2,
                        'size': size2,
                        'pnl': 0.0,
                        'commission': trade_commission,  # Store total commission for both sides
                        'spread': spread
                    })
                    recorded_trades.add(trade_key2)
                
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
        
        # Update capital once per timestamp
        current_timestamp_trades = [trade for trade in trades if trade['timestamp'] == timestamp]  # Get only current timestamp's trades
        current_pnl = sum(trade['pnl'] for trade in current_timestamp_trades)  # Only current timestamp's PnL
        
        # Use timestamp_commission for consistency instead of recalculating
        current_capital = capital[-1] + current_pnl - timestamp_commission  # Use timestamp's commission
        capital.append(current_capital)
        timestamp_data['capital'] = current_capital
        
        # Write summary for timestamp
        write_capital_update(
            timestamp=timestamp,
            capital=current_capital,
            pnl=current_pnl,
            commission=timestamp_commission,  # Use timestamp's total commission
            trade_type='summary'
        )
        
        # Log detailed capital calculation
        logger.info(f"\nDetailed Capital Calculation for {timestamp}:")
        logger.info(f"Previous capital: ${capital[-2]:,.2f}")
        logger.info(f"Current timestamp PnL: ${current_pnl:,.2f}")
        logger.info(f"Current timestamp commission: ${timestamp_commission:,.2f}")  # Use timestamp_commission
        logger.info(f"Calculated capital: ${current_capital:,.2f}")
        
        # Log trade-by-trade breakdown
        logger.info("\nTrade-by-trade breakdown for current timestamp:")
        for trade in current_timestamp_trades:
            logger.info(f"Trade: {trade['symbol']} {trade['type']} {trade['direction'] if 'direction' in trade else ''}")
            logger.info(f"  PnL: ${trade['pnl']:,.2f}")
            logger.info(f"  Commission: ${trade['commission']:,.2f}")
        
        # Log end of timestamp summary
        logger.info(f"\nTimestamp {timestamp} summary:")
        logger.info(f"Total PnL this timestamp: ${current_pnl:,.2f}")
        logger.info(f"Total commission this timestamp: ${timestamp_commission:,.2f}")  # Use timestamp_commission
        logger.info(f"Ending capital: ${current_capital:,.2f}")
        
        # Validate capital calculation
        expected_capital = capital[-2] + current_pnl - timestamp_commission  # Use timestamp_commission
        if abs(current_capital - expected_capital) > 0.01:  # Allow for small floating point differences
            logger.warning(f"Capital calculation mismatch!")
            logger.warning(f"Expected capital: ${expected_capital:,.2f}")
            logger.warning(f"Actual capital: ${current_capital:,.2f}")
            logger.warning(f"Difference: ${abs(current_capital - expected_capital):,.2f}")
            
            # Log detailed breakdown of the discrepancy
            logger.warning("\nDetailed discrepancy breakdown:")
            logger.warning(f"Previous capital: ${capital[-2]:,.2f}")
            logger.warning(f"Current timestamp PnL: ${current_pnl:,.2f}")
            logger.warning(f"Current timestamp commission: ${timestamp_commission:,.2f}")
            logger.warning(f"Expected: ${expected_capital:,.2f}")
            logger.warning(f"Actual: ${current_capital:,.2f}")
        
        # Add tracking data for this timestamp
        tracking_data.append(timestamp_data)
        
    # Calculate performance metrics
    total_pnl = sum(trade['pnl'] for trade in trades)
    total_commission = sum(trade['commission'] for trade in trades)  # Use sum of trade commissions
    
    logger.info("\nBacktest complete - Final Summary:")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")
    logger.info(f"Total PnL: ${total_pnl:,.2f}")
    logger.info(f"Total commission: ${total_commission:,.2f}")
    logger.info(f"Final capital: ${capital[-1]:,.2f}")
    logger.info(f"Number of trades: {len(trades)}")
    
    # Validate final calculations
    expected_final_capital = initial_capital + total_pnl - total_commission
    
    logger.info("\nValidation Summary:")
    logger.info(f"Total PnL from trades: ${total_pnl:,.2f}")
    logger.info(f"Total commission from trades: ${total_commission:,.2f}")
    logger.info(f"Expected final capital: ${expected_final_capital:,.2f}")
    logger.info(f"Actual final capital: ${capital[-1]:,.2f}")
    
    if abs(expected_final_capital - capital[-1]) > 0.01:
        logger.warning("Final capital calculation mismatch!")
        logger.warning(f"Difference: ${abs(expected_final_capital - capital[-1]):,.2f}")
    
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
        'tracking_data': tracking_data,
        'total_commission': total_commission  # Use the correct total commission
    }
    
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
        'timeframe': '4h',
        'data_dir': 'data',
        'initial_capital': 10000.0,
        'max_position_size': 0.1,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        'commission': 0.0004,  # 0.04% commission per trade
        # 'test_duration_hours': 800
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
            commission=config['commission'],
            results_manager=results_manager
        )
        
        results_manager.add_result(result, period)
        
    # Save and display results
    results_manager.save_summary()
    # results_manager.plot_results()

if __name__ == "__main__":
    main() 