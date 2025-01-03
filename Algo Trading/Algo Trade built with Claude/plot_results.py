import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
import os
from trading_types import TradeRecord
def plot_backtest_results(equity_curve: pd.DataFrame, trade_history: List[TradeRecord], 
                         initial_capital: float, benchmark_equity: pd.DataFrame = None):
    """
    Plot backtest results including equity curve and benchmark comparison
    
    Args:
        equity_curve: DataFrame with DatetimeIndex and 'equity' column
        trade_history: List of TradeRecord dictionaries
        initial_capital: Initial portfolio value
        benchmark_equity: DataFrame with DatetimeIndex and 'equity' column
    """
    # Set style
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot equity curve
    plt.plot(equity_curve.index, equity_curve['equity'], 
            label='LSTM Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_equity is not None:
        plt.plot(benchmark_equity.index, benchmark_equity['equity'], 
                label='Equal Weight Benchmark', linewidth=2)
    
    plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    
    plt.title('Portfolio Performance')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Add summary statistics as text
    total_return = float((equity_curve['equity'].iloc[-1] / initial_capital - 1) * 100)
    
    if benchmark_equity is not None:
        bench_return = float((benchmark_equity['equity'].iloc[-1] / initial_capital - 1) * 100)
        stats_text = (f'LSTM Return: {total_return:.2f}%\n'
                     f'Benchmark Return: {bench_return:.2f}%')
    else:
        stats_text = f'Total Return: {total_return:.2f}%'
    
    plt.figtext(0.15, 0.15, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_portfolio_weights(equity_df: pd.DataFrame, trades_list: list, symbols: List[str]) -> plt.Figure:
    """Plot portfolio weights over time"""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Calculate weights over time
    weights_df = pd.DataFrame(index=equity_df.index)
    for symbol in symbols:
        weights_df[symbol] = 0.0  # Initialize with zeros
        
    # Fill in weights from trade history
    for trade in trades_list:
        timestamp = trade['timestamp']
        symbol = trade['symbol']
        position = trade['size']
        weights_df.loc[timestamp, symbol] = position
        
    # Forward fill weights
    weights_df = weights_df.ffill()
    
    # Normalize weights
    row_sums = weights_df.abs().sum(axis=1)
    row_sums = row_sums.where(row_sums != 0, 1)  # Avoid division by zero
    weights_df = weights_df.div(row_sums, axis=0)
    
    # Split into long and short positions
    long_weights = weights_df.copy()
    short_weights = weights_df.copy()
    
    long_weights[long_weights < 0] = 0
    short_weights[short_weights > 0] = 0
    short_weights = short_weights.abs()  # Make short weights positive for plotting
    
    # Plot weights
    ax = plt.gca()
    
    # Plot long positions
    if not long_weights.empty and (long_weights > 0).any().any():
        long_weights.plot(kind='area', stacked=True, ax=ax, colormap='Greens')
    
    # Plot short positions
    if not short_weights.empty and (short_weights > 0).any().any():
        short_weights.plot(kind='area', stacked=True, ax=ax, colormap='Reds')
    
    plt.title('Portfolio Weights Over Time')
    plt.ylabel('Weight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    # Save data
    weights_df.to_csv('portfolio_weights.csv')
    
    plt.tight_layout()
    return fig

def save_backtest_results(mv_stats: pd.DataFrame, bench_stats: Dict, 
                        lstm_metrics: Dict, mv_equity: pd.DataFrame,
                        bench_equity: pd.DataFrame, mv_trades: List[Dict],
                        symbols: List[str], output_dir: str = '.') -> None:
    """
    Save backtest results to CSV files
    
    Args:
        mv_stats: Market neutral strategy statistics
        bench_stats: Benchmark strategy statistics
        lstm_metrics: LSTM model metrics
        mv_equity: Market neutral strategy equity curve
        bench_equity: Benchmark strategy equity curve
        mv_trades: Market neutral strategy trade history
        symbols: List of trading symbols
        output_dir: Directory to save results (default: current directory)
    """
    try:
        # Ensure mv_stats is a DataFrame with required columns
        if isinstance(mv_stats, pd.DataFrame):
            mv_stats_dict = {
                'Total Return': mv_stats.get('total_return', 0.0),
                'Sharpe Ratio': mv_stats.get('sharpe_ratio', 0.0),
                'Max Drawdown': mv_stats.get('max_drawdown', 0.0),
                'Win Rate': mv_stats.get('win_rate', 0.0),
                'Profit Factor': mv_stats.get('profit_factor', 0.0),
                'Total Trades': mv_stats.get('total_trades', 0)
            }
        else:
            mv_stats_dict = {
                'Total Return': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Profit Factor': 0.0,
                'Total Trades': 0
            }

        # Create performance metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': [
                'Total Return (%)',
                'Sharpe Ratio',
                'Max Drawdown (%)',
                'Win Rate (%)',
                'Profit Factor',
                'Total Trades',
                'LSTM Test MAE'
            ],
            'Market Neutral': [
                mv_stats_dict['Total Return'],
                mv_stats_dict['Sharpe Ratio'],
                mv_stats_dict['Max Drawdown'],
                mv_stats_dict['Win Rate'] * 100,
                mv_stats_dict['Profit Factor'],
                mv_stats_dict['Total Trades'],
                lstm_metrics.get('test_mae', 0.0)
            ],
            'Equal Weight': [
                bench_stats.get('total_return', 0.0),
                bench_stats.get('sharpe_ratio', 0.0),
                bench_stats.get('max_drawdown', 0.0),
                bench_stats.get('win_rate', 0.0) * 100,
                bench_stats.get('profit_factor', 0.0),
                bench_stats.get('total_trades', 0),
                0.0
            ]
        })
        
        # Save metrics
        metrics_df.to_csv(os.path.join(output_dir, 'backtest_metrics.csv'), index=False)
        
        # Save equity curves
        mv_equity.to_csv(os.path.join(output_dir, 'regime_equity_curve.csv'), index=True)
        bench_equity.to_csv(os.path.join(output_dir, 'benchmark_equity_curve.csv'), index=True)
        
        # Save trade history
        trades_df = pd.DataFrame(mv_trades)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(output_dir, 'regime_trades.csv'), index=False)
        
        # Calculate and save portfolio weights over time
        if not mv_equity.empty:
            weights_df = pd.DataFrame(index=mv_equity.index)
            for symbol in symbols:
                weights_df[symbol] = 0.0  # Initialize with zeros
                
            # Fill in weights from trade history
            for trade in mv_trades:
                timestamp = trade['timestamp']
                symbol = trade['symbol']
                position = trade['size']
                weights_df.loc[timestamp, symbol] = position
                
            # Forward fill weights
            weights_df = weights_df.ffill()
            
            # Normalize weights
            row_sums = weights_df.abs().sum(axis=1)
            row_sums = row_sums.where(row_sums != 0, 1)  # Avoid division by zero
            weights_df = weights_df.div(row_sums, axis=0)
            
            weights_df.to_csv(os.path.join(output_dir, 'portfolio_weights.csv'), index=True)
            
    except Exception:
        # Create minimal results files
        pd.DataFrame({'Error': ['Error saving results']}).to_csv(os.path.join(output_dir, 'backtest_metrics.csv'), index=False)
        pd.DataFrame({'Error': ['Error saving results']}).to_csv(os.path.join(output_dir, 'regime_equity_curve.csv'), index=False)
        pd.DataFrame({'Error': ['Error saving results']}).to_csv(os.path.join(output_dir, 'benchmark_equity_curve.csv'), index=False)

def plot_btc_regimes(equity_curve: pd.DataFrame, regime_stats: pd.DataFrame) -> plt.Figure:
    """Plot BTC price and detected market regimes"""
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot BTC price
    btc_price = equity_curve['BTCUSDT_price']
    ax1.plot(btc_price.index, btc_price.values, label='BTC Price', color='blue')
    ax1.set_title('BTC Price and Market Regimes')
    ax1.set_ylabel('BTC Price (USDT)')
    ax1.grid(True)
    
    # Plot regimes
    colors = {'Bull': 'green', 'Neutral': 'yellow', 'Bear': 'red'}
    regime_colors = regime_stats['regime'].map(colors)
    
    # Create regime background colors
    for i in range(len(regime_stats)-1):
        start_time = regime_stats.index[i]
        end_time = regime_stats.index[i+1]
        regime = regime_stats['regime'].iloc[i]
        ax1.axvspan(start_time, end_time, alpha=0.2, color=colors[regime])
        ax2.axvspan(start_time, end_time, alpha=0.3, color=colors[regime])
    
    # Plot regime probabilities
    ax2.plot(regime_stats.index, regime_stats['bull_prob'], color='green', label='Bull Probability')
    ax2.plot(regime_stats.index, regime_stats['neutral_prob'], color='yellow', label='Neutral Probability')
    ax2.plot(regime_stats.index, regime_stats['bear_prob'], color='red', label='Bear Probability')
    ax2.set_ylabel('Regime Probabilities')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig