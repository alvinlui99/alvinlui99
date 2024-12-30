import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List

def plot_backtest_results(equity_curve: list, trade_history: list, initial_capital: float, 
                         benchmark_equity: list = None):
    """
    Plot backtest results including equity curve and benchmark comparison
    
    Args:
        equity_curve: List of dictionaries containing timestamp and equity values
        trade_history: List of trade dictionaries
        initial_capital: Initial portfolio value
        benchmark_equity: List of dictionaries containing benchmark equity values
    """
    # Set style
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Convert equity curve to dataframe
    equity_df = pd.DataFrame(equity_curve)
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    equity_df.set_index('timestamp', inplace=True)
    
    # Plot equity curve
    plt.plot(equity_df.index, equity_df['equity'], 
            label='LSTM Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_equity is not None:
        bench_df = pd.DataFrame(benchmark_equity)
        bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp'])
        bench_df.set_index('timestamp', inplace=True)
        plt.plot(bench_df.index, bench_df['equity'], 
                label='Equal Weight Benchmark', linewidth=2)
    
    plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    
    plt.title('Portfolio Performance')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Add summary statistics as text
    total_return = (equity_df['equity'].iloc[-1] / initial_capital - 1) * 100
    
    if benchmark_equity is not None:
        bench_return = (bench_df['equity'].iloc[-1] / initial_capital - 1) * 100
        stats_text = (f'LSTM Return: {total_return:.2f}%\n'
                     f'Benchmark Return: {bench_return:.2f}%')
    else:
        stats_text = f'Total Return: {total_return:.2f}%'
    
    plt.figtext(0.15, 0.15, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_portfolio_weights(equity_df: pd.DataFrame, trades_list: list, symbols: List[str]) -> plt.Figure:
    """Plot portfolio weight allocations over time"""
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades_list)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df.sort_values('timestamp', inplace=True)
    
    # Calculate portfolio values and weights at each timestamp
    weights_df = pd.DataFrame(index=trades_df['timestamp'].unique())
    
    # Initialize position tracking
    positions = {symbol: 0.0 for symbol in symbols}
    
    # Calculate cumulative positions and weights
    for timestamp in weights_df.index:
        # Update positions based on trades
        trades_at_time = trades_df[trades_df['timestamp'] == timestamp]
        for _, trade in trades_at_time.iterrows():
            symbol = trade['symbol']
            positions[symbol] += trade['position']
        
        # Calculate total portfolio value
        total_value = 0
        position_values = {}
        
        for symbol in symbols:
            # Get the latest price for this symbol
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            latest_trade = symbol_trades[symbol_trades['timestamp'] <= timestamp]
            if not latest_trade.empty:
                price = latest_trade.iloc[-1]['price']
                value = positions[symbol] * price
                position_values[symbol] = value
                total_value += value
        
        # Calculate weights
        if total_value > 0:
            for symbol in symbols:
                weights_df.loc[timestamp, symbol] = position_values.get(symbol, 0) / total_value
        else:
            for symbol in symbols:
                weights_df.loc[timestamp, symbol] = 0.0
    
    # Fill NaN values with 0
    weights_df = weights_df.fillna(0)
    
    # Plot weights
    ax = fig.add_subplot(111)
    
    # Create stacked area plot
    weights_df.plot(kind='area', stacked=True, ax=ax)
    
    # Customize plot
    ax.set_title('Portfolio Weight Allocation Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight')
    ax.legend(title='Assets', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    # Save data
    weights_df.to_csv('portfolio_weights.csv')
    
    plt.tight_layout()
    return fig

def save_backtest_results(mv_stats: dict, bench_stats: dict, test_metrics: dict,
                         mv_equity: list, bench_equity: list, mv_trades: list, symbols: List[str]) -> None:
    """Save all relevant backtest results"""
    
    # Save performance metrics
    results = {
        'LSTM Strategy': {
            'Total Return': mv_stats['total_return'],
            'Sharpe Ratio': mv_stats['sharpe_ratio'],
            'Max Drawdown': mv_stats['max_drawdown'],
            'Win Rate': mv_stats['win_rate'],
            'Profit Factor': mv_stats['profit_factor'],
            'Total Trades': mv_stats['trades'],
            'Test MAE': test_metrics['test_mae']
        },
        'Benchmark': {
            'Total Return': bench_stats['total_return'],
            'Sharpe Ratio': bench_stats['sharpe_ratio'],
            'Max Drawdown': bench_stats['max_drawdown'],
            'Win Rate': bench_stats['win_rate'],
            'Profit Factor': bench_stats['profit_factor'],
            'Total Trades': bench_stats['trades']
        }
    }
    
    # Convert to DataFrame and save
    pd.DataFrame(results).to_csv('backtest_metrics.csv')
    
    # Save equity curves
    pd.DataFrame(mv_equity).to_csv('lstm_equity_curve.csv')
    pd.DataFrame(bench_equity).to_csv('benchmark_equity_curve.csv')
    
    # Save trade history
    pd.DataFrame(mv_trades).to_csv('lstm_trades.csv')
    
    # Save portfolio weights
    weights_df = pd.DataFrame(columns=symbols)
    trades_df = pd.DataFrame(mv_trades)
    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df.sort_values('timestamp', inplace=True)
        weights_df.to_csv('portfolio_weights.csv')