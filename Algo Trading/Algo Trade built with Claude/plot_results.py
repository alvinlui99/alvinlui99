import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict

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
    
    # Handle missing timestamp column
    if 'timestamp' not in trades_df.columns:
        print("Warning: No timestamp column found in trades data")
        return fig
        
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
            positions[symbol] += trade['size']  # Using unified 'size' field
        
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

def save_backtest_results(mv_stats: pd.DataFrame, bench_stats: Dict, 
                        lstm_metrics: Dict, mv_equity: pd.DataFrame,
                        bench_equity: pd.DataFrame, mv_trades: List[Dict],
                        symbols: List[str]) -> None:
    """Save backtest results to CSV files"""
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
        metrics_df.to_csv('backtest_metrics.csv', index=False)
        
        # Save equity curves
        mv_equity.to_csv('regime_equity_curve.csv', index=True)
        bench_equity.to_csv('benchmark_equity_curve.csv', index=True)
        
        # Save trade history
        trades_df = pd.DataFrame(mv_trades)
        if not trades_df.empty:
            trades_df.to_csv('regime_trades.csv', index=False)
        
        # Calculate and save portfolio weights over time
        if not mv_equity.empty:
            weights_df = pd.DataFrame(index=mv_equity.index)
            for symbol in symbols:
                weights_df[symbol] = 0.0  # Initialize with zeros
                
            # Fill in weights from trade history
            for trade in mv_trades:
                timestamp = trade['timestamp']
                symbol = trade['symbol']
                position = trade['position']
                weights_df.loc[timestamp, symbol] = position
                
            # Forward fill weights
            weights_df = weights_df.ffill()
            
            # Normalize weights
            row_sums = weights_df.abs().sum(axis=1)
            row_sums = row_sums.where(row_sums != 0, 1)  # Avoid division by zero
            weights_df = weights_df.div(row_sums, axis=0)
            
            weights_df.to_csv('portfolio_weights.csv', index=True)
            
    except Exception as e:
        print(f"Error saving backtest results: {str(e)}")
        # Create minimal results files
        pd.DataFrame({'Error': ['Error saving results']}).to_csv('backtest_metrics.csv', index=False)
        pd.DataFrame({'Error': ['Error saving results']}).to_csv('regime_equity_curve.csv', index=False)
        pd.DataFrame({'Error': ['Error saving results']}).to_csv('benchmark_equity_curve.csv', index=False)

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