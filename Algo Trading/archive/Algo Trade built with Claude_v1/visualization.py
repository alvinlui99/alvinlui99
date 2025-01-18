import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

def plot_equity_curves(strategy_equity: pd.Series, 
                      benchmark_equity: Optional[pd.Series] = None,
                      title: str = "Portfolio Performance",
                      figsize: tuple = (12, 6)) -> None:
    """
    Plot equity curves for strategy and benchmark
    
    Args:
        strategy_equity: Series of strategy equity values indexed by datetime
        benchmark_equity: Optional Series of benchmark equity values indexed by datetime
        title: Plot title
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot strategy equity
    plt.plot(strategy_equity.index, 
             strategy_equity.values, 
             label='Strategy', 
             linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_equity is not None:
        plt.plot(benchmark_equity.index, 
                benchmark_equity.values, 
                label='Benchmark', 
                linewidth=2, 
                linestyle='--')
    
    # Calculate drawdown
    strategy_dd = calculate_drawdown(strategy_equity)
    plt.fill_between(strategy_equity.index, 
                    0, 
                    strategy_dd * 100,
                    alpha=0.2, 
                    color='red',
                    label='Drawdown')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

def calculate_drawdown(equity: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve
    
    Args:
        equity: Series of equity values
        
    Returns:
        Series of drawdown values
    """
    rolling_max = equity.expanding().max()
    drawdown = equity / rolling_max - 1
    return drawdown

def plot_performance_metrics(metrics: Dict[str, float],
                           title: str = "Strategy Performance Metrics",
                           figsize: tuple = (10, 6)) -> None:
    """
    Plot key performance metrics as a bar chart
    
    Args:
        metrics: Dictionary of metric names and values
        title: Plot title
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Create bar chart
    plt.bar(range(len(metrics)), 
            list(metrics.values()),
            color='skyblue')
    
    # Customize x-axis labels
    plt.xticks(range(len(metrics)), 
               list(metrics.keys()),
               rotation=45)
    
    plt.title(title)
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(metrics.values()):
        plt.text(i, v, f'{v:.2f}', 
                ha='center', 
                va='bottom')
    
    plt.tight_layout()

def plot_rolling_metrics(equity: pd.Series,
                        window: int = 30,
                        metrics: Optional[list] = None,
                        figsize: tuple = (12, 8)) -> None:
    """
    Plot rolling performance metrics
    
    Args:
        equity: Series of equity values
        window: Rolling window size in days
        metrics: List of metrics to plot (default: ['returns', 'volatility', 'sharpe'])
        figsize: Figure size as (width, height)
    """
    if metrics is None:
        metrics = ['returns', 'volatility', 'sharpe']
    
    # Calculate rolling metrics
    returns = equity.pct_change()
    rolling_returns = returns.rolling(window).mean() * 252  # Annualized
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
    rolling_sharpe = rolling_returns / rolling_vol
    
    plt.figure(figsize=figsize)
    
    if 'returns' in metrics:
        plt.plot(rolling_returns.index,
                rolling_returns.values,
                label=f'{window}d Rolling Returns',
                linewidth=2)
    
    if 'volatility' in metrics:
        plt.plot(rolling_vol.index,
                rolling_vol.values,
                label=f'{window}d Rolling Volatility',
                linewidth=2)
    
    if 'sharpe' in metrics:
        plt.plot(rolling_sharpe.index,
                rolling_sharpe.values,
                label=f'{window}d Rolling Sharpe',
                linewidth=2)
    
    plt.title(f'Rolling Metrics ({window} days)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout() 