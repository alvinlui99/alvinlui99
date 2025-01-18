import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_curves(strategy: pd.Series, 
                benchmark: Optional[pd.Series] = None,
                leverage: Optional[pd.Series] = None,
                leverage_component: Optional[pd.DataFrame] = None,
                title: str = "Portfolio Performance",
                figsize: tuple = (12, 6)) -> None:
    """
    Plot equity curves for strategy and benchmark
    
    Args:
        strategy: Series of strategy equity values indexed by datetime
        benchmark: Optional Series of benchmark equity values indexed by datetime
        title: Plot title
        figsize: Figure size as (width, height)
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot strategy and benchmark on primary axis
    ax1.plot(strategy.index, strategy.values, label='Strategy', linewidth=2)
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values, 
                label='Benchmark', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary axis for leverage
    if leverage is not None:
        ax2 = ax1.twinx()  # instantiate second axes sharing the same x-axis
        ax2.plot(leverage.index, leverage.values, 
                label='Leverage', linewidth=2, 
                color='red', linestyle='--')
        ax2.set_ylabel('Leverage Ratio')
        
        if leverage_component is not None:
            for column in leverage_component.columns:
                ax2.plot(leverage_component.index, leverage_component[column], 
                        label=column, linewidth=2, linestyle=':')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    plt.title(title)
    plt.tight_layout()