import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from backtester import Backtester
from trading_types import TradeRecord, EquityPoint, BacktestResults, RegimeStats

def run_backtest(
    strategy, 
    strategy_name: str, 
    symbols: List[str],
    initial_capital: float, 
    preloaded_data: pd.DataFrame
) -> BacktestResults:
    """
    Run backtest with given strategy and return results.
    
    Args:
        strategy: Strategy instance implementing __call__ method
        strategy_name: Name identifier for the strategy
        symbols: List of trading pair symbols (e.g. ["BTCUSDT", "ETHUSDT"])
        initial_capital: Starting portfolio value
        preloaded_data: DataFrame containing historical data
        
    Returns:
        BacktestResults containing:
            stats: Dict with performance metrics
                {
                    'total_return': float,  # Percentage return
                    'sharpe_ratio': float,
                    'max_drawdown': float,  # Maximum drawdown percentage
                    'win_rate': float,      # Percentage of winning trades
                    'profit_factor': float,
                    'volatility': float,    # Annualized volatility
                    'trades': int           # Total number of trades
                }
            equity_curve: DataFrame with columns:
                - timestamp (pd.Timestamp): Time index
                - equity (float): Portfolio value
            trade_history: List of TradeRecord dictionaries
    
    Raises:
        ValueError: If symbols are not found or preloaded_data is None
        TypeError: If strategy doesn't implement required interface
    """
    if preloaded_data is None:
        raise ValueError("preloaded_data is required")
        
    backtester = Backtester(
        symbols=symbols,
        strategy=strategy,
        initial_capital=initial_capital,
        preloaded_data=preloaded_data
    )
    
    stats = backtester.run()
    return BacktestResults(
        stats=stats,
        equity_curve=backtester.equity_curve,
        trade_history=backtester.trade_history
    )

def plot_regime_transitions(
    regime_stats: pd.DataFrame,
    save_path: str = 'regime_transitions.png'
) -> Optional[plt.Figure]:
    """
    Plot regime transition statistics.
    
    Args:
        regime_stats: DataFrame with columns:
            - regime (str): Regime label ('Bull', 'Neutral', 'Bear')
            - pct_time (float): Percentage of time in regime
            - avg_duration (float): Average duration in periods
            - transitions (Dict[str, float]): Transition probabilities
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object if successful, None if regime_stats is empty
        
    Raises:
        ValueError: If required columns are missing from regime_stats
    """
    required_columns = ['regime', 'pct_time', 'avg_duration']
    if not all(col in regime_stats.columns for col in required_columns):
        raise ValueError(
            f"regime_stats must contain columns: {required_columns}"
        )
    
    if regime_stats.empty:
        return None
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot regime distribution
    ax1.bar(regime_stats['regime'], regime_stats['pct_time'])
    ax1.set_title('Regime Distribution')
    ax1.set_ylabel('% of Time')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    
    # Plot average duration
    ax2.bar(regime_stats['regime'], regime_stats['avg_duration'])
    ax2.set_title('Average Regime Duration')
    ax2.set_ylabel('Number of Periods')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig 