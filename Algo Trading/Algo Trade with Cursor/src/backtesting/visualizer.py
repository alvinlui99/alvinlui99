import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import os
from datetime import datetime

class BacktestVisualizer:
    def __init__(self, results: pd.DataFrame, metrics: Dict):
        self.results = results
        self.metrics = metrics
        
    def plot_equity_curve(self, save_path: str = None):
        """Plot equity curve with asset returns."""
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve and normalized prices
        plt.plot(self.results.index, self.results['cumulative_returns'], label='Strategy Returns', linewidth=2)
        plt.title('Equity Curve and Asset Returns')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_trade_distribution(self, save_path: str = None):
        """Plot distribution of trade returns."""
        trades = self.results[self.results['signal'] != 0]['strategy_returns']
        
        plt.figure(figsize=(10, 6))
        sns.histplot(trades, bins=50, kde=True)
        plt.title('Distribution of Trade Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_heatmap(self, save_path: str = None):
        """Plot correlation heatmap of returns."""
        returns = self.results[['symbol1_returns', 'symbol2_returns', 'strategy_returns']]
        corr = returns.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Returns Correlation Heatmap')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_all_plots(self, output_dir: str = 'backtest_results', hedge_window: int = 240):
        """Save all plots to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plots
        self.plot_equity_curve(f"{output_dir}/equity_curve_{hedge_window}.png")
        self.plot_trade_distribution(f"{output_dir}/trade_distribution_{hedge_window}.png")
        self.plot_correlation_heatmap(f"{output_dir}/correlation_heatmap_{hedge_window}.png")
        
        print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    from backtesting.backtest import PairBacktest
    from backtesting.performance import PerformanceAnalyzer
    
    # Run backtest
    backtest = PairBacktest(
        symbol1='BTCUSDT',
        symbol2='ETHUSDT',
        start_date='2024-01-01 00:00:00',
        end_date='2024-02-01 00:00:00'
    )
    results = backtest.run_backtest()
    
    # Calculate metrics
    analyzer = PerformanceAnalyzer(results)
    metrics = analyzer.calculate_metrics()
    
    # Create visualizations
    visualizer = BacktestVisualizer(results, metrics)
    visualizer.save_all_plots(hedge_window=240)
