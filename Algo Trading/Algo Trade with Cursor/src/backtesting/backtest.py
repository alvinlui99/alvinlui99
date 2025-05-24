import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import os
# from data.collector import BinanceDataCollector
# from data.processor import DataProcessor
from data import BinanceDataCollector, DataProcessor
from utils import PairIndicators
from backtesting.performance import PerformanceAnalyzer
from backtesting.visualizer import BacktestVisualizer

class PairBacktest:
    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        start_date: str,
        end_date: str,
        interval: str = '1h',
        zscore_threshold: float = 2.0,
        correlation_threshold: float = 0.8,
        window: int = 20
    ):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.zscore_threshold = zscore_threshold
        self.correlation_threshold = correlation_threshold
        self.window = window
        
        self.collector = BinanceDataCollector()
        self.processor = DataProcessor()
        self.trades = []
        self.positions = {}
        
    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and process historical data for both symbols."""
        # Get data for both symbols
        df1 = self.collector.get_historical_klines(
            self.symbol1,
            interval=self.interval,
            start_str=self.start_date,
            end_str=self.end_date
        )
        df2 = self.collector.get_historical_klines(
            self.symbol2,
            interval=self.interval,
            start_str=self.start_date,
            end_str=self.end_date
        )
        
        # Process individual assets
        df1 = self.processor.process_single_asset(df1)
        df2 = self.processor.process_single_asset(df2)
        
        # Align dataframes before returning
        df1_aligned, df2_aligned = PairIndicators.align_dataframes(df1, df2)
        
        return df1_aligned, df2_aligned
    
    def calculate_signals(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on z-score and correlation."""
        # Calculate pair metrics
        correlation = PairIndicators.calculate_correlation(df1, df2, self.window)
        zscore = PairIndicators.calculate_pair_zscore(df1, df2, self.window)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df1.index)
        signals['correlation'] = correlation
        signals['zscore'] = zscore
        
        # Generate trading signals
        signals['signal'] = 0
        signals.loc[(signals['correlation'] >= self.correlation_threshold) & 
                   (signals['zscore'] >= self.zscore_threshold), 'signal'] = -1  # Short signal
        signals.loc[(signals['correlation'] >= self.correlation_threshold) & 
                   (signals['zscore'] <= -self.zscore_threshold), 'signal'] = 1   # Long signal
        
        return signals
    
    def run_backtest(self) -> pd.DataFrame:
        """Run the backtest and return results."""
        # Fetch and process data
        df1, df2 = self.fetch_data()
        
        # Calculate signals
        signals = self.calculate_signals(df1, df2)

        # Initialize results
        results = pd.DataFrame(index=df1.index)
        results['symbol1_price'] = df1['close']
        results['symbol2_price'] = df2['close']
        results['correlation'] = signals['correlation']
        results['zscore'] = signals['zscore']
        results['signal'] = signals['signal']
        
        # Calculate returns
        results['symbol1_returns'] = df1['returns']
        results['symbol2_returns'] = df2['returns']
        
        # Calculate strategy returns
        results['strategy_returns'] = 0.0
        results.loc[results['signal'] == 1, 'strategy_returns'] = (
            results['symbol1_returns'] - results['symbol2_returns']
        )
        results.loc[results['signal'] == -1, 'strategy_returns'] = (
            results['symbol2_returns'] - results['symbol1_returns']
        )
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod() - 1
        
        return results
    
    def save_results(self, results: pd.DataFrame, output_dir: str = 'backtest_results'):
        """Save backtest results to CSV files."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_file = f"{output_dir}/backtest_results_{self.symbol1}_{self.symbol2}_{timestamp}.csv"
        results.to_csv(results_file)
        
        # Save trade summary
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_file = f"{output_dir}/trades_{self.symbol1}_{self.symbol2}_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
        
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    backtest = PairBacktest(
        symbol1='BTCUSDT',
        symbol2='ETHUSDT',
        start_date='2023-01-01 00:00:00',
        end_date='2023-12-31 00:00:00',
        interval='1h',
        zscore_threshold=2.0,
        correlation_threshold=0.8
    )
    
    # Run backtest
    results = backtest.run_backtest()
    
    # Save results
    backtest.save_results(results)

    # Calculate performance metrics
    
    analyzer = PerformanceAnalyzer(results)
    metrics = analyzer.calculate_metrics()
    
    # Create and save visualizations
    visualizer = BacktestVisualizer(results, metrics)
    visualizer.save_all_plots()
    
    # Print summary
    print("\nBacktest Results Summary:")
    print(f"Total Return: {results['cumulative_returns'].iloc[-1]:.2%}")
    print(f"Number of Trades: {len(results[results['signal'] != 0])}")
    print(f"Average Return per Trade: {results['strategy_returns'].mean():.2%}")
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")