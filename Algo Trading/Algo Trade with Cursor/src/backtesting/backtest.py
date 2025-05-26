import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import os

from data import BinanceDataCollector, DataProcessor
from utils import PairIndicators
from strategy import PairStrategy
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
        self.strategy = PairStrategy()
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
    
    def run_backtest(self) -> pd.DataFrame:
        """Run the backtest and return results using portfolio value tracking."""
        # Fetch and process data
        df1, df2 = self.fetch_data()
        signals = self.strategy.generate_signals(df1, df2)

        # Initialize results
        results = pd.DataFrame(index=df1.index)
        results['symbol1_price'] = df1['close']
        results['symbol2_price'] = df2['close']
        results['correlation'] = signals['correlation']
        results['hedge_ratio'] = signals['hedge_ratio']
        results['zscore'] = signals['zscore']
        results['signal'] = signals['signal']
        results['symbol1_returns'] = df1['returns']
        results['symbol2_returns'] = df2['returns']

        # Portfolio tracking variables
        initial_portfolio_value = 10000.0
        portfolio_value = initial_portfolio_value
        in_position = False
        position_type = 0  # 1 for long spread, -1 for short spread
        entry_price1 = entry_price2 = 0.0
        units1 = units2 = 0.0
        results['portfolio_value'] = np.nan
        results['strategy_returns'] = 0.0

        for i, (idx, row) in enumerate(results.iterrows()):
            signal = row['signal']
            price1 = row['symbol1_price']
            price2 = row['symbol2_price']
            hedge_ratio = row['hedge_ratio']
            
            # Enter new position if signal changes
            if not in_position and signal != 0:
                in_position = True
                position_type = signal
                entry_price1 = price1
                entry_price2 = price2
                entry_portfolio_value = portfolio_value
                
                # Calculate position sizes based on hedge ratio
                unit_cost = price1 + price2 * hedge_ratio
                
                if position_type == 1:
                    units1 = portfolio_value / unit_cost
                else:
                    units1 = -portfolio_value / unit_cost
                units2 = -units1 * hedge_ratio
                
            # If in position, update portfolio value
            elif in_position:
                if position_type == 1:
                    portfolio_value = units1 * price1 - units2 * entry_price2 - units2 * (entry_price2 - price2)
                else:
                    portfolio_value = units2 * price2 - units1 * entry_price1 - units1 * (entry_price1 - price1)
                if signal == 0 or signal != position_type:
                    # Exit position if signal goes to 0 or flips
                    in_position = False
                    position_type = 0
                    units1 = units2 = 0.0
                
            # If not in position, portfolio value stays the same
            results.at[idx, 'portfolio_value'] = portfolio_value
            
            # Calculate strategy return (log return for compounding)
            if i > 0 and not np.isnan(results.iloc[i-1]['portfolio_value']):
                prev_value = results.iloc[i-1]['portfolio_value']
                if prev_value > 0:
                    results.at[idx, 'strategy_returns'] = np.log(portfolio_value / prev_value)
                else:
                    results.at[idx, 'strategy_returns'] = 0.0
            else:
                results.at[idx, 'strategy_returns'] = 0.0

        # Fill initial portfolio value
        results['portfolio_value'].ffill()
        results.fillna({'portfolio_value': initial_portfolio_value}, inplace=True)
        # Calculate cumulative returns from portfolio value
        results['cumulative_returns'] = results['portfolio_value'] / initial_portfolio_value - 1
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
        start_date='2020-01-01 00:00:00',
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