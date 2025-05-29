import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import os
import logging

from data import BinanceDataCollector, DataProcessor
from utils import PairIndicators
from strategy import PairStrategy
from backtesting.performance import PerformanceAnalyzer
from backtesting.visualizer import BacktestVisualizer

logger = logging.getLogger(__name__)

class PairBacktest:
    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        start_date: str,
        end_date: str,
        interval: str = '1h',
        window: int = 20,
        commission: float = 0.00045
    ):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.window = window
        self.commission = commission
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
        
        # Initialize results
        results = pd.DataFrame(index=df1.index)
        results['symbol1_price'] = df1['close']
        results['symbol2_price'] = df2['close']
        results['symbol1_returns'] = df1['returns']
        results['symbol2_returns'] = df2['returns']

        # Portfolio tracking variables
        initial_portfolio_value = 10000.0
        portfolio_value = initial_portfolio_value
        in_position = False
        position_type = 0  # 1 for long spread, -1 for short spread
        entry_price1 = entry_price2 = 0.0
        unrealised_pnl = 0.0
        units1 = units2 = 0.0
        results['portfolio_value'] = np.nan
        results['strategy_returns'] = 0.0
        results['commission'] = 0.0  # Add commission tracking
        total_commission = 0.0  # Track total commission paid

        # Initialize position state
        position_state = {
            'in_position': False,
            'position_type': 0,
            'unrealised_pnl': 0.0,
        }

        for i, (idx, row) in enumerate(results.iterrows()):
            # Update position state
            position_state['in_position'] = in_position
            position_state['position_type'] = position_type
            position_state['unrealised_pnl'] = unrealised_pnl

            # Generate signals with current position state
            current_signals = self.strategy.generate_signals(
                df1.iloc[max(0, i-399):i+1], 
                df2.iloc[max(0, i-399):i+1],
                position_state
            )
            
            # Get current signal and metrics
            signal = current_signals['signal']
            hedge_ratio = current_signals['hedge_ratio']
            
            # Update results with current signals
            results.at[idx, 'adf_pvalue'] = current_signals['adf_pvalue']
            results.at[idx, 'hedge_ratio'] = current_signals['hedge_ratio']
            results.at[idx, 'zscore'] = current_signals['zscore']
            results.at[idx, 'signal'] = signal
            results.at[idx, 'vol_ratio'] = current_signals['vol_ratio']
            results.at[idx, 'dynamic_threshold'] = current_signals['dynamic_threshold']
            
            price1 = row['symbol1_price']
            price2 = row['symbol2_price']
            
            # Enter new position if signal changes
            if not in_position and signal != 0:
                in_position = True
                position_type = signal
                entry_price1 = price1
                entry_price2 = price2
                
                # Calculate position sizes based on hedge ratio
                unit_cost = price1 + price2 * abs(hedge_ratio)

                units1 = position_type * portfolio_value / unit_cost
                units2 = -units1 * hedge_ratio
                
                # Calculate and record entry commission
                entry_commission = (abs(units1) * price1 + abs(units2) * price2) * self.commission
                total_commission += entry_commission
                results.at[idx, 'commission'] = entry_commission
            
            # If in position, update portfolio value
            elif in_position:
                unrealised_pnl = units1 * (price1 - entry_price1) + units2 * (price2 - entry_price2)
                portfolio_value = abs(units1) * entry_price1 + abs(units2) * entry_price2 + unrealised_pnl
                if signal == 0 or signal != position_type:
                    # Calculate and record exit commission
                    exit_commission = (abs(units1) * price1 + abs(units2) * price2) * self.commission
                    total_commission += exit_commission
                    results.at[idx, 'commission'] = exit_commission
                    
                    # Exit position if signal goes to 0 or flips
                    in_position = False
                    position_type = 0
                    units1 = units2 = 0.0
                    unrealised_pnl = 0.0

            # If not in position, portfolio value stays the same
            results.at[idx, 'portfolio_value'] = portfolio_value - total_commission  # Subtract total commission from portfolio value
            
            # Calculate strategy return (log return for compounding)
            if i > 0 and not np.isnan(results.iloc[i-1]['portfolio_value']):
                prev_value = results.iloc[i-1]['portfolio_value']
                if prev_value > 0:
                    results.at[idx, 'strategy_returns'] = np.log(portfolio_value / prev_value)
                else:
                    results.at[idx, 'strategy_returns'] = 0.0
            else:
                results.at[idx, 'strategy_returns'] = 0.0

            print(f"Current backtest progress: {i}/{len(results)}")

        # Fill initial portfolio value
        results['portfolio_value'].ffill()
        results.fillna({'portfolio_value': initial_portfolio_value}, inplace=True)
        # Calculate cumulative returns from portfolio value
        results['cumulative_returns'] = results['portfolio_value'] / initial_portfolio_value - 1
        results['total_commission'] = total_commission  # Add total commission to results
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
    backtest = PairBacktest(
        symbol1='BTCUSDT',
        symbol2='ETHUSDT',
        start_date='2020-01-01 00:00:00',
        end_date='2024-12-31 00:00:00',
        interval='1h'
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