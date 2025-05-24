import pandas as pd
import numpy as np
from typing import Dict
import os
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, results: pd.DataFrame):
        self.results = results
        
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        returns = self.results['strategy_returns']
        cumulative_returns = self.results['cumulative_returns']
        
        # Basic metrics
        total_return = cumulative_returns.iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        daily_returns = returns.resample('D').sum()
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Trade metrics
        trades = self.results[self.results['signal'] != 0]
        win_rate = len(trades[trades['strategy_returns'] > 0]) / len(trades) if len(trades) > 0 else 0
        avg_trade_return = trades['strategy_returns'].mean()
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'total_trades': len(trades),
            'avg_trade_duration': self._calculate_avg_trade_duration(),
            'profit_factor': self._calculate_profit_factor(),
            'calmar_ratio': abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        }
        
        return metrics
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in hours."""
        trades = self.results[self.results['signal'] != 0]
        if len(trades) < 2:
            return 0
        
        trade_durations = []
        current_trade_start = None
        
        for idx, row in trades.iterrows():
            if current_trade_start is None:
                current_trade_start = idx
            elif row['signal'] == 0:  # Trade closed
                duration = (idx - current_trade_start).total_seconds() / 3600  # Convert to hours
                trade_durations.append(duration)
                current_trade_start = None
        
        return np.mean(trade_durations) if trade_durations else 0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        trades = self.results[self.results['signal'] != 0]
        gross_profit = trades[trades['strategy_returns'] > 0]['strategy_returns'].sum()
        gross_loss = abs(trades[trades['strategy_returns'] < 0]['strategy_returns'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def save_metrics(self, metrics: Dict, output_dir: str = 'backtest_results'):
        """Save performance metrics to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([metrics])
        
        # Save to CSV
        metrics_file = f"{output_dir}/performance_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"Performance metrics saved to {metrics_file}")
        
        # Print summary
        print("\nPerformance Metrics Summary:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in key or 'ratio' in key:
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    # Example usage
    from backtest import PairBacktest
    
    # Run backtest
    backtest = PairBacktest(
        symbol1='BTCUSDT',
        symbol2='ETHUSDT',
        start_date='2024-01-01 00:00:00',
        end_date='2024-02-01 00:00:00'
    )
    results = backtest.run_backtest()
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(results)
    metrics = analyzer.calculate_metrics()
    
    # Save metrics
    analyzer.save_metrics(metrics)
