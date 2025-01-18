import os
import matplotlib.pyplot as plt
from typing import List
from equal_weight_strategy import EqualWeightStrategy
from backtester import Backtester
from data_utils import load_historical_data, split_data_by_date, prepare_features_for_symbols, FEATURE_NAMES
from ml_strategy import MLStrategy
from visualization import plot_equity_curves, plot_performance_metrics, plot_rolling_metrics
import pandas as pd
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run_backtest(strategy, strategy_name, symbols, initial_capital, test_data):
    """Run backtest with given strategy and return results"""
    backtester = Backtester(
        symbols=symbols,
        strategy=strategy,
        initial_capital=initial_capital
    )
    
    stats = backtester.run(test_data=test_data)
    os.makedirs('results', exist_ok=True)
    backtester.save_positions_to_csv(f'results/{strategy_name}_positions.csv')
    return stats, backtester.equity_curve, backtester.analytics

def main():
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
        "LTCUSDT", "EOSUSDT", "NEOUSDT", "QTUMUSDT"
    ]
    initial_capital = 10000
    
    # Load raw historical data
    print("Loading historical data...")
    raw_data = load_historical_data(symbols, "2020-01-01", "2023-12-31")
    
    # Prepare features (now includes prices)
    print("Engineering features...")
    features_data = prepare_features_for_symbols(raw_data)
    
    # Split features into train, validation, and test sets
    print("\nSplitting data...")
    train_features, val_features, test_features = split_data_by_date(features_data)
    
    # Create and train LSTM strategy
    ml_strategy = MLStrategy()
    ml_strategy.configure(features=FEATURE_NAMES, symbols=symbols)
    
    # Train with validation
    val_metrics = ml_strategy.train(train_features, val_features)
    
    # Print validation metrics
    if val_metrics:
        print("\nValidation Metrics:")
        print(f"Sharpe Ratio: {val_metrics['sharpe_ratio']:.2f}")
        print(f"Annual Return: {val_metrics['mean_return']*100:.1f}%")
        print(f"Annual Volatility: {val_metrics['volatility']*100:.1f}%")
        print(f"Maximum Drawdown: {val_metrics['max_drawdown']*100:.1f}%")
    
    # Run backtests
    print("\nRunning backtests...")
    ml_stats, ml_equity, ml_analytics = run_backtest(
        ml_strategy, "ML Strategy", symbols, initial_capital, test_features
    )
    
    bench_stats, bench_equity, bench_analytics = run_backtest(
        EqualWeightStrategy(), "Equal Weight Benchmark", symbols, initial_capital, test_features
    )

    # Convert equity DataFrames to Series
    ml_equity_series = pd.Series(
        ml_equity['equity'].values,
        index=ml_equity['timestamp']
    )
    bench_equity_series = pd.Series(
        bench_equity['equity'].values,
        index=bench_equity['timestamp']
    )

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Plot equity curves and save
    plt.figure(figsize=(12, 6))
    plot_equity_curves(
        ml_equity_series,
        bench_equity_series,
        title="ML Strategy vs Equal Weight Benchmark"
    )
    plt.savefig('results/equity_curves.png', bbox_inches='tight', dpi=300)
    
    # Get metrics from analytics
    ml_metrics = ml_analytics.calculate_metrics(ml_equity, ml_stats['trades'])
    metrics = {
        'Total Return (%)': ml_metrics['total_return'],
        'Sharpe Ratio': ml_metrics['sharpe_ratio'],
        'Sortino Ratio': ml_metrics['sortino_ratio'],
        'Max Drawdown (%)': ml_metrics['max_drawdown'],
        'Volatility (%)': ml_metrics['volatility'],
        'Win Rate (%)': ml_metrics['win_rate'],
        'Profit Factor': ml_metrics['profit_factor'],
        'Max Consecutive Losses': ml_metrics['max_consecutive_losses']
    }
    # Plot performance metrics and save
    plt.figure(figsize=(10, 6))
    plot_performance_metrics(metrics)
    plt.savefig('results/performance_metrics.png', bbox_inches='tight', dpi=300)
    
    # Plot rolling metrics and save
    plt.figure(figsize=(12, 8))
    plot_rolling_metrics(ml_equity_series, window=30)
    plt.savefig('results/rolling_metrics.png', bbox_inches='tight', dpi=300)
    
    # Print commission analysis
    print("\nCommission Analysis:")
    commission_analysis = ml_stats['commission_analysis']
    print(f"Total Commission: ${commission_analysis['total_commission']:.2f}")
    print(f"Commission Impact: {commission_analysis['commission_impact_pct']:.2f}%")
    print(f"Portfolio Turnover: {commission_analysis['turnover_ratio']:.2f}x")
    print(f"Number of Trades: {commission_analysis['trade_count']}")
    print(f"Average Trade Size: ${commission_analysis['avg_trade_size']:.2f}")
    
    # Save commission analysis to CSV
    commission_df = pd.DataFrame([{
        'Total Commission ($)': commission_analysis['total_commission'],
        'Commission Impact (%)': commission_analysis['commission_impact_pct'],
        'Portfolio Turnover': commission_analysis['turnover_ratio'],
        'Number of Trades': commission_analysis['trade_count'],
        'Average Trade Size ($)': commission_analysis['avg_trade_size']
    }])
    commission_df.to_csv('results/commission_analysis.csv', index=False)
    
    plt.show()

if __name__ == "__main__":
    main()