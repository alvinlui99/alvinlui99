import os
import matplotlib.pyplot as plt
from typing import List
from backtester import Backtester
from data_utils import load_historical_data, split_data_by_date, prepare_features_for_symbols, FEATURE_NAMES
from strategy_ml import MLStrategy
from strategy_equal_weight import EqualWeightStrategy
from visualization import plot_curves
import pandas as pd

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
    backtester.save_portfolio_to_csv(f'results/{strategy_name}_portfolio.csv')
    return stats, backtester.equity_curve

def main():
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
        "LTCUSDT", "EOSUSDT", "NEOUSDT", "QTUMUSDT"
    ]
    initial_capital = 10000
    
    train_features = prepare_features_for_symbols(load_historical_data(symbols, "2020-01-01", "2022-12-31"))
    val_features = prepare_features_for_symbols(load_historical_data(symbols, "2023-01-01", "2023-12-31"))
    test_features = prepare_features_for_symbols(load_historical_data(symbols, "2024-01-01", "2024-12-31"))

    # Create and train LSTM strategy
    ml_strategy = MLStrategy()
    
    # Train with validation
    ml_strategy.train(train_features, val_features, FEATURE_NAMES, symbols)
    
    # Run backtests
    print("\nRunning backtests...")
    ml_stats, ml_equity = run_backtest(
        ml_strategy, "ML Strategy", symbols, initial_capital, test_features
    )

    bench_stats, bench_equity = run_backtest(
        EqualWeightStrategy(), "Equal Weight Benchmark", symbols, initial_capital, test_features
    )

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Plot equity curves and save
    plt.figure(figsize=(12, 6))
    plot_curves(
        ml_equity,
        bench_equity,
        ml_stats['leverage_curve'],
        # ml_stats['leverage_components'],
        title="ML Strategy vs Equal Weight Benchmark"
    )
    plt.savefig('results/equity_curves.png', bbox_inches='tight', dpi=300)
    
    # Print commission analysis
    print("\nCommission Analysis:")
    commission_analysis = ml_stats['commission_analysis']
    print(f"Total Commission: ${commission_analysis['total_commission']:.2f}")
    print(f"Commission Impact: {commission_analysis['commission_impact_pct']:.2f}%")
    
    # Save commission analysis to CSV
    commission_df = pd.DataFrame([{
        'Total Commission ($)': commission_analysis['total_commission'],
        'Commission Impact (%)': commission_analysis['commission_impact_pct']
    }])
    commission_df.to_csv('results/commission_analysis.csv', index=False)
    ml_stats['leverage_components'].to_csv('results/leverage_components.csv', index=False)

if __name__ == "__main__":
    main()