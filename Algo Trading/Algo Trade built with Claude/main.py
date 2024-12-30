import os
import pandas as pd
import matplotlib.pyplot as plt
from lstm_enhanced_strategy import LSTMEnhancedStrategy
from equal_weight_strategy import EqualWeightStrategy
from regime_switching_strategy import RegimeSwitchingStrategy
from plot_results import plot_backtest_results, plot_portfolio_weights, save_backtest_results, plot_btc_regimes
from data_loader import load_and_split_data
from backtest_utils import run_backtest, plot_regime_transitions
from config import ModelConfig, RegimeConfig

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    # Create results directory if it doesn't exist
    results_dir = 'result'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "XRPUSDT",
        "LTCUSDT",
        "EOSUSDT",
        "NEOUSDT",
        "QTUMUSDT"
    ]
    initial_capital = 10000
    
    # Load all historical data
    print("Loading historical data...")

    market_data = load_and_split_data(symbols, "2020-01-01", "2020-01-31", ModelConfig)
    
    # Create strategies
    regime_strategy = RegimeSwitchingStrategy(symbols, lookback_period=RegimeConfig.LOOKBACK_PERIOD)
    equal_weight_strategy = EqualWeightStrategy()
    
    # Train regime switching strategy
    print("\nTraining regime switching strategy...")
    regime_strategy.train(market_data.train_data, market_data.val_data)
    
    # Run backtests
    print("\nRunning backtests...")
    
    regime_stats, regime_equity, regime_trades = run_backtest(
        regime_strategy, "regime", symbols, initial_capital, 
        preloaded_data=market_data.test_data
    )
    
    bench_stats, bench_equity, bench_trades = run_backtest(
        equal_weight_strategy, "benchmark", symbols, initial_capital,
        preloaded_data=market_data.test_data
    )
    
    # Plot and save results
    fig1 = plot_backtest_results(regime_equity, regime_trades, initial_capital, bench_equity)
    fig1.savefig(os.path.join(results_dir, 'strategy_performance.png'), dpi=300, bbox_inches='tight')

    # Plot and save weight allocations
    fig2 = plot_portfolio_weights(regime_equity, regime_trades, symbols)
    fig2.savefig(os.path.join(results_dir, 'portfolio_weights.png'), dpi=300, bbox_inches='tight')
    
    # Plot BTC price and regimes
    regime_stats = regime_strategy.get_regime_stats()
    if not regime_stats.empty:
        fig3 = plot_btc_regimes(regime_equity, regime_stats)
        fig3.savefig(os.path.join(results_dir, 'btc_regimes.png'), dpi=300, bbox_inches='tight')
        print("- btc_regimes.png: BTC price and market regime analysis")
    
    # Plot regime transitions
    if not regime_stats.empty:
        fig4 = plot_regime_transitions(regime_stats, output_dir=results_dir)
        if fig4 is not None:
            fig4.savefig(os.path.join(results_dir, 'regime_transitions.png'), dpi=300, bbox_inches='tight')
            print("- regime_transitions.png: Market regime transition analysis")
    else:
        print("Warning: No regime statistics available for plotting")

    # Save all backtest results with updated paths
    save_backtest_results(
        mv_stats=regime_stats,
        bench_stats=bench_stats,
        lstm_metrics={'test_mae': 0},  # Placeholder for LSTM metrics
        mv_equity=regime_equity, 
        bench_equity=bench_equity,
        mv_trades=regime_trades,
        symbols=symbols,
        output_dir=results_dir  # Add output directory parameter
    )

    # Show plots
    plt.show()

    # Print summary of saved files
    print("\nSaved files in 'result' directory:")
    print("- strategy_performance.png: Strategy vs benchmark performance plot")
    print("- portfolio_weights.png: Portfolio weight allocations plot")
    print("- backtest_metrics.csv: Performance metrics for both strategies")
    print("- regime_equity_curve.csv: Regime strategy equity curve data")
    print("- benchmark_equity_curve.csv: Benchmark equity curve data")
    print("- regime_trades.csv: Detailed trade history")
    print("- portfolio_weights.csv: Portfolio weight allocations over time")

if __name__ == "__main__":
    main()