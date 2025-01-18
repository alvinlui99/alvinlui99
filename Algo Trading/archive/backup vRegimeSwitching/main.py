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
from portfolio import Portfolio

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def plot_debug_regimes(market_data, regime_stats, output_dir):
    """Plot detailed regime analysis with debug information"""
    if regime_stats.empty:
        print("No regime statistics available for plotting")
        return None
    
    # Get BTC price data
    btc_cols = [col for col in market_data.test_data.columns if 'BTCUSDT' in col and 'price' in col]
    if not btc_cols:
        print("No BTC price data found")
        return None
    
    btc_prices = market_data.test_data[btc_cols[0]]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    
    # Plot BTC price
    ax1.plot(btc_prices.index, btc_prices.values, color='gray', alpha=0.6, label='BTC Price')
    
    # Color regions by regime
    colors = {'Bull': 'green', 'Neutral': 'yellow', 'Bear': 'red'}
    last_regime = None
    start_idx = btc_prices.index[0]
    
    for idx, row in regime_stats.iterrows():
        if last_regime != row['regime']:
            if last_regime is not None:
                ax1.axvspan(start_idx, idx, alpha=0.2, color=colors[last_regime])
            last_regime = row['regime']
            start_idx = idx
    
    # Plot the last regime
    if last_regime is not None:
        ax1.axvspan(start_idx, btc_prices.index[-1], alpha=0.2, color=colors[last_regime])
    
    # Plot regime probabilities
    ax2.plot(regime_stats.index, regime_stats['bull_prob'], color='green', label='Bull Probability')
    ax2.plot(regime_stats.index, regime_stats['neutral_prob'], color='yellow', label='Neutral Probability')
    ax2.plot(regime_stats.index, regime_stats['bear_prob'], color='red', label='Bear Probability')
    
    # Customize plots
    ax1.set_title('BTC Price with Market Regimes')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Regime Probabilities')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    fig.savefig(os.path.join(output_dir, 'regime_debug.png'), dpi=300, bbox_inches='tight')
    return fig

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
    # Ensure we have enough data for the lookback period
    lookback_hours = RegimeConfig.get_max_lookback()
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    market_data = load_and_split_data(symbols, start_date, end_date, ModelConfig)
    
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
    
    # Debug print benchmark results
    print("\nBenchmark Strategy Results:")
    print(f"Final Portfolio Value: {bench_equity.iloc[-1] if not bench_equity.empty else 'N/A'}")
    print(f"Number of Trades: {len(bench_trades) if bench_trades is not None else 'N/A'}")
    print(f"Strategy Stats: {bench_stats}")
    
    # Compare strategies
    print("\nStrategy Comparison:")
    if not regime_equity.empty and not bench_equity.empty:
        regime_returns = float((regime_equity.iloc[-1] - initial_capital) / initial_capital * 100)
        bench_returns = float((bench_equity.iloc[-1] - initial_capital) / initial_capital * 100)
        print(f"Regime Strategy Return: {regime_returns:.2f}%")
        print(f"Benchmark Strategy Return: {bench_returns:.2f}%")
        print(f"Outperformance: {regime_returns - bench_returns:.2f}%")
    
    # Get regime statistics
    regime_stats_df = regime_strategy.get_regime_stats()
    
    # Print debug information about regimes
    print("\nRegime Detection Debug Information:")
    if not regime_stats_df.empty:
        regime_counts = regime_stats_df['regime'].value_counts()
        print("\nRegime Distribution:")
        print(regime_counts)
        
        # Calculate regime transitions
        transitions = regime_stats_df['regime'].value_counts()
        print("\nRegime Transitions:")
        for i in range(len(regime_stats_df) - 1):
            if regime_stats_df['regime'].iloc[i] != regime_stats_df['regime'].iloc[i + 1]:
                print(f"Transition at {regime_stats_df.index[i]}: {regime_stats_df['regime'].iloc[i]} -> {regime_stats_df['regime'].iloc[i + 1]}")
        
        # Print regime characteristics
        if hasattr(regime_strategy.regime_detector, 'get_regime_characteristics'):
            print("\nRegime Characteristics:")
            characteristics = regime_strategy.regime_detector.get_regime_characteristics()
            for regime, stats in characteristics.items():
                print(f"\n{regime} Regime:")
                for metric, value in stats.items():
                    print(f"  {metric}: {value:.4f}")
    else:
        print("No regime statistics available")
    
    # Plot and save results
    fig1 = plot_backtest_results(regime_equity, regime_trades, initial_capital, bench_equity)
    fig1.savefig(os.path.join(results_dir, 'strategy_performance.png'), dpi=300, bbox_inches='tight')

    # Plot and save weight allocations
    fig2 = plot_portfolio_weights(regime_equity, regime_trades, symbols)
    fig2.savefig(os.path.join(results_dir, 'portfolio_weights.png'), dpi=300, bbox_inches='tight')
    
    # Plot BTC price and regimes
    if not regime_stats_df.empty:
        fig3 = plot_btc_regimes(regime_equity, regime_stats_df)
        fig3.savefig(os.path.join(results_dir, 'btc_regimes.png'), dpi=300, bbox_inches='tight')
        print("- btc_regimes.png: BTC price and market regime analysis")
        
        # Plot debug regime information
        fig4 = plot_debug_regimes(market_data, regime_stats_df, results_dir)
        if fig4 is not None:
            print("- regime_debug.png: Detailed regime analysis with probabilities")
    
    # Plot regime transitions
    if not regime_stats_df.empty:
        fig5 = plot_regime_transitions(regime_stats_df, output_dir=results_dir)
        if fig5 is not None:
            fig5.savefig(os.path.join(results_dir, 'regime_transitions.png'), dpi=300, bbox_inches='tight')
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
        output_dir=results_dir
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
    print("- regime_debug.png: Detailed regime analysis with probabilities")

if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    main()