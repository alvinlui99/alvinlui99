import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from lstm_enhanced_strategy import LSTMEnhancedStrategy
from equal_weight_strategy import EqualWeightStrategy
from backtester import Backtester
from plot_results import plot_backtest_results, plot_portfolio_weights, save_backtest_results
from config import (DATA_PATH, REBALANCE_THRESHOLD, LOOKBACK_PERIOD,
                   TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, USE_TRAINED_MODEL,
                   MODEL_PATH, SAVE_MODEL)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_historical_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load and prepare historical data for LSTM training"""
    all_data = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}:")
        df = pd.read_csv(f"{DATA_PATH}/{symbol}.csv")
        print(f"  Raw data shape: {df.shape}")
        
        df['datetime'] = pd.to_datetime(df['index'])
        df.set_index('datetime', inplace=True)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print(f"  After date filtering: {df.shape}")
        
        symbol_data = pd.DataFrame(index=df.index)
        
        # Basic features
        symbol_data['price'] = df['Close']
        symbol_data['return'] = df['Close'].pct_change()
        
        # Technical indicators
        # Volatility and momentum
        symbol_data['volatility'] = symbol_data['return'].rolling(LOOKBACK_PERIOD).std()
        symbol_data['momentum'] = symbol_data['return'].rolling(LOOKBACK_PERIOD).mean()
        
        # Price relative to SMA
        price_sma = symbol_data['price'].rolling(LOOKBACK_PERIOD).mean()
        symbol_data['price_to_sma'] = symbol_data['price'] / price_sma
        symbol_data['price_std'] = symbol_data['price'].rolling(LOOKBACK_PERIOD).std()
        
        # RSI
        delta = symbol_data['return']
        gain = (delta.where(delta > 0, 0)).rolling(window=LOOKBACK_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=LOOKBACK_PERIOD).mean()
        rs = gain / loss
        symbol_data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = symbol_data['price'].ewm(span=12, adjust=False).mean()
        exp2 = symbol_data['price'].ewm(span=26, adjust=False).mean()
        symbol_data['macd'] = exp1 - exp2
        symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        bb_sma = symbol_data['price'].rolling(window=LOOKBACK_PERIOD).mean()
        bb_std = symbol_data['price'].rolling(window=LOOKBACK_PERIOD).std()
        bb_upper = bb_sma + (2 * bb_std)
        bb_lower = bb_sma - (2 * bb_std)
        symbol_data['bb_position'] = (symbol_data['price'] - bb_lower) / (bb_upper - bb_lower)
        
        # Statistical moments
        symbol_data['skewness'] = symbol_data['return'].rolling(LOOKBACK_PERIOD).skew()
        symbol_data['kurtosis'] = symbol_data['return'].rolling(LOOKBACK_PERIOD).kurt()
        
        # Equal weight initially
        symbol_data['weight'] = 1.0 / len(symbols)
        
        # Fill NaN values for this symbol's data
        symbol_data = symbol_data.ffill().bfill()
        print(f"  Final features shape: {symbol_data.shape}")
        print(f"  NaN values remaining: {symbol_data.isna().sum().sum()}")
        
        all_data.append(symbol_data)
    
    # Combine all data and handle missing values
    combined_data = pd.concat(all_data).sort_index()
    print(f"\nCombined data shape: {combined_data.shape}")
    
    # Fill any remaining NaN values
    combined_data = combined_data.ffill().bfill()
    print(f"NaN values in combined data: {combined_data.isna().sum().sum()}")
    
    # Only drop rows if we still have NaN values after filling
    if combined_data.isna().sum().sum() > 0:
        print("Warning: Still have NaN values after filling, dropping those rows...")
        combined_data = combined_data.dropna()
        print(f"Final shape after dropping NaN: {combined_data.shape}")
    
    return combined_data

def split_data_by_date(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets by date"""
    data = data.sort_index()
    n = len(data)
    
    train_end = int(n * TRAIN_SIZE)
    val_end = train_end + int(n * VALIDATION_SIZE)
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    print(f"\nData split summary:")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Validation period: {val_data.index[0]} to {val_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    
    return train_data, val_data, test_data

def run_backtest(strategy, strategy_name, symbols, initial_capital, start_date, end_date):
    """Run backtest with given strategy and return results"""
    backtester = Backtester(
        symbols=symbols,
        strategy=strategy,
        initial_capital=initial_capital
    )
    
    stats = backtester.run(start_date=start_date, end_date=end_date)
    return stats, backtester.equity_curve, backtester.trade_history

def main():
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
    historical_data = load_historical_data(symbols, "2020-01-01", "2023-12-31")
    
    # Split data into train, validation, and test sets
    print("\nSplitting data...")
    train_data, val_data, test_data = split_data_by_date(historical_data)
    
    # Create strategies
    lstm_strategy = LSTMEnhancedStrategy()
    equal_weight_strategy = EqualWeightStrategy()
    
    if USE_TRAINED_MODEL:
        print(f"\nLoading trained model from {MODEL_PATH}...")
        lstm_strategy.load_model(MODEL_PATH)
    else:
        # Train the model
        print("\nTraining model...")
        train_metrics = lstm_strategy.train_model(train_data)
        if SAVE_MODEL:
            print(f"\nSaving model to {MODEL_PATH}...")
            lstm_strategy.save_model(MODEL_PATH)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = lstm_strategy.evaluate(test_data)
    print(f"Test MAE: {test_metrics['test_mae']:.6f}")
    
    # Run backtests
    print("\nRunning backtests...")
    test_start = test_data.index[0].strftime('%Y-%m-%d')
    test_end = test_data.index[-1].strftime('%Y-%m-%d')
    
    mv_stats, mv_equity, mv_trades = run_backtest(
        lstm_strategy, "lstm", symbols, initial_capital, 
        start_date=test_start, end_date=test_end
    )
    
    bench_stats, bench_equity, bench_trades = run_backtest(
        equal_weight_strategy, "benchmark", symbols, initial_capital,
        start_date=test_start, end_date=test_end
    )
    
    # Plot and save results
    fig1 = plot_backtest_results(mv_equity, mv_trades, initial_capital, bench_equity)
    fig1.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')

    # Plot and save weight allocations
    fig2 = plot_portfolio_weights(mv_equity, mv_trades, symbols)
    fig2.savefig('portfolio_weights.png', dpi=300, bbox_inches='tight')

    # Save all backtest results
    save_backtest_results(mv_stats, bench_stats, test_metrics, mv_equity, 
                         bench_equity, mv_trades, symbols)

    # Show plots
    plt.show()

    # Print summary of saved files
    print("\nSaved files:")
    print("- strategy_performance.png: Strategy vs benchmark performance plot")
    print("- portfolio_weights.png: Portfolio weight allocations plot")
    print("- backtest_metrics.csv: Performance metrics for both strategies")
    print("- lstm_equity_curve.csv: LSTM strategy equity curve data")
    print("- benchmark_equity_curve.csv: Benchmark equity curve data")
    print("- lstm_trades.csv: Detailed trade history")
    print("- portfolio_weights.csv: Portfolio weight allocations over time")

if __name__ == "__main__":
    main()