import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from src.strategy.macd_strategy import MACDStrategy
from src.strategy.backtest import Backtest
from src.data.market_data import MarketData
from binance.um_futures import UMFutures
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directory for plots
plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)

def plot_backtest_results(symbol, data, results, strategy):
    """
    Plot backtest results.
    
    Args:
        symbol (str): Trading pair
        data (pd.DataFrame): Price data with indicators
        results (dict): Backtest results
        strategy (BaseStrategy): Trading strategy
    """
    # Make sure we have the 'Close' column - already capitalized from Binance
    if 'Close' not in data.columns and 'closePrice' in data.columns:
        data = data.copy()
        data['Close'] = data['closePrice']
    
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(results['equity_curve'], columns=['equity'])
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(results['trades'])
    if not trades_df.empty:
        # Filter trades for the current symbol
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
    else:
        symbol_trades = pd.DataFrame(columns=['timestamp', 'action', 'price', 'size', 'pnl'])
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    
    # Price chart with trades
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['Close'], label='Price')
    
    # Plot buy and sell signals
    if not symbol_trades.empty and 'action' in symbol_trades.columns:
        buy_signals = symbol_trades[symbol_trades['action'] == 'BUY']
        sell_signals = symbol_trades[symbol_trades['action'] == 'SELL']
        close_signals = symbol_trades[symbol_trades['action'] == 'CLOSE']
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['price'], 
                       marker='^', color='green', s=100, label='Buy')
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['price'], 
                       marker='v', color='red', s=100, label='Sell')
        
        if not close_signals.empty:
            ax1.scatter(close_signals['timestamp'], close_signals['price'], 
                       marker='o', color='blue', s=100, label='Close')
    
    ax1.set_title(f'{symbol} Price Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # MACD
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if 'macd' in data.columns and 'signal' in data.columns and 'histogram' in data.columns:
        ax2.plot(data.index, data['macd'], label='MACD')
        ax2.plot(data.index, data['signal'], label='Signal')
        ax2.bar(data.index, data['histogram'], label='Histogram', alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.set_title('MACD Indicator')
        ax2.set_ylabel('MACD')
        ax2.grid(True)
        ax2.legend()
    else:
        logger.error("MACD indicators not found in data")
    
    # RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if 'rsi' in data.columns:
        ax3.plot(data.index, data['rsi'], label='RSI')
        ax3.axhline(y=strategy.rsi_oversold, color='g', linestyle='--', alpha=0.5, 
                   label=f'Oversold ({strategy.rsi_oversold})')
        ax3.axhline(y=strategy.rsi_overbought, color='r', linestyle='--', alpha=0.5, 
                   label=f'Overbought ({strategy.rsi_overbought})')
        ax3.axhline(y=50, color='k', linestyle='-', alpha=0.2)
        ax3.set_title('RSI Indicator')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        ax3.legend()
    else:
        logger.error("RSI indicator not found in data")
    
    # Equity curve
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(data.index[:len(equity_df)], equity_df['equity'], label='Equity Curve')
    ax4.set_title('Equity Curve')
    ax4.set_ylabel('Equity')
    ax4.set_xlabel('Date')
    ax4.grid(True)
    ax4.legend()
    
    # Format x-axis dates
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{symbol}_backtest.png'))
    plt.close()
    
    # Plot equity curve separately
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['equity'])
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'equity_curve.png'))
    plt.close()

def main():
    # Initialize Binance Futures client
    client = UMFutures(
        key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_API_SECRET'),
        base_url="https://testnet.binancefuture.com"
    )
    
    # Define trading pairs and timeframe
    trading_pairs = ['BTCUSDT']
    timeframe = '1h'  # Keep hourly timeframe as requested
    
    # Get historical data - use 90 days for hourly data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 90 days of hourly data
    
    # Create market data object with cache enabled
    market_data = MarketData(
        client=client,
        symbols=trading_pairs,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        timeframe=timeframe,
        use_cache=True  # Enable caching
    )
    
    # Fetch historical data - this will use cache if available
    historical_data = market_data.fetch_historical_data()
    
    # Modify strategy parameters for more signals - use much more relaxed RSI thresholds
    strategy = MACDStrategy(
        trading_pairs=trading_pairs,
        timeframe=timeframe,
        rsi_period=14,
        rsi_overbought=70,  # Very relaxed RSI thresholds to generate more signals
        rsi_oversold=30,
        risk_per_trade=0.02
    )
    
    # Pre-calculate indicators for all data
    for symbol, df in historical_data.items():
        historical_data[symbol] = strategy.calculate_indicators(df)
    
    # Initialize backtest
    backtest = Backtest(
        strategy=strategy,
        initial_capital=10000,  # 10,000 USDT
        commission=0.0004  # 0.04% commission
    )
    
    # Run backtest
    results = backtest.run(historical_data)
    
    # Print results
    logger.info("\nBacktest Results:")
    logger.info(f"Total Return: {results['total_return']:.2f}%")
    logger.info(f"Annual Return: {results['annual_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    logger.info(f"Win Rate: {results['win_rate']:.2f}%")
    logger.info(f"Total Trades: {results['total_trades']}")
    
    if results['total_trades'] > 0:
        logger.info(f"Average Win: {results['avg_win']:.2f} USDT")
        logger.info(f"Average Loss: {results['avg_loss']:.2f} USDT")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        
        # Plot backtest results for each symbol
        for symbol, df in historical_data.items():
            plot_backtest_results(symbol, df, results, strategy)
    else:
        logger.info("No trades were executed during the backtest period.")
        logger.info("Try adjusting the strategy parameters or using a different time period.")
        
        # Plot anyway to debug
        for symbol, df in historical_data.items():
            plot_backtest_results(symbol, df, results, strategy)

if __name__ == "__main__":
    main() 