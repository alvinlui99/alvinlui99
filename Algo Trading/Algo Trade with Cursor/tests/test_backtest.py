import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
from datetime import datetime, timedelta
import logging
from src.strategy.macd_strategy import MACDStrategy
from src.strategy.backtest import Backtest
from src.data.market_data import MarketData
from binance.um_futures import UMFutures
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize Binance Futures client
    client = UMFutures(
        key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_API_SECRET'),
        base_url="https://testnet.binancefuture.com"
    )
    
    # Define trading pairs and timeframe
    trading_pairs = ['BTCUSDT']
    timeframe = '1h'
    
    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Test with 30 days of data
    
    market_data = MarketData(
        client=client,
        symbols=trading_pairs,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Fetch historical data
    historical_data = market_data.fetch_historical_data()
    
    # Print column names for debugging
    for symbol, df in historical_data.items():
        logger.info(f"Columns for {symbol}: {df.columns.tolist()}")
    
    # Initialize strategy
    strategy = MACDStrategy(
        trading_pairs=trading_pairs,
        timeframe=timeframe,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        risk_per_trade=0.02
    )
    
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
    logger.info(f"Average Win: {results['avg_win']:.2f} USDT")
    logger.info(f"Average Loss: {results['avg_loss']:.2f} USDT")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Print trade history
    logger.info("\nTrade History:")
    for trade in results['trades']:
        logger.info(f"Time: {trade['timestamp']}, "
                   f"Symbol: {trade['symbol']}, "
                   f"Action: {trade['action']}, "
                   f"Size: {trade['size']:.4f}, "
                   f"Price: {trade['price']:.2f}, "
                   f"PnL: {trade['pnl']:.2f} USDT")

if __name__ == "__main__":
    main() 