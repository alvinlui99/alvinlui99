import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
from dotenv import load_dotenv
from binance.um_futures import UMFutures
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategy.statistical_arbitrage import StatisticalArbitrageStrategy
from src.strategy.zscore_monitor import ZScoreMonitor
from src.strategy.position_sizer import PositionSizer
from src.strategy.backtest import Backtest
from src.data.market_data import MarketData

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_data(data: Dict[str, pd.DataFrame], 
               train_start: datetime,
               train_end: datetime,
               test_start: datetime,
               test_end: datetime,
               val_start: datetime,
               val_end: datetime) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split data into training, testing, and validation sets.
    """
    split_data = {}
    
    for symbol, df in data.items():
        split_data[symbol] = {
            'train': df[train_start:train_end],
            'test': df[test_start:test_end],
            'val': df[val_start:val_end]
        }
    
    return split_data

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Binance client
    client = UMFutures()
    
    # Define trading pairs
    pairs = [
        ('LINKUSDT', 'NEARUSDT'),
        ('WIFUSDT', 'TRUMPUSDT'),
        ('AVAXUSDT', '1000SHIBUSDT'),
        ('WLDUSDT', 'ETHUSDT'),
        ('DOGEUSDT', '1000PEPEUSDT')
    ]
    
    # Define time periods
    end_date = datetime.now()
    val_end = end_date
    val_start = end_date - timedelta(days=30)
    test_end = val_start
    test_start = test_end - timedelta(days=60)
    train_end = test_start
    train_start = train_end - timedelta(days=180)  # 6 months of training data
    
    logger.info(f"Training period: {train_start} to {train_end}")
    logger.info(f"Testing period: {test_start} to {test_end}")
    logger.info(f"Validation period: {val_start} to {val_end}")
    
    # Initialize components
    strategy = StatisticalArbitrageStrategy(
        client=client,
        pairs=pairs,
        timeframe='15m',
        lookback_periods=100
    )
    
    monitor = ZScoreMonitor(
        client=client,
        pairs=pairs,
        lookback_periods=100
    )
    
    sizer = PositionSizer(
        initial_capital=10000,
        max_position_size=0.2,
        min_confidence=0.4
    )
    
    # Fetch historical data
    symbols = [symbol for pair in pairs for symbol in pair]
    market_data = MarketData(
        client=client,
        symbols=symbols,
        start_date=train_start.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        timeframe='15m'
    )
    
    # Fetch data for all symbols
    data = market_data.fetch_historical_data()
    
    # Split data into train/test/val sets
    split_data_dict = split_data(
        data,
        train_start, train_end,
        test_start, test_end,
        val_start, val_end
    )
    
    # Initialize backtest
    backtest = Backtest(
        strategy=strategy,
        initial_capital=10000,
        commission=0.0004  # 0.04% commission
    )
    
    # Run backtest on each period
    results = {}
    
    for period in ['train', 'test', 'val']:
        logger.info(f"\nRunning backtest on {period} period...")
        period_data = {symbol: data[period] for symbol, data in split_data_dict.items()}
        results[period] = backtest.run(period_data)
        
        # Log results
        logger.info(f"\n{period.upper()} Period Results:")
        logger.info(f"Initial Capital: {results[period]['initial_capital']:.2f} USDT")
        logger.info(f"Final Capital: {results[period]['final_capital']:.2f} USDT")
        logger.info(f"Total PnL: {results[period]['total_pnl']:.2f} USDT")
        logger.info(f"Return: {results[period]['return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {results[period]['sharpe_ratio']:.2f}")
        logger.info(f"Maximum Drawdown: {results[period]['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {results[period]['win_rate']*100:.2f}%")
        logger.info(f"Number of Trades: {results[period]['num_trades']}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('backtest_results.csv')
    logger.info("\nResults saved to backtest_results.csv")

if __name__ == "__main__":
    main() 