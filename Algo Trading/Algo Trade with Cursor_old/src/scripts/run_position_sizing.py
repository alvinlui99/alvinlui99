import os
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from dotenv import load_dotenv
from binance.um_futures import UMFutures
from src.strategy.statistical_arbitrage import StatisticalArbitrageStrategy
from src.strategy.backtest import Backtest
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Binance client
    client = UMFutures(
        key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_API_SECRET')
    )
    
    # Define pairs to monitor
    pairs = [
        ('LINKUSDT', 'NEARUSDT'),
        ('WIFUSDT', 'TRUMPUSDT'),
        ('AVAXUSDT', '1000SHIBUSDT'),
        ('WLDUSDT', 'ETHUSDT'),
        ('DOGEUSDT', '1000PEPEUSDT')
    ]
    
    # Initialize strategy with more lenient parameters
    strategy = StatisticalArbitrageStrategy(
        client=client,
        pairs=pairs,
        lookback_periods=100,
        entry_threshold=1.5,  # Reduced from 2.0
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
        timeframe='15m',
        initial_capital=10000.0,
        max_position_size=0.2,
        max_leverage=2.0,
        min_confidence=0.4,  # Reduced from 0.6
        volatility_threshold=0.05  # Increased from 0.02
    )
    
    # Initialize backtest
    backtest = Backtest(
        strategy=strategy,
        initial_capital=10000.0,
        commission=0.0004
    )
    
    logger.info("Starting position sizing backtest...")
    
    try:
        # Load historical data
        data = {}
        for symbol1, symbol2 in pairs:
            # Get historical data for both symbols
            klines1 = client.klines(
                symbol=symbol1,
                interval='15m',
                limit=1000,
                startTime=int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            )
            klines2 = client.klines(
                symbol=symbol2,
                interval='15m',
                limit=1000,
                startTime=int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            )
            
            # Convert to DataFrame
            df1 = pd.DataFrame(klines1, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df2 = pd.DataFrame(klines2, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
            df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')
            
            # Set index
            df1.set_index('timestamp', inplace=True)
            df2.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df1[col] = df1[col].astype(float)
                df2[col] = df2[col].astype(float)
            
            # Store in data dictionary
            data[symbol1] = df1
            data[symbol2] = df2
        
        # Run backtest
        results = backtest.run(data)
        
        # Print results
        logger.info("\nBacktest Results:")
        logger.info("-" * 50)
        logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
        logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
        logger.info(f"Total PnL: ${results['total_pnl']:,.2f}")
        logger.info(f"Return: {results['return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Number of Trades: {results['num_trades']}")
        logger.info("-" * 50)
        
        # Print trade details
        logger.info("\nTrade Details:")
        for trade in results['trades']:
            logger.info(f"\nSymbol: {trade['symbol']}")
            logger.info(f"Action: {trade['action']}")
            logger.info(f"Size: {trade['size']:.3f}")
            logger.info(f"Price: ${trade['price']:,.2f}")
            logger.info(f"PnL: ${trade['pnl']:,.2f}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")

if __name__ == "__main__":
    main() 