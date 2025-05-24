import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from dotenv import load_dotenv
from binance.um_futures import UMFutures
from src.strategy.zscore_monitor import ZScoreMonitor
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
    
    # Define pairs to monitor (using our selected pairs)
    pairs = [
        ('LINKUSDT', 'NEARUSDT'),
        ('WIFUSDT', 'TRUMPUSDT'),
        ('AVAXUSDT', '1000SHIBUSDT'),
        ('WLDUSDT', 'ETHUSDT'),
        ('DOGEUSDT', '1000PEPEUSDT')
    ]
    
    # Initialize Z-score monitor
    monitor = ZScoreMonitor(
        client=client,
        pairs=pairs,
        lookback_periods=100,  # 100 periods of 15m data
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
        timeframe='15m'
    )
    
    logger.info("Starting Z-score monitoring...")
    
    try:
        while True:
            # Get current prices for all pairs
            for symbol1, symbol2 in pairs:
                try:
                    # Get current prices
                    price1 = float(client.ticker_price(symbol=symbol1)['price'])
                    price2 = float(client.ticker_price(symbol=symbol2)['price'])
                    
                    # Update monitor with new prices
                    monitor.update_prices(symbol1, symbol2, price1, price2)
                    
                    # Get current status
                    status = monitor.get_pair_status(symbol1, symbol2)
                    if status:
                        logger.info(f"\nPair: {symbol1}-{symbol2}")
                        logger.info(f"Z-score: {status['zscore']:.2f}")
                        logger.info(f"Spread: {status['spread']:.4f}")
                        logger.info(f"Signal: {status['signal_type']}")
                        logger.info(f"Timestamp: {status['timestamp']}")
                        logger.info("-" * 50)
                except Exception as e:
                    logger.error(f"Error processing pair {symbol1}-{symbol2}: {str(e)}")
            
            # Wait for 15 minutes before next update
            logger.info("Waiting for next update cycle...")
            time.sleep(15 * 60)  # 15 minutes
            
    except KeyboardInterrupt:
        logger.info("Stopping Z-score monitoring...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main() 