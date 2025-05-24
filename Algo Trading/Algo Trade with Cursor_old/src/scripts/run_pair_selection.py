import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from dotenv import load_dotenv
from binance.um_futures import UMFutures
from src.strategy.pair_selector import PairSelector
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
    
    # Initialize pair selector
    selector = PairSelector(
        client=client,
        min_volume_usd=100_000_000,  # $100M minimum 24h volume
        min_correlation=0.7,
        max_spread=0.001,  # 0.1% maximum spread
        lookback_days=180,  # 6 months
        timeframe='15m',
        max_pairs=10
    )
    
    # Select pairs
    logger.info("Starting pair selection process...")
    selected_pairs = selector.select_pairs()
    
    # Get metrics for selected pairs
    metrics = selector.get_pair_metrics()
    
    # Print results
    logger.info("\nSelected Pairs and Metrics:")
    logger.info("-" * 50)
    for symbol in selected_pairs:
        logger.info(f"\nSymbol: {symbol}")
        logger.info(f"24h Volume: ${metrics[symbol]['volume_24h']:,.2f}")
        logger.info(f"Current Spread: {metrics[symbol]['spread']*100:.4f}%")
        logger.info(f"Current Price: ${metrics[symbol]['price']:,.2f}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main() 