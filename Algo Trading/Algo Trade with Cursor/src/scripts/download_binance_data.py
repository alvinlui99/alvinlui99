import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from binance.um_futures import UMFutures
from dotenv import load_dotenv
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_klines(symbol: str, interval: str, start_time: int, end_time: int, client: UMFutures) -> pd.DataFrame:
    """
    Download klines data from Binance API.
    
    Args:
        symbol: Trading symbol
        interval: Time interval (e.g., '4h')
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        client: Binance client
        
    Returns:
        pd.DataFrame: Klines data
    """
    try:
        all_klines = []
        current_start = start_time
        
        # Keep fetching until we reach the end time
        while current_start < end_time:
            # Get klines data with pagination
            klines = client.klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_time,
                limit=1000  # Maximum allowed by API
            )
            
            if not klines:
                break
                
            all_klines.extend(klines)
            
            # Update start time for next batch
            current_start = klines[-1][0] + 1  # Add 1ms to avoid overlap
            
            # Add a small delay to avoid rate limits
            time.sleep(0.1)
        
        if not all_klines:
            logger.warning(f"No data downloaded for {symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'trades', 'taker_buy_base', 
                          'taker_buy_quote']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {str(e)}")
        return None

def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
        'LINKUSDT', 'NEARUSDT', 'WIFUSDT', 'AVAXUSDT', '1000SHIBUSDT',
        'DOGEUSDT', '1000PEPEUSDT', 'WLDUSDT'
    ]
    interval = '4h'
    data_dir = 'data'
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize Binance client
    client = UMFutures()
    
    # Calculate time range (3 months)
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=360)).timestamp() * 1000)
    
    logger.info(f"Downloading data from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
    
    # Download data for each symbol
    for symbol in symbols:
        try:
            logger.info(f"Downloading data for {symbol}...")
            
            # Download data
            df = download_klines(symbol, interval, start_time, end_time, client)
            if df is None or df.empty:
                logger.warning(f"No data downloaded for {symbol}")
                continue
                
            # Save to CSV with raw timestamps
            filename = f"{symbol}_{interval}_{datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d')}_{datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d')}.csv"
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Saved {len(df)} rows to {filename}")
            logger.info(f"Date range: {df['open_time'].min()} to {df['open_time'].max()} (milliseconds)")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
            
    logger.info("Data download completed")

if __name__ == "__main__":
    main() 