import os
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DataCache:
    """
    Cache historical market data to avoid repeated API calls.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize data cache.
        
        Args:
            cache_dir (str): Directory to store cached data
        """
        if cache_dir is None:
            # Default to 'data' directory in project root
            self.cache_dir = os.path.join(Path(__file__).parent.parent.parent, 'data')
        else:
            self.cache_dir = cache_dir
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Data cache initialized at {self.cache_dir}")
    
    def get_cache_path(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """
        Get path to cached data file.
        
        Args:
            symbol (str): Trading pair
            timeframe (str): Data timeframe (e.g., '1h', '1d')
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            str: Path to cached data file
        """
        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
        return os.path.join(self.cache_dir, filename)
    
    def is_cached(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> bool:
        """
        Check if data is available in cache.
        
        Args:
            symbol (str): Trading pair
            timeframe (str): Data timeframe (e.g., '1h', '1d')
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            bool: True if data is cached, False otherwise
        """
        cache_path = self.get_cache_path(symbol, timeframe, start_date, end_date)
        return os.path.exists(cache_path)
    
    def save_to_cache(self, 
                     symbol: str, 
                     timeframe: str, 
                     start_date: str, 
                     end_date: str, 
                     data: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Args:
            symbol (str): Trading pair
            timeframe (str): Data timeframe (e.g., '1h', '1d')
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            data (pd.DataFrame): Data to cache
        """
        cache_path = self.get_cache_path(symbol, timeframe, start_date, end_date)
        data.to_csv(cache_path)
        logger.info(f"Saved {len(data)} records to {cache_path}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'cached_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'record_count': len(data)
        }
        metadata_path = cache_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def load_from_cache(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load data from cache.
        
        Args:
            symbol (str): Trading pair
            timeframe (str): Data timeframe (e.g., '1h', '1d')
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Cached data
        """
        cache_path = self.get_cache_path(symbol, timeframe, start_date, end_date)
        if not os.path.exists(cache_path):
            logger.warning(f"Cache file not found: {cache_path}")
            return None
            
        try:
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(data)} records from {cache_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
    
    def clear_cache(self, symbol: str = None, timeframe: str = None) -> None:
        """
        Clear cached data.
        
        Args:
            symbol (str): Trading pair (if None, clear all symbols)
            timeframe (str): Data timeframe (if None, clear all timeframes)
        """
        # Build pattern to match files
        pattern = ""
        if symbol:
            pattern += f"{symbol}_"
        if timeframe:
            pattern += f"{timeframe}_"
            
        # Delete matching files
        count = 0
        for filename in os.listdir(self.cache_dir):
            if (pattern == "" or filename.startswith(pattern)) and filename.endswith(('.csv', '.json')):
                os.remove(os.path.join(self.cache_dir, filename))
                count += 1
                
        logger.info(f"Cleared {count} cache files") 