from binance.um_futures import UMFutures
import pandas as pd
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import numpy as np
from .data_cache import DataCache
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, client: UMFutures, symbols: List[str], start_date: str, end_date: str, timeframe: str = '1h', use_cache: bool = True, use_csv: bool = False, csv_dir: str = 'data'):
        """
        Initialize market data collector.
        
        Args:
            client: Binance client
            symbols (List[str]): List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            use_cache (bool): Whether to use data cache
            use_csv (bool): Whether to use CSV files instead of API
            csv_dir (str): Directory containing CSV files
        """
        self.client = client
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.use_cache = use_cache
        self.use_csv = use_csv
        self.csv_dir = csv_dir
        
        # Initialize data cache
        self.cache = DataCache() if use_cache else None

    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all symbols.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of historical data for each symbol
        """
        historical_data = {}
        
        for symbol in self.symbols:
            if self.use_csv:
                # Try to load from CSV file
                csv_path = os.path.join(self.csv_dir, f"{symbol}_{self.timeframe}.csv")
                if os.path.exists(csv_path):
                    logger.info(f"Loading data for {symbol} from CSV file: {csv_path}")
                    try:
                        data = pd.read_csv(csv_path)
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data.set_index('timestamp', inplace=True)
                        historical_data[symbol] = data
                        continue
                    except Exception as e:
                        logger.error(f"Error loading CSV for {symbol}: {str(e)}")
                else:
                    logger.warning(f"CSV file not found for {symbol}: {csv_path}")
            
            if self.use_cache and self.cache.is_cached(symbol, self.timeframe, self.start_date, self.end_date):
                # Load from cache if available
                data = self.cache.load_from_cache(symbol, self.timeframe, self.start_date, self.end_date)
                if data is not None:
                    historical_data[symbol] = data
                    continue
            
            # Fetch from API if not in cache or cache loading failed
            logger.info(f"Fetching historical data for {symbol} from {self.start_date} to {self.end_date}")
            
            try:
                # Convert dates to timestamps (milliseconds)
                start_timestamp = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp() * 1000)
                end_timestamp = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp() * 1000)
                
                # Map timeframe to Binance interval
                interval_map = {
                    '1m': '1m',
                    '5m': '5m',
                    '15m': '15m',
                    '30m': '30m',
                    '1h': '1h',
                    '2h': '2h',
                    '4h': '4h',
                    '6h': '6h',
                    '8h': '8h',
                    '12h': '12h',
                    '1d': '1d',
                    '3d': '3d',
                    '1w': '1w',
                    '1M': '1M'
                }
                
                binance_interval = interval_map.get(self.timeframe, '1h')
                
                # Fetch data from Binance
                klines = self.client.klines(
                    symbol=symbol,
                    interval=binance_interval,
                    startTime=start_timestamp,
                    endTime=end_timestamp,
                    limit=1000
                )
                
                # Convert to DataFrame
                data = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # Convert timestamp to datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
                
                # Convert string values to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    data[col] = data[col].astype(float)
                
                # Set index
                data.set_index('timestamp', inplace=True)
                
                # Save to CSV if use_csv is True
                if self.use_csv:
                    csv_path = os.path.join(self.csv_dir, f"{symbol}_{self.timeframe}.csv")
                    data.to_csv(csv_path)
                    logger.info(f"Saved data for {symbol} to CSV: {csv_path}")
                
                # Save to cache if enabled
                if self.use_cache:
                    self.cache.save_to_cache(symbol, self.timeframe, self.start_date, self.end_date, data)
                
                historical_data[symbol] = data
                
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        
        return historical_data

    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price for a symbol.
        
        Args:
            symbol (str): Trading pair
            
        Returns:
            float: Latest price
        """
        try:
            ticker = self.client.ticker_price(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None

    def get_volume(self, symbol: str) -> float:
        """
        Get the latest volume for a given symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Latest volume
        """
        try:
            ticker = self.client.ticker_24hr(symbol=symbol)
            return float(ticker['volume'])
        except Exception as e:
            logger.error(f"Error fetching volume for {symbol}: {str(e)}")
            return None 