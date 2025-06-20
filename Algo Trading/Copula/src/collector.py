import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from binance.um_futures import UMFutures
from pybit.unified_trading import HTTP

from config import Config

class BinanceDataCollector:
    def __init__(self):
        self.client = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET')
        )
        self.config = Config()
        self.data_cache = {}
        
    def get_historical_klines(
        self,
        symbol: str,
        interval: str = '1h',
        start_str: Optional[str] = None,
        end_str: Optional[str] = None,
        days_back: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical klines (candlestick data) from Binance Futures.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_str: Start time in string format
            end_str: End time in string format
            days_back: Number of days of historical data to fetch
            limit: Number of records to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            if days_back is None:
                days_back = self.config.lookback_days
            if interval is None:
                interval = self.config.interval

            if end_str is None:
                end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                
            end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
            
            if start_str is None:                
                start_time = end_time - timedelta(days=days_back)
                start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
            
            all_klines = []
            current_time = int(datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
            end_time = int(datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
            
            while current_time < end_time:
                if interval.endswith('m'):
                    chunk_days = 7
                elif interval.endswith('h'):
                    chunk_days = 30
                else:
                    chunk_days = 365
                
                chunk_end = min(
                    current_time + (chunk_days * 24 * 60 * 60 * 1000),
                    end_time
                )
                
                klines = self.client.klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_time,
                    endTime=chunk_end,
                    limit=limit
                )
                
                all_klines.extend(klines)

                if interval == '1h':
                    current_time = chunk_end + (60 * 60 * 1000)
                elif interval == '15m':
                    current_time = chunk_end + (15 * 60 * 1000)
                else:
                    current_time = chunk_end
                    print(f"WARNING: Check interval in get_historical_klines")
            
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_symbols_data(
        self,
        symbols: List[str],
        start_str: str = None,
        end_str: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols from Binance Futures.
        
        Args:
            symbols: List of trading pair symbols
            interval: Kline interval
            days_back: Number of days of historical data to fetch
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """

        for symbol in symbols:
            df = self.get_historical_klines(
                symbol=symbol,
                start_str=start_str,
                end_str=end_str
            )
            if not df.empty:
                self.data_cache[symbol] = df
                
        return self.data_cache
    
class BybitDataCollector:
    def __init__(self):
        self.client = HTTP(testnet=True)
        self.config = Config()
        self.data_cache = {}

    def get_historical_klines(
            self,
            symbol: str,
            interval: str = None,
            start_str: Optional[str] = None,
            end_str: Optional[str] = None,
            days_back: Optional[int] = None,
            limit: int = 1000
        ) -> pd.DataFrame:
        if days_back is None:
            days_back = self.config.lookback_days
        if interval is None:
            interval = self.config.interval
        if end_str is None:
            end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                
        end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
        
        if start_str is None:                
            start_time = end_time - timedelta(days=days_back)
            start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
        
        all_klines = []
        current_time = int(datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
        end_time = int(datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
        
        while current_time < end_time:
            chunk_days = 30
            
            chunk_end = min(
                current_time + (chunk_days * 24 * 60 * 60 * 1000),
                end_time
            )
            
            klines = self.client.get_kline(
                category='linear',
                symbol=symbol,
                interval=interval,
                start=current_time,
                end=chunk_end,
                limit=limit
            )['result']['list']
            
            all_klines.extend(klines)

            current_time = chunk_end
        
        df = pd.DataFrame(all_klines, columns=['timestamp',
                                               'open',
                                               'high',
                                               'low',
                                               'close',
                                               'volume',
                                               'turnover'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
        
        # Sort by timestamp
        df = df.sort_values(by='timestamp').reset_index()

        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def get_multiple_symbols_data(
            self,
            symbols: List[str],
            start_str: str = None,
            end_str: str = None
        ) -> Dict[str, pd.DataFrame]:
        for symbol in symbols:
            df = self.get_historical_klines(
                symbol=symbol,
                start_str=start_str,
                end_str=end_str
            )
            if not df.empty:
                self.data_cache[symbol] = df
                
        return self.data_cache