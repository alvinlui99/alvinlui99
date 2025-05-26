import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from binance.um_futures import UMFutures
from dotenv import load_dotenv

class BinanceDataCollector:
    def __init__(self):
        load_dotenv()
        self.client = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET')
        )
        
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
            # Handle days_back parameter
            if days_back is not None:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
                end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
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
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
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
        interval: str = '1h',
        days_back: int = 30
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
        data_dict = {}
        for symbol in symbols:
            df = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                days_back=days_back
            )
            if not df.empty:
                data_dict[symbol] = df
                
        return data_dict

    def get_funding_rate(self, symbol: str, limit: int = 30) -> pd.DataFrame:
        """
        Fetch historical funding rates for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of records to fetch
            
        Returns:
            DataFrame with funding rate history
        """
        try:
            funding_rates = self.client.funding_rate(symbol=symbol, limit=limit)
            df = pd.DataFrame(funding_rates)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            return df[['fundingTime', 'fundingRate']]
        except Exception as e:
            print(f"Error fetching funding rates for {symbol}: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    collector = BinanceDataCollector()
    
    # Test with BTC/USDT and ETH/USDT futures
    symbols = ['BTCUSDT', 'ETHUSDT']
    data = collector.get_multiple_symbols_data(symbols, interval='1h', days_back=7)
    
    # Print first few rows of each dataset
    for symbol, df in data.items():
        print(f"\nData for {symbol}:")
        print(df.head())
        
    # Get funding rates for BTC
    funding_rates = collector.get_funding_rate('BTCUSDT')
    print("\nFunding rates for BTCUSDT:")
    print(funding_rates.head())
