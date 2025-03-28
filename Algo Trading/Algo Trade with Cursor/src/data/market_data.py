from binance.um_futures import UMFutures
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, client: UMFutures, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize MarketData class for fetching market data from Binance Futures.
        
        Args:
            client (UMFutures): Initialized Binance Futures client
            symbols (List[str]): List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
        """
        self.client = client
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def fetch_historical_data(self) -> dict:
        """
        Fetch historical market data for the specified symbols.
        
        Returns:
            dict: Dictionary containing historical data for each symbol
        """
        for symbol in self.symbols:
            try:
                logger.info(f"Fetching historical data for {symbol}")
                
                # Convert dates to timestamps
                start_ts = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp() * 1000)
                end_ts = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp() * 1000)
                
                # Fetch klines (candlestick) data
                klines = self.client.klines(
                    symbol=symbol,
                    interval='1h',  # 1-hour candles
                    startTime=start_ts,
                    endTime=end_ts,
                    limit=1000  # Maximum number of candles
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    klines,
                    columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore']
                )
                
                # Convert string values to float
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Keep only relevant columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                self.data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return self.data

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a given symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Latest price
        """
        try:
            ticker = self.client.ticker_price(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
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