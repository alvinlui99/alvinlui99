"""
Data fetcher module for retrieving market data from Binance Futures.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import os

from config import BaseConfig, DataConfig, BinanceConfig

class DataFetcher:
    """
    Fetches market data from Binance Futures API.
    """
    
    def __init__(self, client, symbols: List[str], logger=None):
        """
        Initialize the data fetcher.
        
        Args:
            client: Binance Futures client
            symbols: List of trading symbols
            logger: Optional logger instance
        """
        self.client = client
        self.symbols = symbols
        self.logger = logger or logging.getLogger(__name__)
        
    def fetch_klines(self, timeframe: str = DataConfig.DEFAULT_TIMEFRAME, 
                     limit: int = DataConfig.DEFAULT_LIMIT,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch klines data for the specified symbols.
        
        Args:
            timeframe: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit: Number of klines to fetch
            start_time: Optional start time for klines
            end_time: Optional end time for klines
            **kwargs: Additional parameters:
                for_model (bool): If True, will remove Return and Log_Return columns for model compatibility
            
        Returns:
            Dictionary of DataFrames with klines data
        """
        klines_dict = {}
        
        self.logger.info(f"Fetching {timeframe} klines for {len(self.symbols)} symbols")
        
        # Check if timeframe is valid
        if timeframe not in DataConfig.AVAILABLE_TIMEFRAMES:
            self.logger.warning(f"Invalid timeframe: {timeframe}. Using default: {DataConfig.DEFAULT_TIMEFRAME}")
            timeframe = DataConfig.DEFAULT_TIMEFRAME
        
        # Convert times to milliseconds if provided
        start_ms = int(start_time.timestamp() * 1000) if start_time else None
        end_ms = int(end_time.timestamp() * 1000) if end_time else None
        
        for symbol in self.symbols:
            retry_count = 0
            while retry_count <= BinanceConfig.MAX_RETRIES:
                try:
                    # Get klines from Binance
                    klines = self.client.klines(
                        symbol=symbol, 
                        interval=timeframe,
                        limit=limit,
                        startTime=start_ms,
                        endTime=end_ms
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
                    ])
                    
                    # Convert types
                    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
                    df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')
                    
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Clean up data if configured
                    if DataConfig.REMOVE_DUPLICATE_TIMESTAMPS:
                        df = df.drop_duplicates(subset=['Open_time'])
                    
                    # Calculate derived features if configured
                    if DataConfig.CALCULATE_RETURNS:
                        df['Return'] = df['Close'].pct_change()
                    
                    if DataConfig.CALCULATE_LOG_RETURNS:
                        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                    
                    # Optional: Remove return columns for model compatibility
                    if DataConfig.REMOVE_RETURNS_FOR_MODEL and 'for_model' in kwargs and kwargs['for_model']:
                        if 'Return' in df.columns:
                            df = df.drop(['Return'], axis=1)
                        if 'Log_Return' in df.columns:
                            df = df.drop(['Log_Return'], axis=1)
                    
                    # Handle filling missing values if needed
                    if DataConfig.FILL_MISSING_VALUES and df.isnull().any().any():
                        df = df.ffill()  # Forward fill NaN values
                    
                    # Save to CSV if configured
                    if DataConfig.STORE_AS_CSV:
                        csv_filename = DataConfig.CSV_FILENAME_TEMPLATE.format(
                            symbol=symbol,
                            timeframe=timeframe
                        )
                        csv_path = os.path.join(BaseConfig.DATA_DIR, csv_filename)
                        df.to_csv(csv_path, index=False)
                        self.logger.debug(f"Saved {symbol} data to {csv_path}")
                    
                    klines_dict[symbol] = df
                    self.logger.debug(f"Fetched {len(df)} klines for {symbol}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > BinanceConfig.MAX_RETRIES:
                        self.logger.error(f"Failed to fetch klines for {symbol} after {BinanceConfig.MAX_RETRIES} retries: {str(e)}")
                        # Return empty DataFrame for this symbol
                        klines_dict[symbol] = pd.DataFrame()
                    else:
                        wait_time = BinanceConfig.RETRY_DELAY * retry_count
                        self.logger.warning(f"Error fetching klines for {symbol}, retrying in {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
        
        return klines_dict
        
    def fetch_latest_klines(self, timeframe: str = DataConfig.DEFAULT_TIMEFRAME, 
                           lookback_periods: int = BaseConfig.DEFAULT_LOOKBACK) -> Dict[str, pd.DataFrame]:
        """
        Fetch the most recent klines data.
        
        Args:
            timeframe: Kline interval
            lookback_periods: Number of historical klines to fetch
            
        Returns:
            Dictionary of DataFrames with klines data
        """
        # Calculate end time as now
        end_time = datetime.now()
        
        # Calculate start time based on lookback periods and timeframe
        start_time = None
        
        if timeframe == '1m':
            start_time = end_time - timedelta(minutes=lookback_periods)
        elif timeframe == '3m':
            start_time = end_time - timedelta(minutes=3 * lookback_periods)
        elif timeframe == '5m':
            start_time = end_time - timedelta(minutes=5 * lookback_periods)
        elif timeframe == '15m':
            start_time = end_time - timedelta(minutes=15 * lookback_periods)
        elif timeframe == '30m':
            start_time = end_time - timedelta(minutes=30 * lookback_periods)
        elif timeframe == '1h':
            start_time = end_time - timedelta(hours=lookback_periods)
        elif timeframe == '2h':
            start_time = end_time - timedelta(hours=2 * lookback_periods)
        elif timeframe == '4h':
            start_time = end_time - timedelta(hours=4 * lookback_periods)
        elif timeframe == '6h':
            start_time = end_time - timedelta(hours=6 * lookback_periods)
        elif timeframe == '8h':
            start_time = end_time - timedelta(hours=8 * lookback_periods)
        elif timeframe == '12h':
            start_time = end_time - timedelta(hours=12 * lookback_periods)
        elif timeframe == '1d':
            start_time = end_time - timedelta(days=lookback_periods)
        elif timeframe == '3d':
            start_time = end_time - timedelta(days=3 * lookback_periods)
        elif timeframe == '1w':
            start_time = end_time - timedelta(weeks=lookback_periods)
        elif timeframe == '1M':
            start_time = end_time - timedelta(days=30 * lookback_periods)
        else:
            # Default to 7 days for unknown timeframes
            self.logger.warning(f"Unknown timeframe: {timeframe}, defaulting to 7 days lookback")
            start_time = end_time - timedelta(days=7)
            
        return self.fetch_klines(timeframe=timeframe, start_time=start_time, end_time=end_time)
    
    def load_historical_data(self, timeframe: str = DataConfig.DEFAULT_TIMEFRAME) -> Dict[str, pd.DataFrame]:
        """
        Load historical data from CSV files.
        
        Args:
            timeframe: Kline interval
            
        Returns:
            Dictionary of DataFrames with klines data
        """
        klines_dict = {}
        
        self.logger.info(f"Loading historical {timeframe} data for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                csv_filename = DataConfig.CSV_FILENAME_TEMPLATE.format(
                    symbol=symbol,
                    timeframe=timeframe
                )
                csv_path = os.path.join(BaseConfig.DATA_DIR, csv_filename)
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['Open_time'] = pd.to_datetime(df['Open_time'])
                    if 'Close_time' in df.columns:
                        df['Close_time'] = pd.to_datetime(df['Close_time'])
                    
                    klines_dict[symbol] = df
                    self.logger.debug(f"Loaded {len(df)} historical klines for {symbol}")
                else:
                    self.logger.warning(f"No historical data found for {symbol} at {csv_path}")
                    klines_dict[symbol] = pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error loading historical data for {symbol}: {str(e)}")
                klines_dict[symbol] = pd.DataFrame()
        
        return klines_dict 