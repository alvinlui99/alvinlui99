from typing import List, Tuple
from binance.um_futures import UMFutures
from binance.exceptions import BinanceAPIException
import pandas as pd
from config import BinanceConfig
import logging
from datetime import datetime

class BinanceService:
    def __init__(self):
        self.client = UMFutures(BinanceConfig.API_KEY, BinanceConfig.API_SECRET, base_url=BinanceConfig.BASE_URL)
        self.client.ping()
        self.logger = logging.getLogger(__name__)

    def get_historical_klines(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch historical klines/candlestick data for a single symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_time: Start datetime
            end_time: End datetime
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            
        Returns:
            DataFrame with historical klines/candlestick data
        """
        try:
            # Convert datetime to milliseconds timestamp if provided
            start_ts = int(start_time.timestamp() * 1000) if start_time else None
            end_ts = int(end_time.timestamp() * 1000) if end_time else None
            klines = self.client.klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df.set_index('timestamp', inplace=True)

            self.logger.info(f"Successfully fetched {symbol} data")
            
            return df[numeric_columns]

        except BinanceAPIException as e:
            self.logger.error(f"Error fetching data from Binance for {symbol}: {str(e)}")
            raise

    def get_historical_klines_multi(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = '1h'
    ) -> dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_historical_klines(symbol, start_time, end_time, interval)
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        return results
    
    def get_current_klines(self, symbol: str, interval: str) -> float:
        """Get current klines for a single symbol"""
        try:
            return float(self.client.klines(symbol=symbol, interval=interval))
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching current klines for {symbol}: {str(e)}")
            raise

    def get_current_klines_multi(self, symbols: list[str], interval: str) -> dict[str, float]:
        """Get current klines for multiple symbols"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_current_klines(symbol, interval)
            except Exception as e:
                self.logger.error(f"Error fetching current klines for {symbol}: {str(e)}")
                continue
        return results
    
    def _split_start_end_time(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = '1h'
    ) -> List[Tuple[datetime, datetime]]:
        # Convert interval to timedelta
        interval_map = {
            '1m': pd.Timedelta(minutes=1),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }

        if interval not in interval_map:
            raise ValueError(f"Interval {interval} not supported")

        interval_td = interval_map[interval]
        
        # Calculate max number of intervals per request (limit=1500)
        max_intervals = 1450
        max_time_span = interval_td * max_intervals
        
        # Initialize result list and current start time
        time_spans = []
        current_start = start_time
        
        while current_start < end_time:
            # Calculate potential end time
            potential_end = current_start + max_time_span
            
            # If potential end exceeds the actual end time, use actual end time
            current_end = min(potential_end, end_time)
            
            # Add the time span tuple
            time_spans.append((current_start, current_end))
            
            # Move start time to next period
            current_start = current_end

        return time_spans

    def _row_estimator(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = '1h'
    ) -> int:
        # Calculate expected number of intervals
        time_diff = end_time - start_time
        hours_diff = time_diff.total_seconds() / 3600
        
        diff_factor = {
            '1m': 60,
            '15m': 60 * 15,
            '30m': 60 * 30,
            '1h': 60 * 60,
            '4h': 60 * 60 * 4,
            '1d': 60 * 60 * 24
        }

        if interval in diff_factor:
            return int(hours_diff / diff_factor[interval]) + 1
        else:
            raise ValueError(f"Interval {interval} not supported for row estimation")