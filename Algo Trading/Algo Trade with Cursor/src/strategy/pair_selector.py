from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from scipy import stats
import logging
from binance.um_futures import UMFutures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairSelector:
    def __init__(
        self,
        client: UMFutures,
        min_volume_usd: float = 100_000_000,  # $100M minimum 24h volume
        min_correlation: float = 0.7,
        max_spread: float = 0.001,  # 0.1% maximum spread
        lookback_days: int = 180,  # 6 months
        timeframe: str = '15m',
        max_pairs: int = 10
    ):
        """
        Initialize the pair selector.
        
        Args:
            client: Binance Futures client
            min_volume_usd: Minimum 24h volume in USD
            min_correlation: Minimum correlation threshold
            max_spread: Maximum allowed spread
            lookback_days: Number of days to look back for historical data
            timeframe: Trading timeframe
            max_pairs: Maximum number of pairs to select
        """
        self.client = client
        self.min_volume_usd = min_volume_usd
        self.min_correlation = min_correlation
        self.max_spread = max_spread
        self.lookback_days = lookback_days
        self.timeframe = timeframe
        self.max_pairs = max_pairs
        self.selected_pairs = []
        
    def get_all_symbols(self) -> List[str]:
        """Get all available USDT futures symbols."""
        try:
            exchange_info = self.client.exchange_info()
            symbols = [
                symbol['symbol'] for symbol in exchange_info['symbols']
                if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING'
            ]
            return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {str(e)}")
            return []

    def get_24h_volume(self, symbol: str) -> float:
        """Get 24h volume for a symbol in USD."""
        try:
            # Use ticker endpoint for 24h data
            ticker = self.client.ticker_24hr_price_change(symbol=symbol)
            if isinstance(ticker, list):
                ticker = ticker[0]
            return float(ticker['quoteVolume'])  # Volume in USDT
        except Exception as e:
            logger.error(f"Error fetching volume for {symbol}: {str(e)}")
            return 0.0

    def get_spread(self, symbol: str) -> float:
        """Get current spread for a symbol."""
        try:
            # Use book ticker for best bid/ask
            book = self.client.book_ticker(symbol=symbol)
            if isinstance(book, list):
                book = book[0]
            best_ask = float(book['askPrice'])
            best_bid = float(book['bidPrice'])
            return (best_ask - best_bid) / best_bid
        except Exception as e:
            logger.error(f"Error fetching spread for {symbol}: {str(e)}")
            return float('inf')

    def calculate_correlation(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix for all pairs."""
        # Extract closing prices
        closes = pd.DataFrame({
            symbol: data['Close'] 
            for symbol, data in price_data.items()
        })
        
        # Calculate correlation matrix
        return closes.corr()

    def test_cointegration(self, price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, float]]:
        """Test cointegration between all pairs."""
        cointegrated_pairs = []
        
        # Extract closing prices and handle missing/invalid data
        closes = pd.DataFrame({
            symbol: data['Close'] 
            for symbol, data in price_data.items()
        })
        
        # Remove any columns with NaN or infinite values
        closes = closes.replace([np.inf, -np.inf], np.nan)
        closes = closes.dropna(axis=1)
        
        if closes.empty:
            logger.warning("No valid price data after cleaning")
            return []
        
        # Test each pair
        symbols = list(closes.columns)
        logger.info(f"Testing cointegration for {len(symbols)} symbols")
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                try:
                    # Get price series for the pair
                    price1 = closes[symbols[i]].values
                    price2 = closes[symbols[j]].values
                    
                    # Skip if either series contains invalid values
                    if np.any(np.isnan(price1)) or np.any(np.isnan(price2)):
                        continue
                    
                    # Perform Engle-Granger test
                    _, pvalue, _ = coint(
                        price1,
                        price2,
                        trend='c'
                    )
                    
                    if pvalue < 0.05:  # Significant cointegration
                        cointegrated_pairs.append((symbols[i], symbols[j], pvalue))
                        logger.info(f"Found cointegrated pair: {symbols[i]}-{symbols[j]} (p-value: {pvalue:.4f})")
                except Exception as e:
                    logger.debug(f"Error testing cointegration for {symbols[i]}-{symbols[j]}: {str(e)}")
                    continue
        
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
        return sorted(cointegrated_pairs, key=lambda x: x[2])  # Sort by p-value

    def select_pairs(self) -> List[str]:
        """
        Select trading pairs based on volume, spread, correlation, and cointegration.
        
        Returns:
            List of selected trading pairs
        """
        logger.info("Starting pair selection process...")
        
        # 1. Get all available symbols
        all_symbols = self.get_all_symbols()
        logger.info(f"Found {len(all_symbols)} available symbols")
        
        # 2. Filter by volume
        volume_filtered = []
        for symbol in all_symbols:
            volume = self.get_24h_volume(symbol)
            if volume >= self.min_volume_usd:
                volume_filtered.append(symbol)
        logger.info(f"Volume filter: {len(volume_filtered)} symbols remaining")
        
        # 3. Filter by spread
        spread_filtered = []
        for symbol in volume_filtered:
            spread = self.get_spread(symbol)
            if spread <= self.max_spread:
                spread_filtered.append(symbol)
        logger.info(f"Spread filter: {len(spread_filtered)} symbols remaining")
        
        # 4. Get historical data for remaining symbols
        price_data = {}
        for symbol in spread_filtered:
            try:
                # Calculate number of 15m candles needed
                # Binance has a limit of 1000 candles per request
                candles_needed = min(1000, 24*60//15*self.lookback_days)
                
                klines = self.client.klines(
                    symbol=symbol,
                    interval=self.timeframe,
                    limit=candles_needed
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                ])
                
                # Convert types
                df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                
                # Remove any rows with NaN or infinite values
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna()
                
                if len(df) >= candles_needed * 0.9:  # Require at least 90% of the data
                    price_data[symbol] = df
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)}/{candles_needed} candles")
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
                continue
        
        if not price_data:
            logger.warning("No valid price data found for any symbols")
            return []
        
        # 5. Calculate correlations
        correlation_matrix = self.calculate_correlation(price_data)
        
        # 6. Test cointegration
        cointegrated_pairs = self.test_cointegration(price_data)
        
        # 7. Select final pairs
        selected_pairs = []
        for pair in cointegrated_pairs:
            symbol1, symbol2, _ = pair
            try:
                correlation = correlation_matrix.loc[symbol1, symbol2]
                if correlation >= self.min_correlation:
                    if symbol1 not in selected_pairs:
                        selected_pairs.append(symbol1)
                    if symbol2 not in selected_pairs:
                        selected_pairs.append(symbol2)
                    if len(selected_pairs) >= self.max_pairs:
                        break
            except KeyError:
                logger.warning(f"Missing correlation data for pair {symbol1}-{symbol2}")
                continue
        
        self.selected_pairs = selected_pairs
        logger.info(f"Selected {len(selected_pairs)} pairs: {selected_pairs}")
        return selected_pairs

    def get_pair_metrics(self) -> Dict[str, Dict]:
        """
        Get detailed metrics for selected pairs.
        
        Returns:
            Dictionary containing metrics for each pair
        """
        metrics = {}
        
        for symbol in self.selected_pairs:
            metrics[symbol] = {
                'volume_24h': self.get_24h_volume(symbol),
                'spread': self.get_spread(symbol),
                'price': float(self.client.ticker_price(symbol=symbol)['price'])
            }
        
        return metrics 