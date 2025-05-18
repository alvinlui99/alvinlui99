"""
Basic market data fetcher using yfinance.
"""

import yfinance as yf
from typing import Dict, List, Optional
import pandas as pd


class MarketDataFetcher:
    """Simple market data fetcher for getting current prices and basic info."""
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for a list of symbols.
        
        Args:
            symbols (List[str]): List of ticker symbols
            
        Returns:
            Dict[str, float]: Dictionary of symbol to current price
        """
        # Fetch data for all symbols at once
        data = yf.download(symbols, period="1d", progress=False)
        
        # Get the latest closing prices
        prices = {}
        for symbol in symbols:
            if symbol in data['Close']:
                prices[symbol] = data['Close'][symbol].iloc[-1]
            else:
                print(f"Warning: No data found for {symbol}")
        
        return prices
    
    def get_basic_info(self, symbol: str) -> Dict:
        """
        Get basic information about a security.
        
        Args:
            symbol (str): Ticker symbol
            
        Returns:
            Dict: Basic information about the security
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'dividend_yield': info.get('dividendYield', 0),
            }
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return {
                'name': symbol,
                'sector': 'Unknown',
                'market_cap': 0,
                'dividend_yield': 0,
            }


if __name__ == "__main__":
    # Example usage
    fetcher = MarketDataFetcher()
    
    # Example symbols
    symbols = ['AAPL', 'MSFT', 'TLT', 'AGG']
    
    # Get current prices
    prices = fetcher.get_current_prices(symbols)
    print("\nCurrent Prices:")
    for symbol, price in prices.items():
        print(f"{symbol}: ${price:.2f}")
    
    # Get basic info for each symbol
    print("\nBasic Info:")
    for symbol in symbols:
        info = fetcher.get_basic_info(symbol)
        print(f"\n{symbol}:")
        for key, value in info.items():
            print(f"  {key}: {value}") 