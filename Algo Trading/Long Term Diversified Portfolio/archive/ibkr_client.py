"""
Interactive Brokers REST API client for fetching market data.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

class IBKRClient:
    """Client for interacting with Interactive Brokers REST API."""
    
    def __init__(self):
        self.base_url = 'https://localhost:5000/v1/api'
    
    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None) -> Dict:
        """Make a request to the IBKR API."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.request(method, url, params=params, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, duration: str = '1Y', bar_size: str = '1d') -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol (str): The symbol to get data for
            duration (str): Duration of data to fetch (e.g., '1Y' for 1 year)
            bar_size (str): Size of each bar (e.g., '1d' for daily)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical data or None if request fails
        """
        endpoint = f"market-data/history"
        params = {
            'symbol': symbol,
            'duration': duration,
            'barSize': bar_size
        }
        
        data = self._make_request(endpoint, params=params)
        if not data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
    
    def get_portfolio_historical_data(self, symbols: List[str], duration: str = '1Y') -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols (List[str]): List of symbols to get data for
            duration (str): Duration of data to fetch
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their historical data
        """
        historical_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, duration)
            if data is not None:
                historical_data[symbol] = data
        
        return historical_data 