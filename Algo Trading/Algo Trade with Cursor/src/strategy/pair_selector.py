from typing import List, Dict, Tuple
import pandas as pd
from data.collector import BinanceDataCollector
from data.processor import DataProcessor
from utils.indicators import PairIndicators

class PairSelector:
    def __init__(self, min_correlation: float = 0.8, min_data_points: int = 100):
        self.min_correlation = min_correlation
        self.min_data_points = min_data_points
        self.collector = BinanceDataCollector()
        self.processor = DataProcessor()
        
    def analyze_pair(
        self,
        symbol1: str,
        symbol2: str,
        interval: str = '1h',
        days_back: int = 30
    ) -> Dict:
        """
        Analyze a pair of symbols for potential trading.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            interval: Data interval
            days_back: Number of days of historical data
            
        Returns:
            Dictionary with pair analysis results
        """
        # Get and process data
        df1 = self.collector.get_historical_klines(symbol1, interval=interval, days_back=days_back)
        df2 = self.collector.get_historical_klines(symbol2, interval=interval, days_back=days_back)
        
        if len(df1) < self.min_data_points or len(df2) < self.min_data_points:
            return None
            
        # Process individual assets
        df1 = self.processor.process_single_asset(df1)
        df2 = self.processor.process_single_asset(df2)
        
        # Calculate pair metrics
        correlation = PairIndicators.calculate_correlation(df1, df2)
        pair_zscore = PairIndicators.calculate_pair_zscore(df1, df2)
        spread_vol = PairIndicators.calculate_spread_volatility(df1, df2)
        
        # Calculate average correlation
        avg_correlation = correlation.mean()
        
        # Return analysis results
        return {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'correlation': avg_correlation,
            'spread_volatility': spread_vol.mean(),
            'zscore_mean': pair_zscore.mean(),
            'zscore_std': pair_zscore.std(),
            'data_points': len(df1)
        }
    
    def find_potential_pairs(
        self,
        symbols: List[str],
        interval: str = '1h',
        days_back: int = 30
    ) -> List[Dict]:
        """
        Find potential trading pairs from a list of symbols.
        
        Args:
            symbols: List of symbols to analyze
            interval: Data interval
            days_back: Number of days of historical data
            
        Returns:
            List of dictionaries with pair analysis results
        """
        potential_pairs = []
        
        # Analyze each possible pair
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                result = self.analyze_pair(symbols[i], symbols[j], interval, days_back)
                if result and result['correlation'] >= self.min_correlation:
                    potential_pairs.append(result)
        
        # Sort by correlation
        potential_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        return potential_pairs

if __name__ == "__main__":
    # Example usage
    selector = PairSelector(min_correlation=0.8)
    
    # Test with some common pairs
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'AVAXUSDT']
    potential_pairs = selector.find_potential_pairs(symbols, interval='1h', days_back=7)
    
    # Print results
    print("\nPotential trading pairs:")
    for pair in potential_pairs:
        print(f"\n{pair['symbol1']} - {pair['symbol2']}")
        print(f"Correlation: {pair['correlation']:.3f}")
        print(f"Spread Volatility: {pair['spread_volatility']:.3f}")
        print(f"Z-score Mean: {pair['zscore_mean']:.3f}")
        print(f"Z-score Std: {pair['zscore_std']:.3f}")