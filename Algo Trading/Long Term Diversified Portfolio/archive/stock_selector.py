"""
Stock selection module for direct holdings using a scoring system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import numpy as np


@dataclass
class ScoringWeights:
    """Weights for different factors in stock scoring."""
    roic_weight: float = 0.25
    profit_margin_weight: float = 0.20
    market_cap_weight: float = 0.15
    debt_to_equity_weight: float = 0.15
    dividend_yield_weight: float = 0.10
    beta_weight: float = 0.15


class StockSelector:
    """Selects stocks based on quality metrics and sector allocation."""
    
    def __init__(self):
        # Define scoring weights for each sector
        self.scoring_weights = {
            'technology': ScoringWeights(
                roic_weight=0.30,
                profit_margin_weight=0.25,
                market_cap_weight=0.20,
                debt_to_equity_weight=0.15,
                beta_weight=0.10
            ),
            'healthcare': ScoringWeights(
                roic_weight=0.25,
                profit_margin_weight=0.25,
                market_cap_weight=0.20,
                debt_to_equity_weight=0.15,
                beta_weight=0.15
            ),
            'financials': ScoringWeights(
                roic_weight=0.20,
                profit_margin_weight=0.15,
                market_cap_weight=0.20,
                debt_to_equity_weight=0.15,
                dividend_yield_weight=0.20,
                beta_weight=0.10
            ),
            'consumer_discretionary': ScoringWeights(
                roic_weight=0.25,
                profit_margin_weight=0.20,
                market_cap_weight=0.20,
                debt_to_equity_weight=0.15,
                beta_weight=0.20
            )
        }
        
        # Define initial stock universe for each sector
        self.stock_universe = {
            'technology': [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AVGO', 'ASML', 'CRM', 'ADBE',
                'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'MU', 'AMAT'
            ],
            'healthcare': [
                'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'LLY', 'DHR',
                'BMY', 'AMGN', 'GILD', 'ISRG', 'VRTX', 'REGN', 'CI'
            ],
            'financials': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
                'V', 'MA', 'C', 'USB', 'PNC', 'TFC', 'MMC'
            ],
            'consumer_discretionary': [
                'AMZN', 'MCD', 'SBUX', 'NKE', 'HD', 'LOW', 'DIS', 'NFLX',
                'TSLA', 'MAR', 'BKNG', 'CMG', 'TGT', 'COST', 'TJX'
            ]
        }
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed information about a stock."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', float('inf')),
                'roic': info.get('returnOnInvestedCapital', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'current_price': info.get('currentPrice', 0)
            }
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return None
    
    def normalize_metric(self, values: List[float], reverse: bool = False) -> List[float]:
        """
        Normalize a list of values to 0-1 range.
        
        Args:
            values (List[float]): List of values to normalize
            reverse (bool): If True, reverse the normalization (for metrics where lower is better)
            
        Returns:
            List[float]: Normalized values
        """
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        if range_val == 0:
            return [0.5] * len(values)
        
        normalized = [(v - min_val) / range_val for v in values]
        return [1 - n for n in normalized] if reverse else normalized
    
    def calculate_stock_score(self, stock: Dict, weights: ScoringWeights) -> float:
        """
        Calculate a composite score for a stock based on multiple factors.
        
        Args:
            stock (Dict): Stock information
            weights (ScoringWeights): Weights for different factors
            
        Returns:
            float: Composite score
        """
        # Handle missing or invalid values
        roic = max(0, stock.get('roic', 0))
        profit_margin = max(0, stock.get('profit_margin', 0))
        market_cap = max(0, stock.get('market_cap', 0))
        debt_to_equity = min(10, max(0, stock.get('debt_to_equity', 10)))  # Cap at 10x
        dividend_yield = max(0, stock.get('dividend_yield', 0))
        beta = min(3, max(0, stock.get('beta', 1.5)))  # Cap at 3
        
        # Calculate score components
        score = (
            weights.roic_weight * roic +
            weights.profit_margin_weight * profit_margin +
            weights.market_cap_weight * (market_cap / 1e12) +  # Normalize market cap to trillions
            weights.debt_to_equity_weight * (1 / (1 + debt_to_equity)) +  # Invert debt-to-equity
            weights.dividend_yield_weight * dividend_yield +
            weights.beta_weight * (1 / beta)  # Invert beta
        )
        
        return score
    
    def get_recommended_stocks(self, sector: str, allocation_amount: float, 
                             max_stocks: int = 5) -> List[Dict]:
        """
        Get recommended stocks for a sector with position sizes.
        
        Args:
            sector (str): Sector to get stocks for
            allocation_amount (float): Total amount to allocate
            max_stocks (int): Maximum number of stocks to recommend
            
        Returns:
            List[Dict]: List of recommended stocks with position sizes
        """
        # Get stock information
        stocks_info = []
        for symbol in self.stock_universe[sector]:
            info = self.get_stock_info(symbol)
            if info:
                stocks_info.append(info)
        
        if not stocks_info:
            return []
        
        # Calculate scores for each stock
        weights = self.scoring_weights[sector]
        for stock in stocks_info:
            stock['score'] = self.calculate_stock_score(stock, weights)
        
        # Sort by score
        stocks_info.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top N stocks
        selected_stocks = stocks_info[:max_stocks]
        
        # Calculate position sizes (equal weight for now)
        position_size = allocation_amount / len(selected_stocks)
        
        # Add position size to each stock
        for stock in selected_stocks:
            stock['position_size'] = position_size
            stock['shares'] = int(position_size / stock['current_price'])
        
        return selected_stocks


if __name__ == "__main__":
    # Example usage
    selector = StockSelector()
    
    # Example allocation amounts (for $1M portfolio, 65% to equities)
    allocation = {
        'technology': 182000,  # 28% of equity allocation
        'healthcare': 84500,   # 13% of equity allocation
        'financials': 78000,   # 12% of equity allocation
        'consumer_discretionary': 65000  # 10% of equity allocation
    }
    
    print("\nRecommended Direct Holdings:")
    print("---------------------------")
    
    for sector, amount in allocation.items():
        print(f"\n{sector.upper()} (${amount:,.2f}):")
        stocks = selector.get_recommended_stocks(sector, amount)
        
        for stock in stocks:
            print(f"\n{stock['symbol']} - {stock['name']}")
            print(f"  Position: ${stock['position_size']:,.2f}")
            print(f"  Shares: {stock['shares']}")
            print(f"  Score: {stock['score']:.3f}")
            print(f"  ROIC: {stock['roic']:.1%}")
            print(f"  Profit Margin: {stock['profit_margin']:.1%}")
            print(f"  Market Cap: ${stock['market_cap']/1e9:.1f}B")
            print(f"  Debt/Equity: {stock['debt_to_equity']:.2f}")
            print(f"  Beta: {stock['beta']:.2f}") 