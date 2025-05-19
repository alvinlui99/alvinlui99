"""
Portfolio analyzer module for target portfolio analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import numpy as np
from .stock_selector import StockSelector


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""
    total_value: float
    equity_weight: float
    fixed_income_weight: float
    cash_weight: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sector_weights: Dict[str, float]
    style_weights: Dict[str, float]


class PortfolioAnalyzer:
    """Analyzes target portfolio composition and risk metrics."""
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.stock_selector = StockSelector()
        
        # Define asset class weights
        self.asset_weights = {
            'equity': 0.65,
            'fixed_income': 0.30,
            'cash': 0.05
        }
        
        # Define sector weights within equity
        self.sector_weights = {
            'technology': 0.28,
            'healthcare': 0.13,
            'financials': 0.12,
            'consumer_discretionary': 0.10,
            'industrials': 0.08,
            'consumer_staples': 0.07,
            'energy': 0.05,
            'materials': 0.05,
            'utilities': 0.03,
            'real_estate': 0.03,
            'communication_services': 0.03
        }
    
    def get_target_holdings(self) -> Dict:
        """
        Get target holdings for the portfolio.
        
        Returns:
            Dict: Target holdings with weights and amounts
        """
        # Calculate asset class allocations
        equity_value = self.portfolio_value * self.asset_weights['equity']
        fixed_income_value = self.portfolio_value * self.asset_weights['fixed_income']
        cash_value = self.portfolio_value * self.asset_weights['cash']
        
        # Get equity holdings
        equity_holdings = {}
        for sector, weight in self.sector_weights.items():
            if sector in ['technology', 'healthcare', 'financials', 'consumer_discretionary']:
                # Direct holdings
                sector_value = equity_value * weight
                stocks = self.stock_selector.get_recommended_stocks(sector, sector_value)
                for stock in stocks:
                    equity_holdings[stock['symbol']] = {
                        'name': stock['name'],
                        'weight': (stock['position_size'] / self.portfolio_value),
                        'amount': stock['position_size'],
                        'shares': stock['shares'],
                        'sector': sector
                    }
            else:
                # ETF holdings
                etf_symbol = self.stock_selector.recommended_etfs.get(sector)
                if etf_symbol:
                    equity_holdings[etf_symbol] = {
                        'name': f"{sector.title()} ETF",
                        'weight': (equity_value * weight) / self.portfolio_value,
                        'amount': equity_value * weight,
                        'sector': sector
                    }
        
        # Add fixed income and cash
        holdings = {
            'equity': equity_holdings,
            'fixed_income': {
                'AGG': {
                    'name': 'iShares Core U.S. Aggregate Bond ETF',
                    'weight': self.asset_weights['fixed_income'] * 0.7,
                    'amount': fixed_income_value * 0.7
                },
                'TLT': {
                    'name': 'iShares 20+ Year Treasury Bond ETF',
                    'weight': self.asset_weights['fixed_income'] * 0.3,
                    'amount': fixed_income_value * 0.3
                }
            },
            'cash': {
                'USD': {
                    'name': 'Cash',
                    'weight': self.asset_weights['cash'],
                    'amount': cash_value
                }
            }
        }
        
        return holdings
    
    def calculate_portfolio_metrics(self, holdings: Dict) -> PortfolioMetrics:
        """
        Calculate portfolio-level metrics.
        
        Args:
            holdings (Dict): Portfolio holdings
            
        Returns:
            PortfolioMetrics: Portfolio metrics
        """
        # Calculate weights
        equity_weight = sum(holding['weight'] for holding in holdings['equity'].values())
        fixed_income_weight = sum(holding['weight'] for holding in holdings['fixed_income'].values())
        cash_weight = holdings['cash']['USD']['weight']
        
        # Calculate sector weights
        sector_weights = {}
        for holding in holdings['equity'].values():
            sector = holding['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + holding['weight']
        
        # Calculate style weights (simplified)
        style_weights = {
            'large_cap': 0.50,  # Assuming 50% of equity is large cap
            'mid_cap': 0.25,    # From IJH
            'small_cap': 0.25   # From IJR
        }
        
        # Expected returns (simplified assumptions)
        expected_returns = {
            'equity': 0.08,  # 8% expected return for equities
            'fixed_income': 0.04,  # 4% expected return for fixed income
            'cash': 0.02  # 2% expected return for cash
        }
        
        # Expected volatility (simplified assumptions)
        expected_volatilities = {
            'equity': 0.15,  # 15% volatility for equities
            'fixed_income': 0.05,  # 5% volatility for fixed income
            'cash': 0.01  # 1% volatility for cash
        }
        
        # Calculate portfolio expected return
        expected_return = (
            equity_weight * expected_returns['equity'] +
            fixed_income_weight * expected_returns['fixed_income'] +
            cash_weight * expected_returns['cash']
        )
        
        # Calculate portfolio expected volatility (simplified)
        expected_volatility = np.sqrt(
            (equity_weight * expected_volatilities['equity'])**2 +
            (fixed_income_weight * expected_volatilities['fixed_income'])**2 +
            (cash_weight * expected_volatilities['cash'])**2
        )
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        sharpe_ratio = (expected_return - 0.02) / expected_volatility
        
        # Estimate maximum drawdown (simplified)
        max_drawdown = expected_volatility * 2.5  # Rough estimate
        
        return PortfolioMetrics(
            total_value=self.portfolio_value,
            equity_weight=equity_weight,
            fixed_income_weight=fixed_income_weight,
            cash_weight=cash_weight,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            sector_weights=sector_weights,
            style_weights=style_weights
        )


if __name__ == "__main__":
    # Example usage
    analyzer = PortfolioAnalyzer(portfolio_value=1_000_000)
    
    # Get target holdings
    holdings = analyzer.get_target_holdings()
    
    # Calculate metrics
    metrics = analyzer.calculate_portfolio_metrics(holdings)
    
    # Print portfolio overview
    print("\nPortfolio Overview:")
    print("------------------")
    print(f"Total Value: ${metrics.total_value:,.2f}")
    print(f"\nAsset Allocation:")
    print(f"Equity: {metrics.equity_weight:.1%}")
    print(f"Fixed Income: {metrics.fixed_income_weight:.1%}")
    print(f"Cash: {metrics.cash_weight:.1%}")
    
    print("\nEquity Holdings:")
    print("---------------")
    for symbol, holding in holdings['equity'].items():
        print(f"\n{symbol} - {holding['name']}")
        print(f"  Weight: {holding['weight']:.1%}")
        print(f"  Amount: ${holding['amount']:,.2f}")
        if 'shares' in holding:
            print(f"  Shares: {holding['shares']}")
    
    print("\nFixed Income Holdings:")
    print("---------------------")
    for symbol, holding in holdings['fixed_income'].items():
        print(f"\n{symbol} - {holding['name']}")
        print(f"  Weight: {holding['weight']:.1%}")
        print(f"  Amount: ${holding['amount']:,.2f}")
    
    print("\nRisk & Return Metrics:")
    print("---------------------")
    print(f"Expected Return: {metrics.expected_return:.1%}")
    print(f"Expected Volatility: {metrics.expected_volatility:.1%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Estimated Max Drawdown: {metrics.max_drawdown:.1%}")
    
    print("\nSector Weights:")
    print("--------------")
    for sector, weight in metrics.sector_weights.items():
        print(f"{sector}: {weight:.1%}")
    
    print("\nStyle Weights:")
    print("-------------")
    for style, weight in metrics.style_weights.items():
        print(f"{style}: {weight:.1%}") 