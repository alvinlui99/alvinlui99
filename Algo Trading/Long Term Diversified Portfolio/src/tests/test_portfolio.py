import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.core.portfolio import Portfolio, StockMetrics, PortfolioMetrics

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.portfolio = Portfolio(initial_capital=1_000_000)
        
    def test_initialization(self):
        """Test portfolio initialization with correct initial values."""
        self.assertEqual(self.portfolio.portfolio_value, 1_000_000)
        self.assertEqual(self.portfolio.asset_weights['equity'], 0.65)
        self.assertEqual(self.portfolio.asset_weights['fixed_income'], 0.30)
        self.assertEqual(self.portfolio.asset_weights['cash'], 0.05)
        
        # Test sector weights sum to 1
        total_sector_weight = sum(self.portfolio.sector_weights.values())
        self.assertAlmostEqual(total_sector_weight, 1.0, places=2)
        
    @patch('yfinance.Ticker')
    def test_get_stock_info(self, mock_ticker):
        """Test getting stock information."""
        # Mock the ticker info
        mock_info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 2_000_000_000_000,
            'profitMargins': 0.25,
            'debtToEquity': 1.5,
            'returnOnInvestedCapital': 0.15,
            'dividendYield': 0.02,
            'beta': 1.2,
            'currentPrice': 150.0
        }
        mock_ticker.return_value.info = mock_info
        
        stock_info = self.portfolio.get_stock_info('AAPL')
        
        self.assertIsNotNone(stock_info)
        self.assertEqual(stock_info.symbol, 'AAPL')
        self.assertEqual(stock_info.name, 'Apple Inc.')
        self.assertEqual(stock_info.sector, 'Technology')
        self.assertEqual(stock_info.market_cap, 2_000_000_000_000)
        self.assertEqual(stock_info.profit_margin, 0.25)
        self.assertEqual(stock_info.debt_to_equity, 1.5)
        self.assertEqual(stock_info.roic, 0.15)
        self.assertEqual(stock_info.dividend_yield, 0.02)
        self.assertEqual(stock_info.beta, 1.2)
        self.assertEqual(stock_info.current_price, 150.0)
        
    @patch('yfinance.Ticker')
    def test_get_historical_data(self, mock_ticker):
        """Test getting historical data for a stock."""
        # Mock historical data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Open': [99, 100, 101, 102, 103],
            'High': [101, 102, 103, 104, 105],
            'Low': [98, 99, 100, 101, 102],
            'Volume': [1000000] * 5
        }, index=pd.date_range(start='2023-01-01', periods=5))
        
        mock_ticker.return_value.history.return_value = mock_data
        
        historical_data = self.portfolio.get_historical_data('AAPL')
        
        self.assertIsNotNone(historical_data)
        self.assertEqual(len(historical_data), 5)
        self.assertTrue('Close' in historical_data.columns)
        
    def test_calculate_stock_score(self):
        """Test stock score calculation."""
        stock = StockMetrics(
            symbol='AAPL',
            name='Apple Inc.',
            sector='technology',
            market_cap=2_000_000_000_000,
            profit_margin=0.25,
            debt_to_equity=1.5,
            roic=0.15,
            dividend_yield=0.02,
            beta=1.2,
            current_price=150.0
        )
        
        score = self.portfolio.calculate_stock_score(stock, 'technology')
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        
    @patch('yfinance.Ticker')
    def test_get_recommended_stocks(self, mock_ticker):
        """Test getting recommended stocks for a sector."""
        # Mock stock info
        mock_info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 2_000_000_000_000,
            'profitMargins': 0.25,
            'debtToEquity': 1.5,
            'returnOnInvestedCapital': 0.15,
            'dividendYield': 0.02,
            'beta': 1.2,
            'currentPrice': 150.0
        }
        mock_ticker.return_value.info = mock_info
        
        recommended_stocks = self.portfolio.get_recommended_stocks('technology', 100000)
        
        self.assertIsInstance(recommended_stocks, list)
        if recommended_stocks:
            self.assertIsInstance(recommended_stocks[0], dict)
            self.assertIn('symbol', recommended_stocks[0])
            self.assertIn('weight', recommended_stocks[0])
            self.assertIn('amount', recommended_stocks[0])
            
    def test_get_target_holdings(self):
        """Test getting target holdings for the portfolio."""
        holdings = self.portfolio.get_target_holdings()
        
        self.assertIsInstance(holdings, dict)
        self.assertIn('equity', holdings)
        self.assertIn('fixed_income', holdings)
        self.assertIn('cash', holdings)
        
        # Test total weights sum to 1
        total_weight = (
            sum(holding['weight'] for holding in holdings['equity'].values()) +
            sum(holding['weight'] for holding in holdings['fixed_income'].values()) +
            holdings['cash']['USD']['weight']
        )
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
    @patch('yfinance.Ticker')
    def test_calculate_portfolio_metrics(self, mock_ticker):
        """Test portfolio metrics calculation."""
        # Mock historical data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Open': [99, 100, 101, 102, 103],
            'High': [101, 102, 103, 104, 105],
            'Low': [98, 99, 100, 101, 102],
            'Volume': [1000000] * 5
        }, index=pd.date_range(start='2023-01-01', periods=5))
        
        mock_ticker.return_value.history.return_value = mock_data
        
        holdings = self.portfolio.get_target_holdings()
        metrics = self.portfolio.calculate_portfolio_metrics(holdings)
        
        self.assertIsInstance(metrics, PortfolioMetrics)
        self.assertEqual(metrics.total_value, 1_000_000)
        self.assertAlmostEqual(metrics.equity_weight + metrics.fixed_income_weight + metrics.cash_weight, 1.0, places=2)
        
    def test_export_to_csv(self):
        """Test exporting portfolio data to CSV."""
        holdings = self.portfolio.get_target_holdings()
        metrics = self.portfolio.calculate_portfolio_metrics(holdings)
        
        # Test export with default directory
        self.portfolio.export_to_csv(holdings, metrics)
        
        # Test export with custom directory
        custom_dir = "test_output"
        self.portfolio.export_to_csv(holdings, metrics, output_dir=custom_dir)
        
        # Clean up test directory
        import shutil
        shutil.rmtree(custom_dir, ignore_errors=True)

if __name__ == '__main__':
    unittest.main() 