import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.portfolio.portfolio_manager import PortfolioManager

class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.symbols = ['BTCUSDT', 'ETHUSDT']
        self.initial_capital = 10000
        self.commission = 0.0004
        self.portfolio = PortfolioManager(
            symbols=self.symbols,
            initial_capital=self.initial_capital,
            commission=self.commission
        )
    
    def test_initialization(self):
        """Test portfolio initialization."""
        # Check initial state
        self.assertEqual(self.portfolio.initial_capital, self.initial_capital)
        self.assertEqual(self.portfolio.available_capital, self.initial_capital)
        self.assertEqual(self.portfolio.commission, self.commission)
        self.assertEqual(set(self.portfolio.symbols), set(self.symbols))
        
        # Check positions initialization
        for symbol in self.symbols:
            position = self.portfolio.positions[symbol]
            self.assertEqual(position['size'], 0)
            self.assertEqual(position['entry_price'], 0)
            self.assertEqual(position['current_price'], 0)
            self.assertEqual(position['unrealized_pnl'], 0)
            self.assertEqual(position['realized_pnl'], 0)
        
        # Check portfolio weights
        for symbol in self.symbols:
            self.assertEqual(self.portfolio.portfolio_weights[symbol], 0.5)  # Equal weights when no positions
    
    def test_update_prices(self):
        """Test price updates and PnL calculations."""
        # Open a long position first
        size = 0.01  # Smaller position size
        entry_price = 50000
        success = self.portfolio.open_position('BTCUSDT', size, entry_price, datetime.now())
        self.assertTrue(success)
        
        # Update prices with profit
        prices = {'BTCUSDT': 51000, 'ETHUSDT': 3000}
        self.portfolio.update_prices(prices)
        
        # Check current prices and PnL
        self.assertEqual(self.portfolio.positions['BTCUSDT']['current_price'], 51000)
        expected_pnl = (51000 - entry_price) * size  # 10 USDT profit
        self.assertEqual(self.portfolio.positions['BTCUSDT']['unrealized_pnl'], expected_pnl)
    
    def test_position_limits(self):
        """Test position size limits and capital constraints."""
        # Try to open position exceeding max size (20% of capital)
        max_size = self.initial_capital * 0.2 / 50000  # Maximum BTC position size
        success = self.portfolio.open_position('BTCUSDT', max_size + 0.1, 50000, datetime.now())
        self.assertFalse(success)
        
        # Try to open position with insufficient capital
        success = self.portfolio.open_position('BTCUSDT', 100, 50000, datetime.now())
        self.assertFalse(success)
        
        # Open valid position (within 20% limit)
        valid_size = (self.initial_capital * 0.19) / 50000  # 19% of capital
        success = self.portfolio.open_position('BTCUSDT', valid_size, 50000, datetime.now())
        self.assertTrue(success)
    
    def test_position_operations(self):
        """Test opening and closing positions."""
        initial_capital = self.portfolio.available_capital
        
        # Open long position
        size = 0.01  # Smaller position size
        entry_price = 50000
        success = self.portfolio.open_position('BTCUSDT', size, entry_price, datetime.now())
        self.assertTrue(success)
        
        # Check position state
        position = self.portfolio.positions['BTCUSDT']
        self.assertEqual(position['size'], size)
        self.assertEqual(position['entry_price'], entry_price)
        
        # Check capital reduction
        expected_cost = size * entry_price * (1 + self.commission)
        self.assertAlmostEqual(self.portfolio.available_capital, initial_capital - expected_cost, places=4)
        
        # Update current price
        exit_price = 51000
        self.portfolio.update_prices({'BTCUSDT': exit_price, 'ETHUSDT': 3000})
        
        # Close position with profit
        success = self.portfolio.close_position('BTCUSDT', exit_price, datetime.now())
        self.assertTrue(success)
        
        # Check realized PnL
        expected_pnl = (exit_price - entry_price) * size  # Profit
        commission_cost = (size * entry_price * self.commission) + (size * exit_price * self.commission)  # Entry + exit commission
        expected_net_pnl = expected_pnl - commission_cost
        self.assertAlmostEqual(position['realized_pnl'], expected_net_pnl, places=4)
        self.assertEqual(position['size'], 0)
        
        # Check final capital
        expected_final_capital = initial_capital + expected_net_pnl
        self.assertAlmostEqual(self.portfolio.available_capital, expected_final_capital, places=4)
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculations."""
        # Open positions
        self.portfolio.open_position('BTCUSDT', 0.1, 50000, datetime.now())
        self.portfolio.open_position('ETHUSDT', 1.0, 3000, datetime.now())
        
        # Update prices
        prices = {'BTCUSDT': 51000, 'ETHUSDT': 3100}
        self.portfolio.update_prices(prices)
        
        # Get metrics
        metrics = self.portfolio.get_portfolio_metrics()
        
        # Check metrics
        self.assertIn('total_value', metrics)
        self.assertIn('available_capital', metrics)
        self.assertIn('total_pnl', metrics)
        self.assertIn('total_commission', metrics)
        self.assertIn('current_drawdown', metrics)
        self.assertIn('portfolio_weights', metrics)
        self.assertIn('positions', metrics)
    
    def test_trade_history(self):
        """Test trade history recording."""
        # Open and close positions
        entry_price = 50000
        exit_price = 51000
        size = 0.01  # Smaller position size
        
        # Open position
        success = self.portfolio.open_position('BTCUSDT', size, entry_price, datetime.now())
        self.assertTrue(success)
        
        # Update price
        self.portfolio.update_prices({'BTCUSDT': exit_price, 'ETHUSDT': 3000})
        
        # Close position
        success = self.portfolio.close_position('BTCUSDT', exit_price, datetime.now())
        self.assertTrue(success)
        
        # Get trade history
        trade_history = self.portfolio.get_trade_history()
        
        # Check trade history
        self.assertEqual(len(trade_history), 2)  # Open and close trades
        self.assertIn('timestamp', trade_history.columns)
        self.assertIn('symbol', trade_history.columns)
        self.assertIn('action', trade_history.columns)
        self.assertIn('size', trade_history.columns)
        self.assertIn('price', trade_history.columns)
        self.assertIn('cost', trade_history.columns)
        self.assertIn('pnl', trade_history.columns)
        
        # Check trade details
        open_trade = trade_history.iloc[0]
        close_trade = trade_history.iloc[1]
        
        self.assertEqual(open_trade['action'], 'BUY')
        self.assertEqual(open_trade['size'], size)
        self.assertEqual(open_trade['price'], entry_price)
        
        self.assertEqual(close_trade['action'], 'CLOSE')
        self.assertEqual(close_trade['size'], -size)
        self.assertEqual(close_trade['price'], exit_price)
    
    def test_equity_curve(self):
        """Test equity curve tracking."""
        # Open position
        self.portfolio.open_position('BTCUSDT', 0.1, 50000, datetime.now())
        
        # Update prices multiple times
        prices = [
            {'BTCUSDT': 51000, 'ETHUSDT': 3000},
            {'BTCUSDT': 52000, 'ETHUSDT': 3000},
            {'BTCUSDT': 53000, 'ETHUSDT': 3000}
        ]
        
        for price in prices:
            self.portfolio.update_prices(price)
        
        # Get equity curve
        equity_curve = self.portfolio.get_equity_curve()
        
        # Check equity curve
        self.assertEqual(len(equity_curve), 4)  # Initial + 3 updates
        self.assertEqual(equity_curve[0], self.initial_capital)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation and limits."""
        # Open position
        size = 0.01  # Smaller position size
        entry_price = 50000
        success = self.portfolio.open_position('BTCUSDT', size, entry_price, datetime.now())
        self.assertTrue(success)
        
        # Update prices to create drawdown
        prices = [
            {'BTCUSDT': 51000, 'ETHUSDT': 3000},  # Profit
            {'BTCUSDT': 49000, 'ETHUSDT': 3000},  # Loss
            {'BTCUSDT': 45000, 'ETHUSDT': 3000}   # Significant loss
        ]
        
        # Track peak value
        peak_value = self.portfolio.total_capital
        
        for price_data in prices:
            self.portfolio.update_prices(price_data)
            btc_price = price_data['BTCUSDT']
            
            # Calculate position value and PnL
            position_value = size * btc_price
            unrealized_pnl = (btc_price - entry_price) * size
            total_value = self.portfolio.available_capital + unrealized_pnl
            
            # Update peak value
            peak_value = max(peak_value, total_value)
            
            # Calculate expected drawdown
            if btc_price < entry_price:
                expected_drawdown = (peak_value - total_value) / peak_value
                self.assertGreater(self.portfolio.current_drawdown, 0)
                self.assertAlmostEqual(self.portfolio.current_drawdown, expected_drawdown, places=4)
    
    def test_portfolio_weights(self):
        """Test portfolio weight calculations."""
        # Open positions
        self.portfolio.open_position('BTCUSDT', 0.1, 50000, datetime.now())
        self.portfolio.open_position('ETHUSDT', 1.0, 3000, datetime.now())
        
        # Update prices
        prices = {'BTCUSDT': 50000, 'ETHUSDT': 3000}
        self.portfolio.update_prices(prices)
        
        # Check portfolio weights
        weights = self.portfolio.portfolio_weights
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)  # Weights should sum to 1
        self.assertGreater(weights['BTCUSDT'], 0)
        self.assertGreater(weights['ETHUSDT'], 0)

if __name__ == '__main__':
    unittest.main() 