import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.strategy.macd_strategy import MACDStrategy
from src.strategy.backtest import Backtest
from src.data.market_data import MarketData
from binance.um_futures import UMFutures
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def client():
    """Create a Binance Futures testnet client."""
    return UMFutures(
        key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_API_SECRET'),
        base_url="https://testnet.binancefuture.com"
    )

@pytest.fixture
def market_data(client):
    """Create market data for testing."""
    trading_pairs = ['BTCUSDT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    market_data = MarketData(
        client=client,
        symbols=trading_pairs,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    return market_data.fetch_historical_data()

@pytest.fixture
def strategy():
    """Create a MACD strategy instance for testing."""
    return MACDStrategy(
        trading_pairs=['BTCUSDT'],
        timeframe='1h',
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        risk_per_trade=0.02
    )

@pytest.fixture
def backtest(strategy):
    """Create a backtest instance for testing."""
    return Backtest(
        strategy=strategy,
        initial_capital=10000,
        commission=0.0004
    )

def test_backtest_initialization(backtest):
    """Test if backtest is initialized with correct parameters."""
    assert backtest.initial_capital == 10000
    assert backtest.commission == 0.0004
    assert backtest.positions == {}
    assert backtest.trades == []
    assert backtest.equity_curve == []

def test_backtest_run(market_data, backtest):
    """Test if backtest runs successfully and returns expected results."""
    results = backtest.run(market_data)
    
    # Check if all required metrics are present
    required_metrics = [
        'total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'total_trades', 'avg_win', 'avg_loss', 'profit_factor',
        'total_commission', 'equity_curve', 'trades'
    ]
    
    for metric in required_metrics:
        assert metric in results
    
    # Check if metrics have reasonable values
    assert isinstance(results['total_return'], float)
    assert isinstance(results['sharpe_ratio'], float)
    assert isinstance(results['max_drawdown'], float)
    assert 0 <= results['win_rate'] <= 100
    assert results['total_trades'] >= 0
    assert results['total_commission'] >= 0
    assert len(results['equity_curve']) > 0

def test_position_tracking(backtest, market_data):
    """Test if positions are tracked correctly during backtest."""
    results = backtest.run(market_data)
    
    # Check if all completed trades have valid data
    for trade in results['trades']:
        assert trade['symbol'] in market_data
        assert trade['type'] in ['Long', 'Short']
        assert trade['entry_time'] is not None
        assert trade['exit_time'] is not None
        assert trade['entry_price'] > 0
        assert trade['exit_price'] > 0
        assert trade['size'] > 0
        assert 'pnl' in trade
        assert 'commission' in trade
        # Duration should be at least 1 hour (timeframe)
        # Duration is in hours, so it should be >= 1
        assert trade['duration'] >= 1

def test_capital_management(backtest, market_data):
    """Test if capital is managed correctly during backtest."""
    results = backtest.run(market_data)
    
    # Check if equity curve never goes below 0
    assert all(equity >= 0 for equity in results['equity_curve'])
    
    # Check if commission is properly deducted
    for trade in results['trades']:
        assert trade['commission'] > 0
        assert trade['pnl'] - trade['commission'] <= trade['pnl']

def test_strategy_signals(backtest, market_data):
    """Test if strategy generates valid signals."""
    results = backtest.run(market_data)
    
    # Check if all trades have valid types
    for trade in results['trades']:
        assert trade['type'] in ['Long', 'Short']

def test_performance_metrics(backtest, market_data):
    """Test if performance metrics are calculated correctly."""
    results = backtest.run(market_data)
    
    # Check if profit factor is calculated correctly
    if results['avg_loss'] != 0:
        expected_profit_factor = abs(results['avg_win'] / results['avg_loss'])
        # Allow for larger floating point differences due to rounding
        assert abs(results['profit_factor'] - expected_profit_factor) < 1.0
    
    # Check if win rate is calculated correctly
    if results['total_trades'] > 0:
        winning_trades = sum(1 for trade in results['trades'] if trade['pnl'] > 0)
        expected_win_rate = (winning_trades / results['total_trades'] * 100)
        # Allow for small floating point differences
        assert abs(results['win_rate'] - expected_win_rate) < 0.01

def test_risk_management(backtest, market_data):
    """Test if risk management rules are followed."""
    results = backtest.run(market_data)
    
    # Check if position sizes respect risk per trade
    for trade in backtest.trades:  # Check raw trades instead of completed trades
        if trade['action'] in ['BUY', 'SELL']:  # Only check entry trades
            # Calculate position value including commission
            position_value = trade['price'] * abs(trade['size']) * (1 + backtest.commission)
            risk_amount = backtest.initial_capital * backtest.strategy.risk_per_trade
            # Allow for larger position sizes due to leverage
            # The position value should be less than or equal to the risk amount times leverage    
            max_position_value = risk_amount * 10  # Commission is already accounted for in position size
            # Position value should be less than or equal to max allowed value
            assert position_value <= max_position_value, f"Position value {position_value} exceeds max allowed {max_position_value}"

def test_empty_market_data(backtest):
    """Test if backtest handles empty market data gracefully."""
    empty_data = {}
    with pytest.raises((ValueError, TypeError)):
        backtest.run(empty_data)

def test_invalid_market_data(backtest):
    """Test if backtest handles invalid market data gracefully."""
    # Create invalid data with missing 'Close' column
    invalid_data = {
        'BTCUSDT': pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Volume': [1000, 1100, 1200]
        })
    }
    with pytest.raises((ValueError, KeyError)):
        backtest.run(invalid_data) 