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
from src.portfolio.portfolio_manager import PortfolioManager
from src.portfolio.portfolio_optimizer import PortfolioOptimizer
from binance.um_futures import UMFutures
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture(scope="session")
def client():
    """Create a Binance Futures testnet client."""
    return UMFutures(
        key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_API_SECRET'),
        base_url="https://testnet.binancefuture.com"
    )

@pytest.fixture(scope="session")
def market_data(client):
    """Create market data for testing."""
    trading_pairs = ['BTCUSDT', 'ETHUSDT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    market_data = MarketData(
        client=client,
        symbols=trading_pairs,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    return market_data.fetch_historical_data()

@pytest.fixture(scope="session")
def strategy():
    """Create a MACD strategy instance for testing."""
    return MACDStrategy(
        trading_pairs=['BTCUSDT', 'ETHUSDT'],
        timeframe='1h',
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        risk_per_trade=0.02
    )

@pytest.fixture(scope="session")
def backtest(strategy):
    """Create a backtest instance for testing."""
    return Backtest(
        strategy=strategy,
        initial_capital=10000,
        commission=0.0004
    )

@pytest.fixture(scope="session")
def portfolio_manager(market_data):
    """Create a portfolio manager instance for testing."""
    return PortfolioManager(
        symbols=list(market_data.keys()),
        initial_capital=10000,
        commission=0.0004
    )

@pytest.fixture(scope="session")
def portfolio_optimizer(market_data):
    """Create a portfolio optimizer instance for testing."""
    # Calculate returns from market data
    returns = pd.DataFrame({
        symbol: df['Close'].pct_change()
        for symbol, df in market_data.items()
    }).dropna()
    
    return PortfolioOptimizer(
        symbols=list(market_data.keys()),
        returns=returns,
        risk_free_rate=0.02
    )

def test_data_infrastructure(market_data):
    """Test if market data infrastructure works correctly."""
    # Check if we have data for all trading pairs
    assert len(market_data) > 0
    assert all(symbol in market_data for symbol in ['BTCUSDT', 'ETHUSDT'])
    
    # Check data structure
    for symbol, df in market_data.items():
        assert isinstance(df, pd.DataFrame)
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        assert len(df) > 0
        assert not df.isnull().any().any()

def test_strategy_infrastructure(strategy, market_data):
    """Test if strategy infrastructure works correctly."""
    # Test indicator calculation
    for symbol, df in market_data.items():
        df_with_indicators = strategy.calculate_indicators(df)
        assert 'macd' in df_with_indicators.columns
        assert 'signal' in df_with_indicators.columns
        assert 'histogram' in df_with_indicators.columns
        assert 'rsi' in df_with_indicators.columns
    
    # Test signal generation
    signals = strategy.generate_signals(market_data)
    assert isinstance(signals, dict)
    assert all(symbol in signals for symbol in market_data.keys())
    
    # Test position size calculation
    for symbol, signal in signals.items():
        if signal['action'] != 'HOLD':
            position_size = strategy.calculate_position_size(symbol, signal, 10000)
            assert position_size >= 0
            assert position_size * signal['price'] <= 10000 * strategy.risk_per_trade * 10

def test_backtest_infrastructure(backtest, market_data):
    """Test if backtest infrastructure works correctly."""
    # Run backtest
    results = backtest.run(market_data)
    
    # Check results structure
    required_metrics = [
        'total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'total_trades', 'avg_win', 'avg_loss', 'profit_factor',
        'total_commission', 'equity_curve', 'trades'
    ]
    assert all(metric in results for metric in required_metrics)
    
    # Check equity curve
    assert len(results['equity_curve']) > 0
    assert all(equity >= 0 for equity in results['equity_curve'])
    
    # Check trades
    if results['total_trades'] > 0:
        for trade in results['trades']:
            assert trade['pnl'] is not None
            assert trade['commission'] >= 0
            assert trade['duration'] >= 0

def test_portfolio_infrastructure(portfolio_manager, portfolio_optimizer, market_data):
    """Test if portfolio infrastructure works correctly."""
    # Test portfolio manager
    assert portfolio_manager.initial_capital == 10000
    assert portfolio_manager.commission == 0.0004
    assert len(portfolio_manager.symbols) == 2  # BTCUSDT and ETHUSDT
    
    # Test position limits
    for symbol in portfolio_manager.symbols:
        assert portfolio_manager.max_position_size[symbol] == 2000  # 20% of initial capital
    
    # Test portfolio optimizer
    assert portfolio_optimizer.risk_free_rate == 0.02
    assert len(portfolio_optimizer.symbols) == 2
    assert portfolio_optimizer.returns.shape[1] == 2  # Returns for both symbols

def test_integration(market_data, strategy, backtest, portfolio_manager, portfolio_optimizer):
    """Test if all components work together correctly."""
    # Run backtest
    results = backtest.run(market_data)
    
    # Test portfolio manager with backtest results
    if results['total_trades'] > 0:
        for trade in results['trades']:
            # Calculate position value
            position_value = abs(trade['size'] * trade['entry_price'])
            max_position_value = portfolio_manager.max_position_size[trade['symbol']]
            
            # Skip trades that exceed position limits
            if position_value > max_position_value:
                continue
            
            # Check if position can be opened
            assert portfolio_manager.can_open_position(
                symbol=trade['symbol'],
                size=trade['size'],
                price=trade['entry_price']
            )
            
            # Update portfolio with trade
            portfolio_manager.open_position(
                symbol=trade['symbol'],
                size=trade['size'],
                price=trade['entry_price'],
                timestamp=trade['entry_time']
            )
    
    # Test portfolio optimization
    weights, portfolio_return, portfolio_risk = portfolio_optimizer.minimum_variance_optimization()
    assert len(weights) == len(portfolio_manager.symbols)
    assert abs(sum(weights) - 1.0) < 1e-6  # Weights sum to 1
    assert all(0 <= w <= 1 for w in weights)  # Weights between 0 and 1 