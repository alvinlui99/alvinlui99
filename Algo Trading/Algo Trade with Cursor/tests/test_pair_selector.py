import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.strategy.pair_selector import PairSelector

@pytest.fixture
def mock_client():
    client = Mock()
    
    # Mock exchange info
    client.exchange_info.return_value = {
        'symbols': [
            {'symbol': 'BTCUSDT', 'status': 'TRADING'},
            {'symbol': 'ETHUSDT', 'status': 'TRADING'},
            {'symbol': 'BNBUSDT', 'status': 'TRADING'},
            {'symbol': 'XRPUSDT', 'status': 'TRADING'},
            {'symbol': 'ADAUSDT', 'status': 'TRADING'},
            {'symbol': 'DOGEUSDT', 'status': 'TRADING'},
            {'symbol': 'DOTUSDT', 'status': 'TRADING'},
            {'symbol': 'SOLUSDT', 'status': 'TRADING'},
            {'symbol': 'AVAXUSDT', 'status': 'TRADING'},
            {'symbol': 'MATICUSDT', 'status': 'TRADING'}
        ]
    }
    
    # Mock 24h volume
    client.ticker_24hr.return_value = {
        'quoteVolume': '150000000'  # $150M volume
    }
    
    # Mock order book
    client.depth.return_value = {
        'asks': [['50000', '1']],
        'bids': [['49995', '1']]
    }
    
    # Mock price
    client.ticker_price.return_value = {'price': '50000'}
    
    # Mock klines
    client.klines.return_value = [
        [1625097600000, '50000', '50100', '49900', '50050', '100', 1625097900000, '5000000', 1000, '50', '2500000', '0'],
        [1625097900000, '50050', '50200', '50000', '50150', '120', 1625098200000, '6000000', 1200, '60', '3000000', '0']
    ]
    
    return client

def test_get_all_symbols(mock_client):
    selector = PairSelector(mock_client)
    symbols = selector.get_all_symbols()
    
    assert len(symbols) == 10
    assert all(s.endswith('USDT') for s in symbols)
    assert 'BTCUSDT' in symbols
    assert 'ETHUSDT' in symbols

def test_get_24h_volume(mock_client):
    selector = PairSelector(mock_client)
    volume = selector.get_24h_volume('BTCUSDT')
    
    assert volume == 150000000
    mock_client.ticker_24hr.assert_called_once_with(symbol='BTCUSDT')

def test_get_spread(mock_client):
    selector = PairSelector(mock_client)
    spread = selector.get_spread('BTCUSDT')
    
    # (50000 - 49995) / 49995 ≈ 0.0001
    assert spread == pytest.approx(0.0001)
    mock_client.depth.assert_called_once_with(symbol='BTCUSDT', limit=1)

def test_calculate_correlation():
    # Create sample price data
    price_data = {
        'BTCUSDT': pd.DataFrame({
            'Close': [50000, 50100, 50200, 50300, 50400]
        }),
        'ETHUSDT': pd.DataFrame({
            'Close': [3000, 3010, 3020, 3030, 3040]
        })
    }
    
    selector = PairSelector(Mock())
    correlation = selector.calculate_correlation(price_data)
    
    assert correlation.loc['BTCUSDT', 'ETHUSDT'] > 0.9  # High correlation expected

def test_select_pairs(mock_client):
    selector = PairSelector(mock_client)
    selected_pairs = selector.select_pairs()
    
    assert len(selected_pairs) <= 10  # Should not exceed max_pairs
    assert all(s.endswith('USDT') for s in selected_pairs)
    
    # Check metrics
    metrics = selector.get_pair_metrics()
    assert len(metrics) == len(selected_pairs)
    for symbol in selected_pairs:
        assert 'volume_24h' in metrics[symbol]
        assert 'spread' in metrics[symbol]
        assert 'price' in metrics[symbol]

def test_error_handling(mock_client):
    # Test error in volume fetch
    mock_client.ticker_24hr.side_effect = Exception("API Error")
    selector = PairSelector(mock_client)
    volume = selector.get_24h_volume('BTCUSDT')
    assert volume == 0.0
    
    # Test error in spread fetch
    mock_client.depth.side_effect = Exception("API Error")
    spread = selector.get_spread('BTCUSDT')
    assert spread == float('inf') 