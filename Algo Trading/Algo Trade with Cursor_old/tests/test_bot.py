import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance.um_futures import UMFutures
from src.trading.trading_bot import TradingBot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bot():
    # Test pairs
    trading_pairs = ['BTCUSDT']
    initial_capital = 100  # USDT
    
    try:
        # Initialize bot
        bot = TradingBot(trading_pairs, initial_capital)
        
        # Test market data
        btc_price = bot.market_data.get_latest_price('BTCUSDT')
        logger.info(f"Current BTC price: {btc_price}")
        
        # Test account balance
        balance = bot.get_account_balance()
        logger.info(f"Account balance: {balance} USDT")
        
        # Test order placement (commented out for safety)
        # bot.place_order('BTCUSDT', 'BUY', 0.001)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_bot() 