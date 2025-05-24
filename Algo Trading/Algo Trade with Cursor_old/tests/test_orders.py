import sys
import os
import time
import logging
from datetime import datetime
import math

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.trading_bot import TradingBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_order_placement():
    """Test order placement and balance updates."""
    # Test pairs
    trading_pairs = ['BTCUSDT']
    initial_capital = 1000  # USDT
    leverage = 2  # Set leverage to 2x
    
    try:
        # Initialize bot
        bot = TradingBot(trading_pairs, initial_capital)
        
        # Get initial balance
        initial_balance = bot.get_account_balance()
        logger.info(f"Initial balance: {initial_balance} USDT")
        
        # Set leverage
        logger.info(f"Setting leverage to {leverage}x...")
        bot.set_leverage('BTCUSDT', leverage)
        
        # Get current BTC price
        btc_price = bot.market_data.get_latest_price('BTCUSDT')
        logger.info(f"Current BTC price: {btc_price} USDT")
        
        # Calculate order quantity to meet minimum notional value (100 USDT)
        min_notional = 100  # Minimum order value in USDT
        quantity = math.ceil((min_notional / btc_price) * 1000) / 1000  # Round up to 3 decimal places
        logger.info(f"Calculated order quantity: {quantity} BTC (Value: {quantity * btc_price:.2f} USDT)")
        
        # Place buy order
        logger.info(f"Placing buy order for {quantity} BTC...")
        buy_order = bot.place_order('BTCUSDT', 'BUY', quantity)
        
        if buy_order:
            logger.info("Buy order placed successfully")
            
            # Wait a few seconds
            time.sleep(5)
            
            # Get updated balance
            updated_balance = bot.get_account_balance()
            logger.info(f"Updated balance: {updated_balance} USDT")
            
            # Place sell order to close position
            logger.info(f"Placing sell order for {quantity} BTC...")
            sell_order = bot.place_order('BTCUSDT', 'SELL', quantity)
            
            if sell_order:
                logger.info("Sell order placed successfully")
                
                # Wait a few seconds
                time.sleep(5)
                
                # Get final balance
                final_balance = bot.get_account_balance()
                logger.info(f"Final balance: {final_balance} USDT")
                
                # Calculate P&L
                pnl = final_balance - initial_balance
                logger.info(f"P&L: {pnl:.2f} USDT")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_order_placement() 