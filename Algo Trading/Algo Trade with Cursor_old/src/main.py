import os
from dotenv import load_dotenv
from trading.trading_bot import TradingBot
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Trading pairs to trade
    trading_pairs = [
        'BTC/USDT',
        'ETH/USDT',
        'BNB/USDT',
        'SOL/USDT',
        'ADA/USDT',
        'DOT/USDT',
        'AVAX/USDT',
        'MATIC/USDT'
    ]
    
    # Initial capital in USDT
    initial_capital = 1000  # Adjust based on your capital
    
    try:
        # Initialize and run trading bot
        bot = TradingBot(trading_pairs, initial_capital)
        bot.run()
        
    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user")
    except Exception as e:
        logging.error(f"Error running trading bot: {str(e)}")
        raise

if __name__ == "__main__":
    main() 