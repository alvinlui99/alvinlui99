from binance.um_futures import UMFutures
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

from ..data.market_data import MarketData

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, trading_pairs: List[str], initial_capital: float):
        """
        Initialize the trading bot.
        
        Args:
            trading_pairs (List[str]): List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            initial_capital (float): Initial capital in USDT
        """
        self.trading_pairs = trading_pairs
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        
        # Initialize Binance Futures client with testnet
        self.client = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET'),
            base_url="https://testnet.binancefuture.com"  # Testnet URL
        )
        
        # Synchronize time with Binance server
        self._sync_time()
        
        # Initialize market data
        self.market_data = MarketData(
            client=self.client,
            symbols=trading_pairs,
            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        # Load historical data
        self.historical_data = self.market_data.fetch_historical_data()
        
        # Set up testnet account
        self._setup_testnet_account()
        
        # Set up initial positions
        self._update_positions()

    def _sync_time(self) -> None:
        """Synchronize time with Binance server."""
        try:
            server_time = self.client.time()
            local_time = int(time.time() * 1000)
            time_diff = server_time['serverTime'] - local_time
            
            if abs(time_diff) > 1000:  # If difference is more than 1 second
                logger.warning(f"Time difference with server: {time_diff}ms")
                logger.info("Please synchronize your system time with an internet time server")
                
        except Exception as e:
            logger.error(f"Error syncing time: {str(e)}")

    def _setup_testnet_account(self) -> None:
        """Set up testnet account with initial capital."""
        try:
            # Get current balance
            balance = self.get_account_balance()
            logger.info(f"Current testnet balance: {balance} USDT")
            
            # If balance is too low, request testnet funds
            if balance < 100:
                logger.info("Requesting testnet funds...")
                logger.info("Please visit https://testnet.binance.vision/ to request testnet funds")
            
        except Exception as e:
            logger.error(f"Error setting up testnet account: {str(e)}")

    def _update_positions(self) -> None:
        """Update current positions from exchange."""
        try:
            positions = self.client.get_position_risk()
            for position in positions:
                symbol = position['symbol']
                if symbol in self.trading_pairs:
                    self.positions[symbol] = {
                        'size': float(position['positionAmt']),
                        'entry_price': float(position['entryPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit'])
                    }
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")

    def set_margin_type(self, symbol: str, margin_type: str) -> None:
        """
        Set margin type for a trading pair.
        
        Args:
            symbol (str): Trading pair
            margin_type (str): 'ISOLATED' or 'CROSS'
        """
        try:
            self.client.change_margin_type(
                symbol=symbol,
                marginType=margin_type
            )
            logger.info(f"Set margin type to {margin_type} for {symbol}")
        except Exception as e:
            logger.error(f"Error setting margin type: {str(e)}")

    def get_margin_type(self, symbol: str) -> str:
        """
        Get current margin type for a trading pair.
        
        Args:
            symbol (str): Trading pair
            
        Returns:
            str: Current margin type ('ISOLATED' or 'CROSS')
        """
        try:
            position = self.client.get_position_risk(symbol=symbol)
            if position:
                return position[0].get('marginType', 'ISOLATED')
            return 'ISOLATED'
        except Exception as e:
            logger.error(f"Error getting margin type: {str(e)}")
            return 'ISOLATED'

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Set leverage for a trading pair.
        
        Args:
            symbol (str): Trading pair
            leverage (int): Leverage value (1-125)
        """
        try:
            self.client.change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            logger.info(f"Set leverage to {leverage}x for {symbol}")
        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")

    def get_account_balance(self) -> float:
        """Get current account balance in USDT."""
        try:
            account = self.client.account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching account balance: {str(e)}")
            return 0.0

    def place_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """
        Place a market order.
        
        Args:
            symbol (str): Trading pair
            side (str): 'BUY' or 'SELL'
            quantity (float): Order quantity
            
        Returns:
            Dict: Order response
        """
        try:
            # Update positions before placing order
            self._update_positions()
            
            # Check if we have enough balance
            balance = self.get_account_balance()
            current_price = self.market_data.get_latest_price(symbol)
            order_value = quantity * current_price
            
            if order_value > balance:
                logger.error(f"Insufficient balance. Required: {order_value:.2f} USDT, Available: {balance:.2f} USDT")
                return None
            
            # Place the order
            order = self.client.new_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Update positions after order
            self._update_positions()
            
            logger.info(f"Placed {side} order for {quantity} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    def run(self) -> None:
        """Main trading loop."""
        while True:
            try:
                # Get current balance
                balance = self.get_account_balance()
                logger.info(f"Current balance: {balance} USDT")
                
                # Update positions
                self._update_positions()
                
                # Log current positions
                for symbol, position in self.positions.items():
                    logger.info(f"Position in {symbol}: {position['size']} units, PnL: {position['unrealized_pnl']:.2f} USDT")
                
                # Wait for next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait before retrying 