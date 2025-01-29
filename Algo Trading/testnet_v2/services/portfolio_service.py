import datetime
from binance.um_futures import UMFutures
from binance.exceptions import BinanceAPIException
import pandas as pd
from typing import List, Dict, Optional
from config import TradingConfig
import logging

class PortfolioService:
    def __init__(self, binance_client: UMFutures):
        self.client = binance_client
        self.logger = logging.getLogger(__name__)
        self.account_refresh_time = None
        self.account = None
    
    def __str__(self):
        self._refresh_account()
        return f"""
        Account Value: {self.account['totalWalletBalance']}
        PnL: {self.account['totalUnrealizedProfit']}
        Open Positions: {self.account['positions']}
        Open Positions Size: {self.get_open_positions()}
        Open Positions Weights: {self.get_current_weights()}
        """

    def get_open_positions(self):
        """Get all current open positions"""
        self._refresh_account()
        positions_account = self.account['positions']
        positions_size = {symbol: 0 for symbol in TradingConfig.SYMBOLS}
        if not positions_account:
            return {}
        for position in positions_account:
            symbol = position['symbol']
            size = float(position['initialMargin'])
            positions_size[symbol] = size
        return positions_size

    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights of the portfolio"""
        self._refresh_account()
        positions_size = self.get_open_positions()
        positions_weights = {symbol: 0 for symbol in TradingConfig.SYMBOLS}
        total_wallet_balance = float(self.account['totalWalletBalance'])
        for symbol, size in positions_size.items():
            positions_weights[symbol] = size / total_wallet_balance
        return positions_weights
    
    def get_account_value(self):
        self._refresh_account()
        return float(self.account['totalWalletBalance'])

    def get_pnl(self):
        """Get current PnL of the portfolio"""
        self._refresh_account()
        return float(self.account['totalUnrealizedProfit'])

    def reset_account_refresh_time(self):
        """Reset account refresh time"""
        self.account_refresh_time = None

    def _refresh_account(self):
        if self.account_refresh_time is None or datetime.datetime.now(datetime.UTC) - self.account_refresh_time > datetime.timedelta(seconds=60):
            self.account = self.client.account()
            self.account_refresh_time = datetime.datetime.now(datetime.UTC)

