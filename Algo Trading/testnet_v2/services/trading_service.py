from binance.um_futures import UMFutures
from typing import Dict

class TradingService:
    def __init__(self, binance_client: UMFutures):
        self.client = binance_client
        
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        type: str = 'market'
    ) -> None:
        """Place a new order"""
        self.client.new_order(
            symbol=symbol,
            side=side,
            type=type,
            quantity=quantity
        )
    
    def set_leverage(
        self,
        symbol: str,
        leverage: int
    ) -> None:
        """Set leverage for a symbol"""
        self.client.change_leverage(
            symbol=symbol,
            leverage=leverage
        )

    def set_all_leverages(
        self,
        leverages: Dict[str, int]
    ) -> None:
        """Set leverage for all symbols"""
        for symbol, leverage in leverages.items():
            self.set_leverage(symbol, leverage)

    def cancel_order(self, symbol, order_id):
        """Cancel an existing order"""
        pass
        
    def get_order_status(self, symbol, order_id):
        """Check status of an order"""
        pass