# Standard library imports
import logging

# Third-party imports


# Local application/library specific imports
from strategy import Strategy

class LGBMstrategy(Strategy):
    def __init__(self, model):
        self.model = model

    def get_signals(self, klines) -> dict:
        signals = {
            symbol: {
                "side": "Buy",
                "type": "MARKET",
                "quantity": 0.0,
                "price": 0.0
            }
            for symbol in klines.keys()
        }
        return None
    
        return signals

