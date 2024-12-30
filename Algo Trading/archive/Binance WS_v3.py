import json
import pandas as pd
from collections import deque
import asyncio
import websockets
import requests
import statistics
import matplotlib.pyplot as plt

from binance.um_futures import UMFutures

pd.options.mode.copy_on_write = True

# Constants
base_url_REST_testnet = "https://testnet.binancefuture.com"
base_url_WS_testnet = "wss://stream.binancefuture.com"
MAX_DATA = 180
symbols = ["ETHUSDT",
           "BTCUSDT"]
assets = dict.fromkeys(symbols, None)
for symbol in assets:
    assets[symbol] = {
        "symbol": symbol,
        "price": deque(maxlen=MAX_DATA),
        "mean": None,
        "std": None,
        "Z score": None,
        "quantity": None,
        "size": None
    }
is_long = False
is_short = False

# Test Net API Key and Secret
API_KEY = "b88d219ce4b7f4408ab0f51fc8be1730b264904e48e9a6ac7c7fa716e1e18dc7"
API_SECRET = "15acdeca3bb1dd761e472034c8b09bb597e165d18c69e485738ccf9a3aa4326a"

headers = {
    "Content-Type": "application/json;charset=utf-8",
    "X-MBX-APIKEY": API_KEY
}

class Asset:
    def __init__(self,
                 symbol: int
                 ):
        MAX_DATA = 180
        self.symbol = symbol
        self.price = deque(maxlen=MAX_DATA)
        self.mean = None
        self.std = None
        self.z_score = None

    def cal_mean_std(self):
        self.mean = statistics.mean(self.price)
        self.std = statistics.stdev(self.price)
        self.z_score = (self.price[-1] - self.mean) / self.std

    def get_latest_price(self):
        return self.price[-1] if self.price else None
class Portfolio:
    def __init__(self):
        self.assets = {}
        self.quantities = {}
        
    def __repr__(self):
        """
        Print a summary of the assets and their quantities in the portfolio.
        """
        print("Portfolio Summary:")
        for symbol, asset in self.assets.items():
            quantity = self.quantities[symbol]
            latest_price = asset.get_latest_price()
            value = latest_price * quantity if latest_price is not None else 0
            print(f"Symbol: {symbol}, Quantity: {quantity}, Latest Price: {latest_price}, Value: {value}")

    def add_asset(self,
                  asset,
                  quantity: float = 0):
        self.assets[asset.symbol] = asset
        self.quantities[asset.symbol] = quantity

    def update_quantity(self,
                        asset,
                        quantity: float):
        if asset.symbol in self.quantities:
            self.quantities[asset.symbol] = quantity
    
    def get_portfolio_value(self):
        total_value = 0
        for symbol, asset in self.assets.items():
            latest_price = asset.get_latest_price()
            if latest_price is not None:
                total_value += latest_price * self.quantities[symbol]
        return total_value

def break_response(message):
    trade_data = json.loads(message)["data"]
    event_type = trade_data["e"]
    symbol = trade_data["s"]
    price = trade_data['p']

    return (event_type, symbol, price)
def get_url(base_url, stream_names):
    url = f"{base_url}/stream?streams="
    for n in stream_names:
        if n == stream_names[-1]:
            url += n
        else:
            url += f"{n}/"
    return url
def long_strategy(kappa, leverage: int=1):
    """
    When the spread is negative, it means that asset 1 is underpriced and asset 2 is overpriced
    Therefore long asset 1 and short asset 2
    """
    global symbols, assets, um_futures_client
    balance = float(um_futures_client.account()["totalWalletBalance"])
    budget = balance * 0.5
    total_cost = kappa * assets[symbols[0]]["price"][-1] + assets[symbols[1]]["price"][-1]
    qty = budget / total_cost

    assets[symbols[0]]["quantity"] = round(qty * kappa, 3)
    assets[symbols[0]]["side"] = "BUY"

    assets[symbols[1]]["quantity"] = round(qty, 3)
    assets[symbols[1]]["side"] = "SELL"

    params = {
        "symbol": symbols[0],
        "side": "BUY",
        "type": "MARKET",
        # "type": "LIMIT",
        # "timeInForce": "GTC",
        # "price": round(assets[symbols[0]]["price"][-1], 2),
        "quantity": assets[symbols[0]]["quantity"]
    }    
    um_futures_client.new_order(**params)
    params = {
        "symbol": symbols[1],
        "side": "SELL",
        "type": "MARKET",
        # "type": "LIMIT",
        # "timeInForce": "GTC",
        # "price": round(assets[symbols[1]]["price"][-1], 1),
        "quantity": assets[symbols[1]]["quantity"]
    }
    um_futures_client.new_order(**params)
def short_strategy(kappa, leverage: int=1):
    """
    When the spread is positive, it means that asset 1 is overpriced and asset 2 is underpriced
    Therefore short asset 1 and long asset 2
    """
    global symbols, assets, um_futures_client
    balance = float(um_futures_client.account()["totalWalletBalance"])
    budget = balance * 0.5
    total_cost = kappa * assets[symbols[0]]["price"][-1] + assets[symbols[1]]["price"][-1]
    qty = budget / total_cost

    assets[symbols[0]]["quantity"] = round(qty * kappa, 3)
    assets[symbols[0]]["side"] = "SELL"

    assets[symbols[1]]["quantity"] = round(qty, 3)
    assets[symbols[1]]["side"] = "BUY"

    params = {
        "symbol": symbols[0],
        "side": "SELL",
        "type": "MARKET",
        # "type": "LIMIT",
        # "timeInForce": "GTC",
        # "price": round(assets[symbols[0]]["price"][-1], 2),
        "quantity": assets[symbols[0]]["quantity"]
    }
    um_futures_client.new_order(**params)
    params = {
        "symbol": symbols[1],
        "side": "BUY",
        "type": "MARKET",
        # "type": "LIMIT",
        # "timeInForce": "GTC",
        # "price": round(assets[symbols[1]]["price"][-1], 1),
        "quantity": assets[symbols[1]]["quantity"]
    }
    um_futures_client.new_order(**params)
def close_long_position():
    global symbols, assets, um_futures_client

    params = {
        "symbol": assets[symbols[0]]["symbol"],
        "side": "SELL",
        "type": "MARKET",
        # "type": "LIMIT",
        # "timeInForce": "GTC",
        # "price": assets[symbols[0]]["price"][-1],
        "quantity": assets[symbols[0]]["quantity"]
    }
    um_futures_client.new_order(**params)
    params = {
        "symbol": assets[symbols[1]]["symbol"],
        "side": "BUY",
        "type": "MARKET",
        # "type": "LIMIT",
        # "timeInForce": "GTC",
        # "price": assets[symbols[1]]["price"][-1],
        "quantity": assets[symbols[1]]["quantity"]
    }
    um_futures_client.new_order(**params)
def close_short_position():
    global symbols, assets, um_futures_client
    params = {
        "symbol": assets[symbols[0]]["symbol"],
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": assets[symbols[0]]["price"],
        "quantity": assets[symbols[0]]["quantity"]
    }
    um_futures_client.new_order(**params)
    params = {
        "symbol": assets[symbols[1]]["symbol"],
        "side": "SELL",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": assets[symbols[1]]["price"],
        "quantity": assets[symbols[1]]["quantity"]
    }
    um_futures_client.new_order(**params)
def on_message(message):
    """
    The websocket is mainly used to obtain market price of futures from Binance
    This on_message function is used to calculate the running mean and std of the data
    The output of this function is Z score, which is stored as global variable for the use of other functions
    """
    global symbols, assets, is_long, is_short

    (event_type, symbol, price) = break_response(message)

    if event_type == "markPriceUpdate":
        assets[symbol]["price"].append(float(price))

        if len(assets[symbol]["price"]) == MAX_DATA:
            assets[symbol] = cal_mean_std(assets[symbol])

            if symbol == symbols[-1] \
                and assets[symbols[-1]]["std"] is not None \
                    and assets[symbols[0]]["std"] is not None:
                # Calculate kappa
                kappa = assets[symbols[-1]]["std"] / assets[symbols[0]]["std"]
                # Check Z score difference
                spread = assets[symbols[0]]["Z score"] - assets[symbols[-1]]["Z score"]
                if is_long and spread < 0:
                    is_long = False
                    # close_long_position()
                if is_short and spread > 0:
                    # close_short_position()
                    is_short = False
                if not is_long and spread > 3.5:
                    print(f"{spread:.2f}")
                    long_strategy(kappa)
                    is_long = True
                if not is_short and spread < -3.5:
                    print(f"{spread:.2f}")
                    short_strategy(kappa)
                    is_short = True
def get_listen_key(url,
                   headers,
                   v: str = "v1",
                   endpoint = "listenKey"):
    return requests.post(
        f"{url}/fapi/{v}/{endpoint}",
        headers=headers
        ).json()["listenKey"]

async def connect_to_websocket(base_url):
    stream_names = [
        "ethusdt@markPrice",
        "btcusdt@markPrice"
    ]
    url = get_url(base_url, stream_names)
    async with websockets.connect(url) as ws:
        try:
            async for message in ws:
                on_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")
async def connect_user_data(base_url: str, listen_key: str):
    url = f"{base_url}/ws/{listen_key}"
    async with websockets.connect(url) as ws:
        try:
            async for message in ws:
                print(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")
async def main(base_url_ws, listen_key):
    await asyncio.gather(
        connect_to_websocket(base_url_ws),
        connect_user_data(base_url_ws, listen_key)
    )

if __name__ == "__main__":
    listen_key = get_listen_key(base_url_REST_testnet, headers)
    um_futures_client = UMFutures(key=API_KEY,
                                  secret=API_SECRET,
                                  base_url=base_url_REST_testnet)
    asyncio.run(main(base_url_WS_testnet, listen_key))