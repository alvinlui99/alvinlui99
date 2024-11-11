from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.um_futures import UMFutures

import json
import pandas as pd
from collections import deque
import asyncio
import websockets
import requests
import statistics
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True

# Constants
base_url_REST_testnet = "https://testnet.binancefuture.com"
base_url_WS_testnet = "wss://stream.binancefuture.com"
MAX_DATA = 10
symbols = ["ETHUSDT",
           "BTCUSDT"]
data    = {symbol: deque(maxlen=MAX_DATA) for symbol in symbols}
mean    = {symbol: deque(maxlen=MAX_DATA) for symbol in symbols}
std     = {symbol: deque(maxlen=MAX_DATA) for symbol in symbols}
z_score = {symbol: deque(maxlen=MAX_DATA) for symbol in symbols}
asset_1 = {"symbol": symbols[0]}
asset_2 = {"symbol": symbols[1]}
is_long = False
is_short = False

# Test Net API Key and Secret
API_KEY = "b88d219ce4b7f4408ab0f51fc8be1730b264904e48e9a6ac7c7fa716e1e18dc7"
API_SECRET = "15acdeca3bb1dd761e472034c8b09bb597e165d18c69e485738ccf9a3aa4326a"

def break_response(message):
    trade_data = json.loads(message)
    event_type = trade_data["e"]
    symbol = trade_data["s"]
    price = trade_data['p']

    return (event_type, symbol, price)
def update_data(symbol):
    global mean, std, z_score
    mean[symbol].append(statistics.mean(data[symbol]))
    std[symbol].append(statistics.stdev(data[symbol]))
    z_score[symbol].append((data[symbol][-1] - mean[symbol][-1]) / std[symbol][-1])
def long_strategy(kappa, leverage: int=1):
    """
    When the spread is negative, it means that asset 1 is underpriced and asset 2 is overpriced
    Therefore long asset 1 and short asset 2
    """
    global asset_1, asset_2, um_futures_client
    balance = float(um_futures_client.account()["totalWalletBalance"])
    budget = balance * 0.5
    total_cost = kappa * asset_1["price"] + asset_2["price"]
    qty = budget / total_cost

    print(f"""
    balance:{balance:.3f}
    budget:{budget:.3f}
    total cost:{total_cost:.3f}
    quantity:{qty:.3f}
    kappa:{kappa:.3f}
    
    {asset_1["symbol"]}:{qty * kappa:.3f}
    {asset_2["symbol"]}:{qty:.3f}
    """)

    asset_1["qty"] = round(qty * kappa, 3)
    asset_1["side"] = "BUY"

    asset_2["qty"] = round(qty, 3)
    asset_1["side"] = "SELL"

    params = {
        "symbol": asset_1["symbol"],
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_1["price"],
        "quantity": asset_1["qty"]
    }
    um_futures_client.new_order(**params)
    params = {
        "symbol": asset_2["symbol"],
        "side": "SELL",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_2["price"],
        "quantity": asset_2["qty"]
    }
    um_futures_client.new_order(**params)
def short_strategy(kappa, leverage: int=1):
    """
    When the spread is positive, it means that asset 1 is overpriced and asset 2 is underpriced
    Therefore short asset 1 and long asset 2
    """
    global asset_1, asset_2, um_futures_client
    balance = float(um_futures_client.account()["totalWalletBalance"])
    budget = balance * 0.5
    total_cost = kappa * asset_1["price"] + asset_2["price"]
    qty = budget / total_cost

    print(f"""
    balance:{balance:.3f}
    budget:{budget:.3f}
    total cost:{total_cost:.3f}
    quantity:{qty:.3f}
    kappa:{kappa:.3f}

    {asset_1["symbol"]}:{qty * kappa:.3f}
    {asset_2["symbol"]}:{qty:.3f}
    """)

    asset_1["qty"] = round(qty * kappa, 3)
    asset_1["side"] = "SELL"

    asset_2["qty"] = round(qty, 3)
    asset_1["side"] = "BUY"

    params = {
        "symbol": asset_1["symbol"],
        "side": "SELL",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_1["price"],
        "quantity": asset_1["qty"]
    }
    um_futures_client.new_order(**params)
    params = {
        "symbol": asset_2["symbol"],
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_2["price"],
        "quantity": asset_2["qty"]
    }
    um_futures_client.new_order(**params)
def close_long_position():
    global asset_1, asset_2, um_futures_client

    params = {
        "symbol": asset_1["symbol"],
        "side": "SELL",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_1["price"],
        "quantity": asset_1["qty"]
    }
    um_futures_client.new_order(**params)
    params = {
        "symbol": asset_2["symbol"],
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_2["price"],
        "quantity": asset_2["qty"]
    }
    um_futures_client.new_order(**params)
def close_short_position():
    global asset_1, asset_2, um_futures_client
    params = {
        "symbol": asset_1["symbol"],
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_1["price"],
        "quantity": asset_1["qty"]
    }
    um_futures_client.new_order(**params)
    params = {
        "symbol": asset_2["symbol"],
        "side": "SELL",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": asset_2["price"],
        "quantity": asset_2["qty"]
    }
    um_futures_client.new_order(**params)
def on_message(_, message):
    """
    The websocket is mainly used to obtain market price of futures from Binance
    This on_message function is used to calculate the running mean and std of the data
    The output of this function is Z score, which is stored as global variable for the use of other functions
    """
    global symbols, data, z_score, um_futures_client, asset_1, asset_2, is_long, is_short

    (event_type, symbol, price) = break_response(message)

    if event_type == "markPriceUpdate":
        data[symbol].append(float(price))

        if len(data[symbol]) == MAX_DATA:
            update_data(symbol)

            if symbol == symbols[-1]:
                # Calculate kappa
                kappa = std[symbols[-1]][-1] / std[symbols[0]][-1]
                # Check Z score difference
                spread = z_score[symbols[0]][-1] - z_score[symbols[-1]][-1]
                asset_1["price"] = float(round(data[asset_1["symbol"]][-1], 2))
                asset_2["price"] = float(round(data[asset_2["symbol"]][-1], 2))
                if is_long and spread < 0:
                    is_long = False
                    # close_long_position()
                if is_short and spread > 0:
                    # close_short_position()
                    is_short = False
                if not is_long and spread > 2:
                    print(f"{spread:.2f}")
                    long_strategy(kappa)
                    is_long = True
                if not is_short and spread < -2:
                    print(f"{spread:.2f}")
                    short_strategy(kappa)
                    is_short = True
def on_error(_, error):
    print(error)
    # print(json.loads(error))
def on_close(_):
    print("WebSocket Connection Closed")
def on_open(_):
    print("WebSocket Connection Opened")
async def connect_to_stream(url):
    ws_client = UMFuturesWebsocketClient(stream_url="wss://stream.binancefuture.com",
                                         on_message=on_message,
                                         on_error=on_error,
                                         on_close=on_close,
                                         on_open=on_open)   
    # ws_client.mark_price("ethusdt", 1)
    # ws_client.mark_price("btcusdt", 1)

if __name__ == "__main__":
    um_futures_client = UMFutures(key=API_KEY, secret=API_SECRET, base_url=base_url_REST_testnet)
    asyncio.run(connect_to_stream(base_url_WS_testnet))