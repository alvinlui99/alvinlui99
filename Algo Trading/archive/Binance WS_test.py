# from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
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

MAX_DATA = 5
symbols = ["ETHUSDT",
           "BTCUSDT"]
data    = {symbol: deque(maxlen=MAX_DATA) for symbol in symbols}
mean    = dict.fromkeys(symbols, None)
std     = dict.fromkeys(symbols, None)
z_score = dict.fromkeys(symbols, None)
asset_1 = {"symbol": symbols[0]}
asset_2 = {"symbol": symbols[1]}
is_long = False
is_short = False

# Test Net API Key and Secret
API_KEY = "b88d219ce4b7f4408ab0f51fc8be1730b264904e48e9a6ac7c7fa716e1e18dc7"
API_SECRET = "15acdeca3bb1dd761e472034c8b09bb597e165d18c69e485738ccf9a3aa4326a"

def read_mark_price_stream(data):
    key_mapping = {
        "e": "Event Type",
        "E": "Event Time",
        "s": "Symbol",
        "p": "Mark Price",
        "i": "Index Price",
        "P": "Est Settle Price",
        "r": "Funding Rate",
        "T": "Next Funding Time",
    }
    for old_key, new_key in key_mapping.items():
        data[new_key] = data.pop(old_key)
    return data
def get_url(base_url, stream_names):
    url = f"{base_url}/stream?streams="
    for n in stream_names:
        if n == stream_names[-1]:
            url += n
        else:
            url += f"{n}/"
    return url
def update_data(symbol):
    global mean, std, z_score
    mean[symbol] = statistics.mean(data[symbol])
    std[symbol] = statistics.stdev(data[symbol])
    z_score[symbol] = (data[symbol][-1] - mean[symbol]) / std[symbol]
def long_strategy(kappa, leverage: int=1):
    """
    When the spread is negative, it means that asset 1 is underpriced and asset 2 is overpriced
    Therefore long asset 1 and short asset 2
    """
    global asset_1, asset_2
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
def on_message(message):
    """
    The websocket is mainly used to obtain market price of futures from Binance
    This on_message function is used to calculate the running mean and std of the data
    The output of this function is Z score, which is stored as global variable for the use of other functions
    """
    global symbols, data, mean, std, z_score, asset_1, asset_2, is_long, is_short

    message = json.loads(message)
    feed = read_mark_price_stream(message["data"])
    symbol = feed["Symbol"]
    index_price = feed["Index Price"]

    if feed["Event Type"] == "markPriceUpdate":
        data[symbol].append(float(index_price))
        if len(data[symbol]) == MAX_DATA:
            update_data(symbol)
            if symbol == symbols[-1] and std[symbols[-1]] is not None and std[symbols[0]] is not None:
                # Calculate kappa
                kappa = std[symbols[-1]] / std[symbols[0]]
                # Check Z score difference
                spread = z_score[symbols[0]] - z_score[symbols[-1]]
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
async def connect_to_websocket():
    base_url_WS_testnet = "wss://fstream.binancefuture.com" # "wss://stream.binancefuture.com/"
    stream_names = [
        "ethusdt@markPrice",
        "btcusdt@markPrice"
    ]

    url = get_url(base_url_WS_testnet, stream_names)
    
    async with websockets.connect(url) as ws:
        print("Connection opened")
        try:
            async for message in ws:
                on_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")

def get_listen_key(url,
                         headers,
                         v: str = "v1",
                         listen_key_endpoint = "listenKey"):
    return requests.post(
        f"{url}/fapi/{v}/{listen_key_endpoint}",
        headers=headers
        ).json()["listenKey"]


if __name__ == "__main__":
    # asyncio.run(connect_to_websocket())
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "X-MBX-APIKEY": API_KEY
    }

    listen_key = get_listen_key(base_url_REST_testnet, headers)