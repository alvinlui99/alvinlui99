from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

import json
import time
import datetime
import numpy as np
import pandas as pd
from collections import deque
import asyncio
import websockets
import requests

pd.options.mode.copy_on_write = True

# Constants
base_url_WS_testnet = "wss://testnet.binancefuture.com/ws-fapi/v1" # "wss://fstream.binancefuture.com"
MAX_DATA = 20

# Initialize a deque with fixed size to store data
data_queue = deque(maxlen=MAX_DATA)

def on_message(_, message):
    trade_data = json.loads(message)
    event_type = trade_data["e"]
    if event_type == "markPriceUpdate":
        print(f"""
        Timestamp: {trade_data['E']}
        Symbol: {trade_data['s']}
        Mark Price: {trade_data['p']}
        Index Price: {trade_data['i']}
        ---
        """)
    else:
        print("ERROROROROROR")

def on_error(_, error):
    print(error)
    # print(json.loads(error))

def on_close(_):
    print("WebSocket Connection Closed")

def on_open(_):
    print("WebSocket Connection Opened")

if __name__ == "__main__":
    # Subscribe to a single symbol stream
    # asyncio.run(stream_mark_price("btcusdt"))
    ws_client = UMFuturesWebsocketClient(stream_url="wss://stream.binancefuture.com",
                                         on_message=on_message,
                                         on_error=on_error,
                                         on_close=on_close,
                                         on_open=on_open)
    ws_client.mark_price("btcusdt", 1)