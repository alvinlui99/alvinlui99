# Version notes
# This version is modified from v4
# This version uses REST API and gets data every minute


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
ENTRY_SPREAD = 3
EXIT_SPREAD = 0.5
MAX_LEN = 26

symbols = ["ETHUSDT",
           "BTCUSDT"]

# Test Net API Key and Secret
API_KEY = "b88d219ce4b7f4408ab0f51fc8be1730b264904e48e9a6ac7c7fa716e1e18dc7"
API_SECRET = "15acdeca3bb1dd761e472034c8b09bb597e165d18c69e485738ccf9a3aa4326a"

headers = {
    "Content-Type": "application/json;charset=utf-8",
    "X-MBX-APIKEY": API_KEY
}

class Asset:
    def __init__(self,
                 symbol: str,
                 maxlen: int = 300
                 ):
        self.symbol = symbol
        self.prices = deque(maxlen=maxlen)
        self.quantity = 0
        self.mean = None
        self.std = None
        self.z_score = None
    def cal_mean_std(self):
        if len(self.prices) == self.prices.maxlen:
            self.mean = statistics.mean(self.prices)
            self.std = statistics.stdev(self.prices)
            self.z_score = (self.prices[-1] - self.mean) / self.std
    def get_latest_price(self):
        return self.prices[-1] if self.prices else None
    def update_quantity(self,
                        quantity: float):
        self.quantity = quantity
    def update_price(self,
                     price: float):
        self.prices.append(price)
        self.cal_mean_std()
class Portfolio:
    def __init__(self,
                 symbols: str,
                 maxlen: int = 300
                 ):
        self.assets = {}
        [self.add_asset(symbol, maxlen) for symbol in symbols]
    def __repr__(self):
        """
        Print a summary of the assets and their quantities in the portfolio.
        """
        print("Portfolio Summary:")
        for symbol, asset in self.assets.items():
            quantity = asset.quantity
            latest_price = asset.get_latest_price()
            value = latest_price * quantity if latest_price is not None else 0
            print(f"Symbol: {symbol}, Quantity: {quantity}, Latest Price: {latest_price}, Value: {value}")
    def add_asset(self,
                  symbol: str,
                  maxlen: int):
        self.assets[symbol] = Asset(symbol, maxlen)
    def update_portfolio_quantity(self,
                                  symbol,
                                  quantity: float):
        self.assets[symbol].update_quantity(quantity)
    def update_portfolio_price(self,
                               symbol: str,
                               price: float):
        # precision mapping
        precision_mapping = {
            "BTCUSDT": 1,
            "ETHUSDT": 2
        }
        precision = precision_mapping[symbol]
        self.assets[symbol].update_price(round(price, precision))
    def get_portfolio_value(self):
        """
        get the total value of the portfolio with the latest price

        Returns:
            total_value: float
        """
        total_value = 0
        for symbol, asset in self.assets.items():
            latest_price = asset.get_latest_price()
            if latest_price is not None:
                total_value += latest_price * asset.quantity[symbol]
        return total_value
    def get_portfolio_zscores(self):
        """
        get the z scores of all assets in the portfolio

        Returns:
            z_scores: a dict with symbol as key and z score as value
        """
        z_scores = {}
        for symbol, asset in self.assets.items():
            z_score = asset.z_score
            if z_score is None:
                return None
            z_scores[symbol] = z_score
        return z_scores
    def get_portfolio_kappa(self,
                            symbols: list):
        """
        kappa is the ratio in quantity to be long/short between the two assets
        """
        return self.assets[symbols[1]].std / self.assets[symbols[0]].std
    def get_portfolio_is_in_position(self):
        for asset in self.assets.values():
            if asset.quantity != 0:
                # is in position
                return True
        return False

def get_decision(
    assets: dict,
    symbols: list,
    budget: float,
    is_in_position: bool,
    **kwargs):
    """
    kwargs:
        kappa
        spread
        entry_spread
        exit_spread

    If not triggered, return None
    If triggered, return (orders, in_position)
    orders:
        fixed format dict of dict
        Symbol: str
            Order: Market/Limit
            Price: float
            Quantity: float
    in_position:
        bool. True if in position AFTER decision
    """
    orders = {}
    
    for symbol in symbols:
        orders[symbol] = {
            "type": None,
            "side": None,
            # "price": None,
            "quantity": None
        }
        
    # strategy specific --------------------------------------
    kappa = kwargs.get("kappa", None)
    spread = kwargs.get("spread", None)
    entry_spread = kwargs.get("entry_spread", None)
    exit_spread = kwargs.get("exit_spread", None)

    if is_in_position:
        if abs(spread) < exit_spread:
            # close positions
            for symbol, asset in assets.items():
                orders[symbol]["type"] = "MARKET"
                orders[symbol]["side"] = "BUY" if asset.quantity < 0 else "SELL"
                # COME BACK AND FIX THIS
                orders[symbol]["quantity"] = abs(asset.quantity)
            return (orders, False)
    else:
        total_cost = kappa * assets[symbols[0]].prices[-1] + assets[symbols[1]].prices[-1]
        if spread < - entry_spread:
            orders[symbols[0]]["type"] = "Market"
            orders[symbols[0]]["side"] = "SELL"
            orders[symbols[0]]["quantity"] = round(budget / total_cost * kappa, 3)
            orders[symbols[1]]["type"] = "Market"
            orders[symbols[1]]["side"] = "BUY"
            orders[symbols[1]]["quantity"] = round(budget / total_cost, 3)

            return (orders, True)

        elif spread > entry_spread:
            orders[symbols[0]]["type"] = "Market"
            orders[symbols[0]]["side"] = "BUY"
            orders[symbols[0]]["quantity"] = round(budget / total_cost * kappa, 3)
            orders[symbols[1]]["type"] = "Market"
            orders[symbols[1]]["side"] = "SELL"
            orders[symbols[1]]["quantity"] = round(budget / total_cost, 3)

            return (orders, True)
    # strategy specific --------------------------------------
    
    return None
def break_response(
    message):
    trade_data = json.loads(message)["data"]
    event_type = trade_data["e"]
    symbol = trade_data["s"]
    price = trade_data['p']

    return (event_type, symbol, price)
def get_url(
    base_url,
    stream_names):
    url = f"{base_url}/stream?streams="
    for n in stream_names:
        if n == stream_names[-1]:
            url += n
        else:
            url += f"{n}/"
    return url
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
def on_message(
    message):
    """
    The websocket is mainly used to obtain market price of futures from Binance
    This on_message function is used to calculate the running mean and std of the data
    The output of this function is Z score, which is stored as global variable for the use of other functions
    """
    global ENTRY_SPREAD, EXIT_SPREAD, portfolio, symbols, um_futures_client

    (event_type, symbol, price) = break_response(message)

    if event_type == "markPriceUpdate":
        portfolio.update_portfolio_price(symbol, float(price))

        zscores = portfolio.get_portfolio_zscores()
        if zscores is not None:
            if symbol == symbols[1]:
                print(f"Spread: {zscores[symbols[0]] - zscores[symbols[1]]}")
                decision = get_decision(
                    assets=portfolio.assets,
                    symbols=symbols,
                    budget=float(um_futures_client.account()["totalWalletBalance"]) * 0.9,
                    is_in_position=portfolio.get_portfolio_is_in_position(),
                    kappa=portfolio.get_portfolio_kappa(symbols=symbols),
                    spread=zscores[symbols[0]] - zscores[symbols[1]],
                    entry_spread=ENTRY_SPREAD,
                    exit_spread=EXIT_SPREAD
                )
                if decision is not None:
                    orders, in_position = decision
                    # place order
                    print(orders)
                    for symbol, params in orders.items():
                        params["symbol"] = symbol
                        um_futures_client.new_order(**params)
                        if in_position:
                            # open position
                            if params["side"] == "SELL":
                                portfolio.update_portfolio_quantity(symbol, - params["quantity"])
                            elif params["side"] == "BUY":
                                portfolio.update_portfolio_quantity(symbol, params["quantity"])
                        else:
                            # close position
                            portfolio.update_portfolio_quantity(symbol, 0)
def get_listen_key(
    url,
    headers,
    v: str = "v1",
    endpoint = "listenKey"):
    return requests.post(
        f"{url}/fapi/{v}/{endpoint}",
        headers=headers
        ).json()["listenKey"]

async def connect_to_websocket(
    base_url):
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
async def connect_user_data(
    base_url: str,
    listen_key: str):
    url = f"{base_url}/ws/{listen_key}"
    async with websockets.connect(url) as ws:
        try:
            async for message in ws:
                print(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")
async def get_market_data_RESTAPI(
    api_key,
    api_secret,
    url):
    global ENTRY_SPREAD, EXIT_SPREAD, portfolio, symbols
    while True:
        um_futures_client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url=url)
        for symbol in symbols:
            price = um_futures_client.mark_price(symbol)["markPrice"]
            portfolio.update_portfolio_price(symbol, round(float(price), 2))
        zscores = portfolio.get_portfolio_zscores()
        if zscores is not None:
            print(f"Spread: {zscores[symbols[0]] - zscores[symbols[1]]}")
            decision = get_decision(
                assets=portfolio.assets,
                symbols=symbols,
                budget=float(um_futures_client.account()["totalWalletBalance"]) * 10,
                is_in_position=portfolio.get_portfolio_is_in_position(),
                kappa=portfolio.get_portfolio_kappa(symbols=symbols),
                spread=zscores[symbols[0]] - zscores[symbols[1]],
                entry_spread=ENTRY_SPREAD,
                exit_spread=EXIT_SPREAD
            )
            if decision is not None:
                orders, in_position = decision
                # place order
                print(orders)
                for symbol, params in orders.items():
                    params["symbol"] = symbol
                    um_futures_client.new_order(**params)
                    if in_position:
                        # open position
                        if params["side"] == "SELL":
                            portfolio.update_portfolio_quantity(symbol, - params["quantity"])
                        elif params["side"] == "BUY":
                            portfolio.update_portfolio_quantity(symbol, params["quantity"])
                    else:
                        # close position
                        portfolio.update_portfolio_quantity(symbol, 0)
        await asyncio.sleep(60)
async def main(
    api_key,
    api_secret,
    url):
    await asyncio.gather(
        get_market_data_RESTAPI(
            api_key=api_key,
            api_secret=api_secret,
            url=url)
    )

if __name__ == "__main__":
    portfolio = Portfolio(symbols, MAX_LEN)
    asyncio.run(
        main(
            API_KEY,
            API_SECRET,
            base_url_REST_testnet))