import os
import pandas as pd
import ta
from dotenv import load_dotenv
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler

def connectBinanceAPI():
    load_dotenv()
    API_KEY    = os.getenv('API_KEY')
    API_SECRET = os.getenv('API_SECRET')

    return Client(API_KEY, API_SECRET)

def getBinanceData(client,
                   symbol: str,
                   interval: str,
                   start_str: str,
                   end_str: str):
    kline = client.get_historical_klines(symbol=symbol,
                                         interval=interval,
                                         start_str=start_str,
                                         end_str=end_str)
    columns = ['index',
               'Open',
               'High',
               'Low',
               'Close',
               'Volume']
    data = pd.DataFrame(kline).iloc[:,:6]
    data.columns  = columns
    data['index'] = pd.to_datetime(data['index'], unit='ms')
    data.set_index('index', inplace=True)
    data = data.astype(float)
    data.dropna(inplace=True)
    
    return data

def create_previous_data_df(df, num_points):
    data_lists = []

    for i in range(len(df)):
        if i < num_points:
            continue
        
        prev_data = [df.iloc[i-n, 0] if i >= n else None for n in range(1, num_points+1)]
        data_lists.append(prev_data)

    columns = [f"X_{n}" for n in range(1, num_points+1)]
    prev_df = pd.DataFrame(data_lists, columns=columns)

    return prev_df


def sma(df, window):
    sma = ta.trend.SMAIndicator(pd.Series(df), window=window).sma_indicator()
    return sma

def rsi(df, window=14):
    rsi = ta.momentum.RSIIndicator(pd.Series(df), window=window).rsi()
    return rsi

def ema(df, period=200):
    ema = ta.trend.EMAIndicator(pd.Series(df), window=window).ema_indicator()
    return ema

def macd(df):
    macd = ta.trend.MACD(pd.Series(df)).macd()
    return macd

def signal_h(df):
    return ta.volatility.BollingerBands(pd.Series(df)).bollinger_hband()
def signal_l(df):
    return ta.volatility.BollingerBands(pd.Series(df)).bollinger_lband()

def transformData(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df)
    columns = ['Close']
    scaled_df.columns = columns

    return scaler, scaled_df

def featureGeneration(data):
    close = data.Close

    sma10 = sma(data.Close, 10)
    sma20 = sma(data.Close, 20)
    sma50 = sma(data.Close, 50)
    sma100 = sma(data.Close, 100)
    upper = signal_h(data.Close)
    lower = signal_l(data.Close)

    # Design matrix / independent features:

    # Price-derived features
    data['X_SMA10'] = (close - sma10) / close
    data['X_SMA20'] = (close - sma20) / close
    data['X_SMA50'] = (close - sma50) / close
    data['X_SMA100'] = (close - sma100) / close

    data['X_DELTA_SMA10'] = (sma10 - sma20) / close
    data['X_DELTA_SMA20'] = (sma20 - sma50) / close
    data['X_DELTA_SMA50'] = (sma50 - sma100) / close

    # Indicator features
    data['X_MOM'] = data.Close.pct_change(periods=2)
    data['X_BB_upper'] = (upper - close) / close
    data['X_BB_lower'] = (lower - close) / close
    data['X_BB_width'] = (upper - lower) / close

    data = data.dropna().astype(float)
    return data