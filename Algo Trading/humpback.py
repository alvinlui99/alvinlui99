import os
import pandas as pd
import numpy as np
import ta
from dotenv import load_dotenv
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

def create_previous_data_data(data, num_points):
    data_lists = []

    for i in range(len(data)):
        if i < num_points:
            continue
        
        prev_data = [data.iloc[i-n, 0] if i >= n else None for n in range(1, num_points+1)]
        data_lists.append(prev_data)

    columns = [f"X_{n}" for n in range(1, num_points+1)]
    prev_data = pd.DataFrame(data_lists, columns=columns)

    return prev_data

def sma(data, window=14):
    sma = ta.trend.SMAIndicator(data.loc[:, 'Close'], window=window).sma_indicator()
    return sma

def ema(data, window=14):
    ema = ta.trend.EMAIndicator(data.loc[:, 'Close'], window=window).ema_indicator()
    return ema

def rsi(data, window=14):
    rsi = ta.momentum.RSIIndicator(data.loc[:, 'Close'], window=window).rsi()
    return rsi

def macd(data):
    macd = ta.trend.MACD(data.loc[:, 'Close']).macd()
    return macd

def atr(data):
    return ta.volatility.AverageTrueRange(data.loc[:, 'High'], data.loc[:, 'Low'], data.loc[:, 'Close']).average_true_range()

def signal_h(data):
    return ta.volatility.BollingerBands(data.loc[:, 'Close']).bollinger_hband()
def signal_l(data):
    return ta.volatility.BollingerBands(data.loc[:, 'Close']).bollinger_lband()
    
def addSMA(data, sma_start: int, sma_end: int, sma_step: int):
    sma_range = range(sma_start, sma_end, sma_step)

    for window in sma_range:
        # SMA/Close
        # relative value of SMA to closing price
        data.loc[:,f'SMA_{window}'] = sma(data, window) / data.loc[:,'Close']

    for i in range(len(sma_range)):
        for j in range(i+1, len(sma_range)):
            # (SMA_1 - SMA_2) / Close
            data.loc[:,f'SMA_DELTA_{sma_range[i]}_{sma_range[j]}'] = data.loc[:,f'SMA_{sma_range[i]}'] - data.loc[:,f'SMA_{sma_range[j]}']

    return data

def addEMA(data, ema_start: int, ema_end: int, ema_step: int):
    ema_range = range(ema_start, ema_end, ema_step)

    for window in ema_range:
        # EMA/Close
        # relative value of EMA to closing price
        data.loc[:,f'EMA_{window}'] = ema(data, window) / data.loc[:,'Close']

    for i in range(len(ema_range)):
        for j in range(i+1, len(ema_range)):
            # (EMA_1 - EMA_2) / Close
            data.loc[:,f'EMA_DELTA_{ema_range[i]}_{ema_range[j]}'] = data.loc[:,f'EMA_{ema_range[i]}'] - data.loc[:,f'EMA_{ema_range[j]}']

    return data

def featureGeneration_bu(
    data,
    sma_start: int = 20,
    sma_end  : int = 100,
    sma_step : int = 20,
    ema_start: int = 20,
    ema_end  : int = 100,
    ema_step : int = 20
    ):
    """
    Input
    --------
    X: pd.DataFrame


    Output
    --------
    X: pd.DataFrame
    """
    close = data.Close

    # SMA
    data = addSMA(data=data, sma_start=sma_start, sma_end=sma_end, sma_step=sma_step)

    # EMA
    data = addEMA(data=data, ema_start=ema_start, ema_end=ema_end, ema_step=ema_step)

    # RSI
    data['RSI'] = rsi(data=data)

    # MACD
    data['MACD'] = macd(data=data)

    # ATR
    data['ATR'] = atr(data=data)

    # BollingerBands
    upper = signal_h(data.Close)
    lower = signal_l(data.Close)

    data['BB_upper'] = (upper - close) / close
    data['BB_lower'] = (lower - close) / close
    data['BB_width'] = (upper - lower) / close

    # Garman Klass Volatility
    data['garman_klass_vol'] = ((np.log(data['High'])-np.log(data['Low']))**2)/2-(2*np.log(2)-1)*((np.log(data['Close'])-np.log(data['Open']))**2)

    # Output
    OHLC_cols = ['Open', 'High', 'Low', 'Close']
    data = data.drop(OHLC_cols, axis=1).dropna().astype(float)

    return data

def getReturn(data, y_column: str = 'Close', dropna: bool = True):
    """Reads OHLCV and compute the period return based on closing price

    Args:
        data (pd.DataFrame): Raw OHLCV
        y_column (str)     : which column the return is calculated based on

    Returns:
        data (pd.DataFrame): with one more column, and dropped na
    """

    data['Return'] = data[y_column].shift(-1) / data[y_column] - 1
    if dropna:
        data = data.dropna(inplace=False)
    return data

def getXy(data, y_column: str = 'Return'):
    """split data into feature and target

    Args:
        data (pd.DataFrame): dataframe
        y_column (str, optional): target column name. Defaults to 'Return'.

    Returns:
        X: feature
        y: target
    """
    data.dropna(inplace=True)
    y = data[[y_column]]
    X = data.drop(y_column, axis=1)
    return X, y

def preprocess(
    data_df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    apply_PCA: bool,
    PCA_components: int = None):
    """Splits data into train/val/test sets and scales the data.

    Args:
        data_df: DataFrame of shape `(num_time_steps, num_features)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.
        test_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the test split.

    Returns:
        `train_df`, `val_df`, `test_df`
    """

    train_df, val_df, test_df = train_val_test_split(data_df, train_size, val_size, test_size)

    scaler = StandardScaler()
    scaler = scaler.fit(train_df)
    
    # train_df = pd.DataFrame(
    #     scaler.transform(train_df),
    #     index=train_df.index,
    #     columns=train_df.columns)

    # val_df = pd.DataFrame(
    #     scaler.transform(val_df),
    #     index=val_df.index,
    #     columns=val_df.columns)

    # test_df = pd.DataFrame(
    #     scaler.transform(test_df),
    #     index=test_df.index,
    #     columns=test_df.columns)

    train_array = scaler.transform(train_df)
    val_array   = scaler.transform(val_df)
    if test_df.empty:
        test_array  = []
    else:
        test_array  = scaler.transform(test_df)

    # PCA
    pca = PCA(n_components=PCA_components)
    pca.fit(train_array)

    if apply_PCA:
        train_array = pca.transform(train_array)
        val_array   = pca.transform(val_array)
        if test_array:
            test_array  = pca.transform(test_array)
        return train_array, val_array, test_array, scaler, pca

    return train_array, val_array, test_array, scaler

def train_val_test_split(
    data_df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float
    ):

    if train_size + val_size + test_size != 1:
        raise Exception("train_size + val_size + test_size != 1")

    train_df, val_test_df = train_test_split(
        data_df,
        train_size=train_size,
        shuffle=False)

    if test_size == 0:
        return train_df, val_test_df, pd.DataFrame()
    
    val_df, test_df = train_test_split(
        val_test_df,
        train_size=(val_size/(val_size+test_size)),
        shuffle=False)

    return train_df, val_df, test_df

def featureGeneration(
    data,
    sma_start: int = 20,
    sma_end  : int = 100,
    sma_step : int = 20,
    ema_start: int = 20,
    ema_end  : int = 100,
    ema_step : int = 20
    ):
    """
    Input
    --------
    X: pd.DataFrame


    Output
    --------
    X: pd.DataFrame
    """
    close = data.loc[:, 'Close']

    # SMA
    data = addSMA(data=data, sma_start=sma_start, sma_end=sma_end, sma_step=sma_step)

    # EMA
    data = addEMA(data=data, ema_start=ema_start, ema_end=ema_end, ema_step=ema_step)

    # RSI
    data.loc[:,'RSI'] = rsi(data=data)

    # MACD
    data.loc[:,'MACD'] = macd(data=data)

    # ATR
    data.loc[:,'ATR'] = atr(data=data)

    # BollingerBands
    upper = signal_h(data.Close)
    lower = signal_l(data.Close)

    data.loc[:,'BB_upper'] = (upper - close) / close
    data.loc[:,'BB_width'] = (upper - lower) / close
    data.loc[:,'BB_lower'] = (lower - close) / close

    # Garman Klass Volatility
    data.loc[:,'garman_klass_vol'] = ((np.log(data['High'])-np.log(data['Low']))**2)/2-(2*np.log(2)-1)*((np.log(data['Close'])-np.log(data['Open']))**2)

    # Output
    OHLC_cols = ['Open', 'High', 'Low', 'Close']
    data = data.drop(OHLC_cols, axis=1).dropna().astype(float)

    return data

