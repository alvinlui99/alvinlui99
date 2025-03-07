import datetime
import pandas as pd

def convert_date_to_milliseconds(date_str):
    """
    Convert a date string in the format 'YYYY-MM-DD' to a timestamp in milliseconds.
    """
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

def convert_milliseconds_to_date(milliseconds):
    """
    Convert a timestamp in milliseconds to a date string in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    dt = datetime.datetime.fromtimestamp(milliseconds / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_klines_from_symbols(client, symbols: list[str], timeframe: str,
                            startTime: str = None, endTime: str = None) -> dict[str, pd.DataFrame]:
    """
    Get klines from multiple symbols and timeframe, returning a dictionary of DataFrames.
    """
    all_klines = {}
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
               'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
               'Taker buy quote asset volume', 'Ignore']
    for symbol in symbols:
        if startTime is None and endTime is None:
            klines_list = client.klines(
                symbol=symbol,
                interval=timeframe,
                limit=500)
        else:
            start = convert_date_to_milliseconds(startTime)
            end = convert_date_to_milliseconds(endTime)
            klines_list = []
            while start < end:
                klines = client.klines(
                    symbol=symbol,
                    startTime=start,
                    endTime=end,
                    interval=timeframe,
                    limit=500)
                if not klines:
                    break
                klines_list.extend(klines)
                start = klines[-1][0] + 1
        
        # Convert to DataFrame
        all_klines[symbol] = pd.DataFrame(klines_list, columns=columns)
    return all_klines

def save_dfs_to_csv(dfs: dict[str, pd.DataFrame], filename: str) -> None:
    """
    Save klines to a CSV file using pandas.
    """
    for symbol, df in dfs.items():
        filename_symbol = filename.replace('.csv', f'_{symbol}.csv')
        df.to_csv(filename_symbol, index=False)

def load_dfs_from_csv(symbols: list[str], filename: str) -> dict[str, pd.DataFrame]:
    """
    Load klines from a CSV file using pandas.
    """
    dfs = {}
    for symbol in symbols:
        filename_symbol = filename.replace('.csv', f'_{symbol}.csv')
        dfs[symbol] = pd.read_csv(filename_symbol)
    return dfs

def split_dfs(dfs: dict[str, pd.DataFrame],
              val_ratio: float = 0.15,
              test_ratio: float = 0.15) -> tuple[dict[str, pd.DataFrame],
                                                 dict[str, pd.DataFrame],
                                                 dict[str, pd.DataFrame]]:
    """
    Split the dfs into training and validation sets.
    """
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}

    common_index = None
    for df in dfs.values():
        df = df.set_index('Open time')[['Open', 'High', 'Low', 'Close', 'Volume']]
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    train_size = int(len(common_index) * (1 - val_ratio - test_ratio))
    val_size = int(len(common_index) * val_ratio)
    test_size = int(len(common_index) * test_ratio)

    for symbol, df in dfs.items():
        df = df.set_index('Open time')[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.reindex(common_index)
        train_dfs[symbol] = df.iloc[:train_size]
        val_dfs[symbol] = df.iloc[train_size:train_size + val_size]
        test_dfs[symbol] = df.iloc[train_size + val_size:]

    return train_dfs, val_dfs, test_dfs
