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
    Split the dataframes into training, validation, and test sets.
    Uses forward fill to handle missing dates and ensures all dataframes have the same date range.
    
    Args:
        dfs: Dictionary of DataFrames with OHLCV data
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        
    Returns:
        Tuple of dictionaries containing train, validation, and test DataFrames
    """
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}

    # Create a union of all dates across all dataframes
    all_dates = set()
    processed_dfs = {}
    
    for symbol, df in dfs.items():
        # Convert to DataFrame if Series
        if isinstance(df, pd.Series):
            df = df.to_frame()
            
        # Ensure we have the right column name for the timestamp
        time_col = 'Open time' if 'Open time' in df.columns else 'Open_time'
        
        if time_col in df.columns:
            # Set the datetime index
            df_indexed = df.set_index(time_col)[['Open', 'High', 'Low', 'Close', 'Volume']]
            processed_dfs[symbol] = df_indexed
            
            # Add all dates from this dataframe to the set of all dates
            all_dates.update(df_indexed.index)
        else:
            raise ValueError(f"DataFrame for {symbol} does not have a valid timestamp column")
    
    # Sort all dates to ensure chronological order
    all_dates = sorted(all_dates)
    
    # Reindex all dataframes to include all dates, using forward fill
    aligned_dfs = {}
    for symbol, df in processed_dfs.items():
        # Reindex to include all dates
        aligned_df = df.reindex(all_dates)
        
        # Forward fill missing values (use the last known price)
        aligned_df = aligned_df.ffill()
        
        # In case there are still NaNs at the beginning (no previous value to fill with),
        # backward fill those
        aligned_df = aligned_df.bfill()
        
        aligned_dfs[symbol] = aligned_df
    
    # Calculate sizes for train/val/test splits
    total_size = len(all_dates)
    train_size = int(total_size * (1 - val_ratio - test_ratio))
    val_size = int(total_size * val_ratio)
    # test_size is the remainder
    
    # Split each aligned dataframe
    for symbol, df in aligned_dfs.items():
        train_dfs[symbol] = df.iloc[:train_size].copy()
        val_dfs[symbol] = df.iloc[train_size:train_size + val_size].copy()
        test_dfs[symbol] = df.iloc[train_size + val_size:].copy()
        
        # Reset index to put the datetime back as a column if needed
        # train_dfs[symbol].reset_index(inplace=True)
        # val_dfs[symbol].reset_index(inplace=True)
        # test_dfs[symbol].reset_index(inplace=True)
    
    return train_dfs, val_dfs, test_dfs
