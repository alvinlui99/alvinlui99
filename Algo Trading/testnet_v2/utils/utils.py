from typing import Dict
import datetime
import pandas as pd
import numpy as np
import os

from config import ModelConfig


def convert_str_to_datetime(date_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').replace(hour=0, minute=0, second=0, tzinfo=datetime.timezone.utc)

def save_historical_klines(data: dict[str, pd.DataFrame], path: str, interval: str):
    for symbol, df in data.items():
        df.to_csv(f"{path}/{interval}/{symbol}.csv", index=False)

def read_klines_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(by='timestamp')
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    return df[['timestamp'] + numeric_columns]

def read_klines_from_csv(path: str, interval: str,
                         start_date: datetime.datetime = None,
                         end_date: datetime.datetime = None) -> dict[str, pd.DataFrame]:
    folder = f"{path}/{interval}"
    data = {}
    for symbol in os.listdir(folder):
        symbol = symbol.split('.')[0]
        df = pd.read_csv(f"{folder}/{symbol}.csv")
        df = read_klines_to_df(df)
        df.set_index('timestamp', inplace=True)
        if start_date:
            start_date = pd.to_datetime(start_date, utc=True)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date, utc=True)
            df = df[df.index <= end_date]
        data[symbol] = df
    return data

def get_target(data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {symbol: data[symbol]['close'].pct_change() for symbol in data.keys()}

def train_model(train_data, val_data):
    from models.LGBM_Regressor.model_LGBM_Regressor import LGBMRegressorModel
    model = LGBMRegressorModel()
    model.configure()

    train_target = get_target(train_data)
    val_target = get_target(val_data)
    model.train(train_data, train_target, val_data, val_target)
    
    return model

def save_model():
    pass

def data_df_to_nparray(data: Dict[str, pd.DataFrame]) -> np.ndarray:
    symbols = ModelConfig.GeneralConfig.SYMBOLS
    features = ModelConfig.GeneralConfig.FEATURES
    
    data = remove_incomplete_rows(data)
    # Create a list to hold the ordered dataframes
    ordered_dataframes = []
    
    # Iterate over each symbol and feature to ensure consistent order
    for symbol in symbols:
        if symbol in data:
            df = data[symbol].reset_index(drop=True)
            # Append each feature in the specified order
            ordered_dataframes.append(df[features])
        else:
            raise KeyError(f"Symbol {symbol} not found in data.")
    
    # Concatenate all dataframes horizontally
    concatenated_df = pd.concat(ordered_dataframes, axis=1)
    
    return concatenated_df.to_numpy()

def remove_incomplete_rows(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Create a boolean mask for complete rows based on the union of all indices
    all_indices = pd.Index([])
    for df in data.values():
        all_indices = all_indices.union(df.index)
    
    complete_mask = pd.Series(True, index=all_indices, dtype=bool)
    
    # Iterate over each DataFrame to update the mask
    for df in data.values():
        # Align the mask with the current DataFrame's index
        aligned_mask = complete_mask.reindex(df.index, fill_value=False)
        # Update the mask to False where any NaN values are found
        aligned_mask &= df.notna().all(axis=1)
        complete_mask.loc[aligned_mask.index] = aligned_mask
    
    # Filter each DataFrame using the complete mask
    filtered_data = {symbol: df[complete_mask.reindex(df.index, fill_value=False)] for symbol, df in data.items()}
    
    return filtered_data
