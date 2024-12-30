import pandas as pd
from typing import List, Tuple
from config import DATA_PATH
import numpy as np
from config import ModelConfig
from trading_types import HistoricalData, MarketData

def load_and_split_data(symbols: List[str], start_date: str, end_date: str, model_config: ModelConfig) -> MarketData:
        """
        Load historical data and split into train, validation and test sets.
        
        Args:
            symbols: List of trading pair symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            MarketData containing train, validation, test and full datasets
        """
        # Load all historical data
        historical_data: HistoricalData = load_historical_data(symbols, start_date, end_date)
        
        # Split data into train, validation, and test sets
        print("\nSplitting data...")
        train_data, val_data, test_data = split_data_by_date(historical_data,
                                                            model_config.TRAIN_SIZE, 
                                                            model_config.VALIDATION_SIZE)
        
        return MarketData(
            train_data=train_data,
            val_data=val_data, 
            test_data=test_data,
            full_data=historical_data
        )

def load_historical_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load and prepare historical data for training"""
    all_data = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}:")
        df = pd.read_csv(f"{DATA_PATH}/{symbol}.csv")
        print(f"  Raw data shape: {df.shape}")
        
        df['datetime'] = pd.to_datetime(df['index'])
        df.set_index('datetime', inplace=True)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Create numeric features only
        symbol_data = pd.DataFrame(index=df.index)
        symbol_data[f'{symbol}_price'] = df['Close'].astype(float)
        symbol_data[f'{symbol}_volume'] = df['Volume'].astype(float)
        symbol_data[f'{symbol}_return'] = df['Close'].pct_change().astype(float)
        
        # Fill NaN values
        symbol_data = symbol_data.ffill().bfill()
        print(f"  Final shape: {symbol_data.shape}")
        
        all_data.append(symbol_data)
    
    # Combine all data
    combined_data = pd.concat(all_data, axis=1)
    
    # Ensure all data is numeric
    numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
    combined_data = combined_data[numeric_cols]
    
    return combined_data

def split_data_by_date(data: pd.DataFrame, train_size: float, validation_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets by date"""
    data = data.sort_index()
    n = len(data)
    
    train_end = int(n * train_size)
    val_end = train_end + int(n * validation_size)
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    print(f"\nData split summary:")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Validation period: {val_data.index[0]} to {val_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    
    return train_data, val_data, test_data 