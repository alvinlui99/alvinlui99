import os
import sys
import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Local application imports
from config import BaseConfig, ModelConfig, DataConfig, setup_logging
from model.LGBMmodel import LGBMmodel
from utils.feature_engineering import FeatureEngineer
from utils import utils

# Set up logging
logger = setup_logging()

def load_data(symbols=None):
    """
    Load data for the specified symbols from csv files
    
    Args:
        symbols: List of symbols to load data for. If None, load all available symbols.
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if symbols is None:
        symbols = BaseConfig.SYMBOLS
    
    logger.info(f"Loading data for symbols: {symbols}")
    
    dfs = {}
    for symbol in symbols:
        # First try with the 1h timeframe format
        filename = f"data/{symbol}_{DataConfig.DEFAULT_TIMEFRAME}.csv"
        if os.path.exists(filename):
            logger.info(f"Loading data from {filename}")
            df = pd.read_csv(filename)
            
            # Check if dataframe is empty or has only header
            if len(df) <= 1:
                logger.warning(f"File {filename} is empty or has only header. Skipping.")
                continue
            
            # Convert timestamp to datetime if it's a string
            if 'Open_time' in df.columns and isinstance(df['Open_time'].iloc[0], str):
                df['Open_time'] = pd.to_datetime(df['Open_time'])
                
            dfs[symbol] = df
        else:
            # Try with the klines format
            filename = f"data/klines_{symbol}.csv"
            if os.path.exists(filename):
                logger.info(f"Loading data from {filename}")
                df = pd.read_csv(filename)
                
                # Check if dataframe is empty or has only header
                if len(df) <= 1:
                    logger.warning(f"File {filename} is empty or has only header. Skipping.")
                    continue
                
                # Rename columns to match expected format if needed
                if 'Open time' in df.columns:
                    df = df.rename(columns={
                        'Open time': 'Open_time',
                        'Close time': 'Close_time',
                        'Quote asset volume': 'Quote_asset_volume',
                        'Number of trades': 'Number_of_trades',
                        'Taker buy base asset volume': 'Taker_buy_base_asset_volume',
                        'Taker buy quote asset volume': 'Taker_buy_quote_asset_volume'
                    })
                
                # Convert timestamp to datetime if needed
                if 'Open_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Open_time']):
                    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
                
                dfs[symbol] = df
            else:
                logger.warning(f"No data file found for symbol {symbol}")
    
    return dfs

def split_data_for_training_and_backtesting(dfs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split data into training, validation, and test sets, leaving test set for future backtesting
    
    Args:
        dfs: Dictionary of DataFrames with OHLCV data
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to reserve for backtesting
        
    Returns:
        Tuple of dictionaries containing train, validation, and test DataFrames
    """
    if train_ratio + val_ratio + test_ratio != 1.0:
        logger.warning(f"Split ratios don't sum to 1.0: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        logger.info(f"Normalized ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}
    
    for symbol, df in dfs.items():
        try:
            # Ensure data is sorted by time
            time_col = 'Open_time'
            if time_col in df.columns:
                df = df.sort_values(time_col)
            
            # Calculate split indices
            n = len(df)
            train_idx = int(n * train_ratio)
            val_idx = train_idx + int(n * val_ratio)
            
            # Make sure we have enough data for each split
            if train_idx < 50:
                logger.warning(f"Not enough data for {symbol}: only {train_idx} training rows")
                if train_idx < 20:  # Absolute minimum
                    logger.error(f"Insufficient data for {symbol}: skipping")
                    continue
            
            # Split the data
            train_df = df.iloc[:train_idx]
            val_df = df.iloc[train_idx:val_idx]
            test_df = df.iloc[val_idx:]
            
            # Validate that we have data in all splits
            if len(train_df) == 0:
                logger.error(f"No training data available for {symbol} after splitting!")
                continue
                
            if len(val_df) == 0:
                logger.error(f"No validation data available for {symbol} after splitting!")
                continue
                
            if len(test_df) == 0:
                logger.warning(f"No test data available for {symbol} after splitting!")
            
            # Log split sizes
            logger.info(f"Split for {symbol}: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
            
            # Store the splits
            train_dfs[symbol] = train_df
            val_dfs[symbol] = val_df
            test_dfs[symbol] = test_df
        except Exception as e:
            logger.error(f"Error splitting data for {symbol}: {str(e)}")
    
    # Log counts of dataframes
    logger.info(f"After splitting: {len(train_dfs)} symbols in train, {len(val_dfs)} in validation, {len(test_dfs)} in test")
    
    # Return the three splits
    return train_dfs, val_dfs, test_dfs

def prepare_data_for_training(train_dfs, val_dfs):
    """
    Apply feature engineering to prepare data for training
    
    Args:
        train_dfs: Dictionary of training DataFrames
        val_dfs: Dictionary of validation DataFrames
        
    Returns:
        Processed training and validation DataFrames
    """
    feature_engineer = FeatureEngineer(ModelConfig.FEATURE_CONFIG, logger=logger)
    
    processed_train = {}
    processed_val = {}
    
    for symbol in train_dfs.keys():
        try:
            # Process training data
            processed_train[symbol] = feature_engineer.add_all_features(train_dfs[symbol].copy())
            # Drop any NaN rows or fill with 0
            processed_train[symbol] = processed_train[symbol].fillna(0)
            
            # Process validation data
            processed_val[symbol] = feature_engineer.add_all_features(val_dfs[symbol].copy())
            # Drop any NaN rows or fill with 0
            processed_val[symbol] = processed_val[symbol].fillna(0)
            
            logger.info(f"Processed data for {symbol}:")
            logger.info(f"  Train: {processed_train[symbol].shape}, NaN values: {processed_train[symbol].isna().sum().sum()}")
            logger.info(f"  Val: {processed_val[symbol].shape}, NaN values: {processed_val[symbol].isna().sum().sum()}")
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            continue
    
    return processed_train, processed_val

def save_test_data_for_backtesting(test_dfs, output_dir):
    """
    Save the test data for later backtesting
    
    Args:
        test_dfs: Dictionary of test DataFrames
        output_dir: Directory to save the test data
    """
    os.makedirs(output_dir, exist_ok=True)
    for symbol, df in test_dfs.items():
        output_file = os.path.join(output_dir, f"{symbol}_backtest.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved backtest data for {symbol} to {output_file}")

def main():
    """Main function to retrain the model"""
    print("\n" + "="*80)
    print("RETRAINING TRADING MODEL WITH SYMBOLS FROM CONFIG")
    print("="*80 + "\n")
    
    # Create output directories
    model_dir = BaseConfig.MODEL_DIR
    backtest_dir = os.path.join(BaseConfig.DATA_DIR, "backtest")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(backtest_dir, exist_ok=True)
    
    # 1. Load data for all symbols in config
    symbols = BaseConfig.SYMBOLS
    dfs = load_data(symbols)
    
    if not dfs:
        logger.error("No data available for any symbols! Aborting.")
        print("\n❌ No data available for any symbols! Aborting.")
        return
    
    print(f"\nLoaded data for {len(dfs)} symbols:")
    for symbol, df in dfs.items():
        print(f"- {symbol}: {len(df)} rows")
    
    # 2. Split data into train/val/test (reserve test for backtesting)
    train_dfs, val_dfs, test_dfs = split_data_for_training_and_backtesting(
        dfs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    # Save test data for future backtesting
    save_test_data_for_backtesting(test_dfs, backtest_dir)
    print(f"\nSaved test data for backtesting in {backtest_dir}")
    
    # 3. Prepare data for training (apply feature engineering)
    processed_train, processed_val = prepare_data_for_training(train_dfs, val_dfs)
    
    if not processed_train:
        logger.error("No processed data available for training! Aborting.")
        print("\n❌ No processed data available for training! Aborting.")
        return
    
    # 4. Initialize and train model
    print("\nInitializing model...")
    available_symbols = list(processed_train.keys())
    model = LGBMmodel(available_symbols, feature_config=ModelConfig.FEATURE_CONFIG, logger=logger)
    
    try:
        print("\nTraining model...")
        model.train(processed_train, processed_val)
        print(f"✅ Model training completed successfully with {len(available_symbols)} symbols")
        
        # 5. Save trained model
        print(f"\nSaving trained model to {model_dir}...")
        model.save_model(model_dir)
        
        # Save list of trained symbols for reference
        with open(os.path.join(model_dir, 'trained_symbols.txt'), 'w') as f:
            f.write('\n'.join(available_symbols))
        
        print(f"✅ Model saved successfully to {model_dir}")
        print(f"✅ Trained using symbols: {', '.join(available_symbols)}")
        print("\nNext step: Use the trained model for backtesting with the reserved test data")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        print(f"\n❌ Error training model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 