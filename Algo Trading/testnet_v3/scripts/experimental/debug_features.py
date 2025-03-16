import os
import pandas as pd
import numpy as np
from utils.feature_engineering import FeatureEngineer
import logging
from model.LGBMmodel import LGBMmodel
from config import ModelConfig, BaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load training data
def load_training_data(symbols):
    dfs = {}
    for symbol in symbols:
        filename = f'data/klines_{symbol}.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            # Rename columns to match live data format
            if 'Open time' in df.columns:
                df = df.rename(columns={
                    'Open time': 'Open_time',
                    'Close time': 'Close_time',
                    'Quote asset volume': 'Quote_asset_volume',
                    'Number of trades': 'Number_of_trades',
                    'Taker buy base asset volume': 'Taker_buy_base_asset_volume',
                    'Taker buy quote asset volume': 'Taker_buy_quote_asset_volume'
                })
            dfs[symbol] = df
            logger.info(f"Loaded training data for {symbol}, shape: {df.shape}")
    return dfs

# Function to load live data
def load_live_data(symbols):
    dfs = {}
    for symbol in symbols:
        filename = f'data/{symbol}_1h.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            dfs[symbol] = df
            logger.info(f"Loaded live data for {symbol}, shape: {df.shape}")
    return dfs

# Main function to analyze differences
def analyze_feature_differences():
    # Use a smaller set of symbols for testing
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    # Load data
    training_data = load_training_data(test_symbols)
    live_data = load_live_data(test_symbols)
    
    # Check which symbols are available in both datasets
    common_symbols = [s for s in test_symbols if s in training_data and s in live_data]
    logger.info(f"Common symbols: {common_symbols}")
    
    if not common_symbols:
        logger.error("No common symbols found in both datasets!")
        return
    
    # Create feature engineer with default config
    feature_engineer = FeatureEngineer(ModelConfig.FEATURE_CONFIG)
    
    # Process training data
    training_processed = {}
    for symbol in common_symbols:
        df = training_data[symbol].copy()
        # Convert types for TA-Lib
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        try:
            # Process with feature engineering
            processed_df = feature_engineer.add_all_features(df)
            training_processed[symbol] = processed_df
            logger.info(f"Training data features for {symbol}: {processed_df.shape[1]}")
            logger.info(f"Training data columns: {list(processed_df.columns)[:10]}...")
        except Exception as e:
            logger.error(f"Error processing training data for {symbol}: {str(e)}")
    
    # Process live data
    live_processed = {}
    for symbol in common_symbols:
        df = live_data[symbol].copy()
        # Convert types for TA-Lib
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        try:
            # Process with feature engineering
            processed_df = feature_engineer.add_all_features(df)
            live_processed[symbol] = processed_df
            logger.info(f"Live data features for {symbol}: {processed_df.shape[1]}")
            logger.info(f"Live data columns: {list(processed_df.columns)[:10]}...")
        except Exception as e:
            logger.error(f"Error processing live data for {symbol}: {str(e)}")
    
    # Compare feature counts
    if training_processed and live_processed:
        # Count per-symbol features
        training_features = sum(df.shape[1] for df in training_processed.values())
        live_features = sum(df.shape[1] for df in live_processed.values())
        
        # Calculate expected feature counts in concatenated data
        logger.info(f"Training data: {len(common_symbols)} symbols × " 
                   f"{training_processed[common_symbols[0]].shape[1]} features/symbol = "
                   f"{training_features} total features")
        logger.info(f"Live data: {len(common_symbols)} symbols × "
                   f"{live_processed[common_symbols[0]].shape[1]} features/symbol = "
                   f"{live_features} total features")
        
        # Find differences in column names
        training_cols = set(training_processed[common_symbols[0]].columns)
        live_cols = set(live_processed[common_symbols[0]].columns)
        
        if training_cols != live_cols:
            logger.info(f"Columns in training but not in live: {training_cols - live_cols}")
            logger.info(f"Columns in live but not in training: {live_cols - training_cols}")

if __name__ == "__main__":
    analyze_feature_differences() 