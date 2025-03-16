"""
Simplified test script to train the model with a small dataset.
This script uses a more controlled approach with a small sample of data
to help diagnose and fix the empty dataset issue.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Local imports
from config import BaseConfig, ModelConfig, DataConfig
from model.LGBMmodel import LGBMmodel
from utils.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/test_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def create_test_dataset():
    """
    Create a small, clean test dataset for each symbol.
    This ensures we have valid data for training.
    """
    logger.info("Creating test datasets...")
    
    # Use only two symbols for simplicity
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    
    # Dictionary to hold data frames
    dfs = {}
    
    for symbol in test_symbols:
        # Try both file formats
        live_format = f'data/{symbol}_1h.csv'
        train_format = f'data/klines_{symbol}.csv'
        
        filename = None
        if os.path.exists(live_format):
            filename = live_format
            logger.info(f"Using live format data for {symbol}")
        elif os.path.exists(train_format):
            filename = train_format
            logger.info(f"Using training format data for {symbol}")
        else:
            logger.warning(f"No data found for {symbol}, skipping")
            continue
        
        # Load the data
        try:
            df = pd.read_csv(filename)
            
            if len(df) == 0:
                logger.warning(f"Empty dataframe for {symbol}, skipping")
                continue
                
            logger.info(f"Loaded {len(df)} rows for {symbol}")
            
            # Take just a small sample for testing
            if len(df) > 500:
                df = df.iloc[-500:].copy()
                logger.info(f"Using last 500 rows for {symbol}")
            
            # Standardize column names if needed
            if 'Open time' in df.columns:
                logger.info(f"Converting column names for {symbol}")
                df = df.rename(columns={
                    'Open time': 'Open_time',
                    'Close time': 'Close_time',
                    'Quote asset volume': 'Quote_asset_volume',
                    'Number of trades': 'Number_of_trades',
                    'Taker buy base asset volume': 'Taker_buy_base_asset_volume',
                    'Taker buy quote asset volume': 'Taker_buy_quote_asset_volume'
                })
            
            # Convert types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean any NaN values
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.info(f"Dropping {nan_count} NaN values from {symbol} data")
                df = df.dropna()
            
            # Remove Return columns if that's what we want for model compatibility
            if DataConfig.REMOVE_RETURNS_FOR_MODEL:
                if 'Return' in df.columns:
                    df = df.drop(['Return'], axis=1)
                if 'Log_Return' in df.columns:
                    df = df.drop(['Log_Return'], axis=1)
                logger.info(f"Removed Return columns from {symbol} for model compatibility")
            
            # Store the processed dataframe
            if len(df) >= 100:  # Ensure we have enough data
                dfs[symbol] = df
                logger.info(f"Prepared {len(df)} rows for {symbol}")
            else:
                logger.warning(f"Not enough data for {symbol} after cleaning: {len(df)} rows")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    return dfs

def main():
    """Main function to test training with a small dataset."""
    logger.info("=" * 80)
    logger.info("TESTING MODEL TRAINING WITH SMALL DATASET")
    logger.info("=" * 80)
    
    # Create test datasets
    dfs = create_test_dataset()
    
    if not dfs:
        logger.error("Failed to create any test datasets!")
        print("\n❌ Failed to create any test datasets! Check your data files.")
        return
    
    logger.info(f"Created test datasets for {len(dfs)} symbols: {list(dfs.keys())}")
    
    # Split into train/val/test
    logger.info("Splitting data...")
    symbols = list(dfs.keys())
    
    train_dfs = {}
    val_dfs = {}
    
    for symbol, df in dfs.items():
        # Use a simple 80/20 split
        split_idx = int(len(df) * 0.8)
        
        train_dfs[symbol] = df.iloc[:split_idx].copy()
        val_dfs[symbol] = df.iloc[split_idx:].copy()
        
        logger.info(f"Split {symbol}: train={len(train_dfs[symbol])}, val={len(val_dfs[symbol])}")
    
    # Verify we have data
    for symbol in symbols:
        logger.info(f"Train data for {symbol}: {train_dfs[symbol].shape}")
        logger.info(f"Val data for {symbol}: {val_dfs[symbol].shape}")
    
    # Apply feature engineering
    logger.info("Applying feature engineering...")
    feature_engineer = FeatureEngineer(ModelConfig.FEATURE_CONFIG)
    
    try:
        # Process one sample to check for errors
        symbol = symbols[0]
        sample = train_dfs[symbol].head(10).copy()
        processed = feature_engineer.add_all_features(sample)
        logger.info(f"Feature engineering test successful: {processed.shape}")
        logger.info(f"Features: {list(processed.columns)}")
    except Exception as e:
        logger.error(f"Feature engineering test failed: {str(e)}")
        print(f"\n❌ Feature engineering failed: {str(e)}")
        return
    
    # Process all data
    processed_train = {}
    processed_val = {}
    
    for symbol in symbols:
        try:
            # Process training data
            processed_train[symbol] = feature_engineer.add_all_features(train_dfs[symbol].copy())
            # Drop any NaN rows
            processed_train[symbol] = processed_train[symbol].dropna()
            
            # Process validation data
            processed_val[symbol] = feature_engineer.add_all_features(val_dfs[symbol].copy())
            # Drop any NaN rows
            processed_val[symbol] = processed_val[symbol].dropna()
            
            logger.info(f"Processed data for {symbol}:")
            logger.info(f"  Train: {processed_train[symbol].shape}, NaN values: {processed_train[symbol].isna().sum().sum()}")
            logger.info(f"  Val: {processed_val[symbol].shape}, NaN values: {processed_val[symbol].isna().sum().sum()}")
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            print(f"\n❌ Error processing data for {symbol}: {str(e)}")
            return
    
    # Initialize and train model
    logger.info("Initializing model...")
    model = LGBMmodel(symbols, feature_config=ModelConfig.FEATURE_CONFIG, logger=logger)
    
    logger.info("Training model...")
    try:
        model.train(processed_train, processed_val)
        logger.info("Model training completed successfully!")
        
        # Log feature counts
        if hasattr(model, 'feature_count'):
            logger.info(f"Model trained with {model.feature_count} total features")
            
        # Save model to test directory
        test_model_dir = os.path.join(BaseConfig.MODEL_DIR, "test_model")
        os.makedirs(test_model_dir, exist_ok=True)
        model.save_model(test_model_dir)
        logger.info(f"Model saved to {test_model_dir}")
        
        # Test model
        predictions = model.predict(processed_val)
        logger.info(f"Model predictions successful: {len(predictions)} symbols predicted")
        
        print("\n✅ Test training completed successfully!")
        print(f"Model saved to: {test_model_dir}")
        print("Now you can run the full retrain_model.py script")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        
        # Print debug info
        for symbol in symbols:
            if symbol in processed_train:
                logger.info(f"Training data for {symbol}:")
                logger.info(f"  Shape: {processed_train[symbol].shape}")
                logger.info(f"  NaN values: {processed_train[symbol].isna().sum().sum()}")
                logger.info(f"  Column dtypes: {processed_train[symbol].dtypes}")
            
            if symbol in processed_val:
                logger.info(f"Validation data for {symbol}:")
                logger.info(f"  Shape: {processed_val[symbol].shape}")
                logger.info(f"  NaN values: {processed_val[symbol].isna().sum().sum()}")
                logger.info(f"  Column dtypes: {processed_val[symbol].dtypes}")
                
        print(f"\n❌ Model training failed: {str(e)}")
        print("Check the logs for details.")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during test training: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1) 