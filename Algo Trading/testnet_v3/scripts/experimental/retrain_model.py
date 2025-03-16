"""
Retrain Model Script - Feature Count Mismatch Solution

This script retrains the model with production-identical data processing to ensure
feature count consistency between training and inference.

Key issues being addressed:
1. Training data was missing Return and Log_Return columns
2. Different symbols were used during training versus inference
3. Feature count mismatch between training and prediction

Feature Standardization Approach:
---------------------------------
This script implements a robust solution to the feature count mismatch problem by:

1. Creating a standardized feature pipeline that's used consistently between 
   training and inference
   
2. Saving the feature configuration during training, allowing it to be loaded
   during inference to ensure identical feature sets
   
3. Implementing feature alignment that ensures live prediction data matches
   exactly what the model expects (adding missing features, removing extra ones)

Usage for Training:
------------------
python retrain_model.py --symbols BTCUSDT ETHUSDT --min-symbols 1

Usage for Inference (in your prediction scripts):
-----------------------------------------------
```python
from retrain_model import standardize_live_data_for_prediction

# Load live data
live_dfs = {...}  # Your live data loading code

# Standardize through the same pipeline used in training
model_dir = "model/trained_models"
standardized_dfs = standardize_live_data_for_prediction(model_dir, live_dfs)

# Make predictions with the standardized data
model = LGBMmodel.load(model_dir)
predictions = model.predict(standardized_dfs)
```

This approach ensures feature engineering consistency across the entire ML lifecycle.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import json
import traceback

# Local imports
from config import BaseConfig, ModelConfig, DataConfig
from model.LGBMmodel import LGBMmodel
from utils.feature_engineering import FeatureEngineer

# Setup enhanced logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/retrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def load_data(symbols, data_dir="data"):
    """Load and preprocess historical data for the specified symbols"""
    logger.info(f"Loading data for symbols: {symbols}")
    
    dfs = {}
    for symbol in symbols:
        try:
            # Try both file formats
            live_format = os.path.join(data_dir, f"{symbol}_1h.csv")
            train_format = os.path.join(data_dir, f"klines_{symbol}.csv")
            
            csv_path = None
            if os.path.exists(live_format):
                csv_path = live_format
                logger.info(f"Using live format data for {symbol}")
            elif os.path.exists(train_format):
                csv_path = train_format
                logger.info(f"Using training format data for {symbol}")
            else:
                logger.error(f"Data file for {symbol} not found")
                continue
                
            # Load the data
            df = pd.read_csv(csv_path)
            
            # Check if we have enough data
            if len(df) < 100:
                logger.warning(f"Not enough data for {symbol}, only {len(df)} rows available")
                if len(df) < 50:  # Absolute minimum needed
                    continue
            
            # Use much more data for training
            # Use at least 5000 rows if available
            if len(df) > 5000:
                logger.info(f"Using 5000 rows out of {len(df)} for {symbol}")
                df = df.iloc[-5000:]
            else:
                logger.info(f"Using all {len(df)} available rows for {symbol}")
                
            # Basic validation
            if 'Open time' in df.columns:
                logger.info(f"Converting column names for {symbol}")
                df = df.rename(columns={
                    'Open time': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Close time': 'close_time',
                    'Quote asset volume': 'quote_volume',
                    'Number of trades': 'num_trades',
                    'Taker buy base asset volume': 'taker_buy_volume',
                    'Taker buy quote asset volume': 'taker_buy_quote_volume'
                })
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
            if missing:
                logger.error(f"Missing required columns for {symbol}: {missing}")
                continue
                
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                # Find the actual column name (case insensitive)
                col_match = next((c for c in df.columns if c.lower() == col.lower()), None)
                if col_match:
                    df[col_match] = pd.to_numeric(df[col_match], errors='coerce')
                
            # Make sure timestamp is the index
            if 'timestamp' in df.columns:
                # Try to convert timestamp to datetime
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    # Try millisecond timestamp format
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    except Exception as e:
                        logger.warning(f"Could not convert timestamp for {symbol}: {str(e)}")
                
                df.set_index('timestamp', inplace=True)
                
            # Remove Return columns for model compatibility
            if 'Return' in df.columns:
                logger.info(f"Removing Return column from {symbol} data")
                df = df.drop('Return', axis=1)
            
            if 'Log_Return' in df.columns:
                logger.info(f"Removing Log_Return column from {symbol} data")
                df = df.drop('Log_Return', axis=1)
                
            # Store the dataframe
            dfs[symbol] = df
            logger.info(f"Loaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            
    if not dfs:
        raise ValueError("No valid data loaded for any symbols")
        
    return dfs

def split_data(dfs, train_ratio=0.7, val_ratio=0.15):
    """
    Split the data into training, validation, and test sets.
    Returns dictionaries of dataframes for each split.
    """
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}
    
    for symbol, df in dfs.items():
        try:
            # Ensure data is sorted by time
            if df.index.name == 'timestamp':
                df = df.sort_index()
            elif 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
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

def prepare_for_lightgbm(train_dfs, val_dfs):
    """
    Prepare the dataframes for LightGBM training, ensuring feature consistency
    and handling NaN values properly.
    """
    logger.info("Preparing data for LightGBM training")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer()
    
    # Process training data
    train_processed = {}
    train_feature_counts = {}
    for symbol, df in train_dfs.items():
        try:
            # Reset index to ensure feature engineering works properly
            df_reset = df.reset_index() if hasattr(df, 'reset_index') else df
            
            # Process features
            processed_df = feature_engineer.add_all_features(df_reset.copy())
            
            # CRITICAL CHANGE: Fill NaN values instead of dropping rows
            nan_count = processed_df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in training data for {symbol}")
                # First try forward fill, then backward fill
                processed_df = processed_df.ffill().bfill()
                
                # If there are still NaNs, fill with zeros
                remaining_nans = processed_df.isna().sum().sum()
                if remaining_nans > 0:
                    logger.warning(f"Still have {remaining_nans} NaN values after ffill/bfill, filling with zeros")
                    processed_df = processed_df.fillna(0)
                    
                # Verify we have no more NaNs
                final_nans = processed_df.isna().sum().sum()
                if final_nans > 0:
                    logger.error(f"Still have {final_nans} NaN values after all filling methods for {symbol}")
            
            # Check for infinity values
            if (processed_df.values == np.inf).any() or (processed_df.values == -np.inf).any():
                logger.warning(f"Found infinite values in {symbol} features, replacing with large values")
                processed_df = processed_df.replace([np.inf, -np.inf], [1e9, -1e9])
            
            # Remove non-numeric columns
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(processed_df.columns):
                logger.warning(f"Dropping {len(processed_df.columns) - len(numeric_cols)} non-numeric columns for {symbol}")
                processed_df = processed_df[numeric_cols]
            
            # Store processed dataframe and feature count
            train_processed[symbol] = processed_df
            train_feature_counts[symbol] = processed_df.shape[1]
            logger.info(f"Processed training data for {symbol}: {processed_df.shape[0]} rows, {processed_df.shape[1]} features")
        except Exception as e:
            logger.error(f"Error processing training data for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # Process validation data
    val_processed = {}
    val_feature_counts = {}
    for symbol, df in val_dfs.items():
        try:
            # Reset index to ensure feature engineering works properly
            df_reset = df.reset_index() if hasattr(df, 'reset_index') else df
            
            # Process features
            processed_df = feature_engineer.add_all_features(df_reset.copy())
            
            # CRITICAL CHANGE: Fill NaN values instead of dropping rows
            nan_count = processed_df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in validation data for {symbol}")
                # First try forward fill, then backward fill
                processed_df = processed_df.ffill().bfill()
                
                # If there are still NaNs, fill with zeros
                remaining_nans = processed_df.isna().sum().sum()
                if remaining_nans > 0:
                    logger.warning(f"Still have {remaining_nans} NaN values after ffill/bfill, filling with zeros")
                    processed_df = processed_df.fillna(0)
                    
                # Verify we have no more NaNs
                final_nans = processed_df.isna().sum().sum()
                if final_nans > 0:
                    logger.error(f"Still have {final_nans} NaN values after all filling methods for {symbol}")
            
            # Check for infinity values
            if (processed_df.values == np.inf).any() or (processed_df.values == -np.inf).any():
                logger.warning(f"Found infinite values in {symbol} features, replacing with large values")
                processed_df = processed_df.replace([np.inf, -np.inf], [1e9, -1e9])
                
            # Remove non-numeric columns
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(processed_df.columns):
                logger.warning(f"Dropping {len(processed_df.columns) - len(numeric_cols)} non-numeric columns for {symbol}")
                processed_df = processed_df[numeric_cols]
            
            # Store processed dataframe and feature count
            val_processed[symbol] = processed_df
            val_feature_counts[symbol] = processed_df.shape[1]
            logger.info(f"Processed validation data for {symbol}: {processed_df.shape[0]} rows, {processed_df.shape[1]} features")
        except Exception as e:
            logger.error(f"Error processing validation data for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # Check for common symbols between training and validation
    common_symbols = set(train_processed.keys()) & set(val_processed.keys())
    if not common_symbols:
        raise ValueError("No common symbols between training and validation sets")
    
    logger.info(f"Common symbols for training and validation: {common_symbols}")
    
    # Check feature count consistency
    for symbol in common_symbols:
        if train_feature_counts[symbol] != val_feature_counts[symbol]:
            logger.warning(f"Feature count mismatch for {symbol}: {train_feature_counts[symbol]} in training vs {val_feature_counts[symbol]} in validation")
    
    # Return only the dataframes for common symbols
    return {symbol: train_processed[symbol] for symbol in common_symbols}, {symbol: val_processed[symbol] for symbol in common_symbols}

# Add this new function to standardize feature processing
def create_standardized_feature_pipeline(symbols_to_use, feature_config=None):
    """
    Creates a standardized feature pipeline to ensure consistency between 
    training and prediction. This function creates and returns a pipeline object
    that can be used both during training and inference.
    
    Args:
        symbols_to_use: List of symbols that will be used with this pipeline
        feature_config: Optional configuration for feature engineering
        
    Returns:
        A dictionary containing:
        - 'feature_engineer': The FeatureEngineer instance
        - 'expected_features': List of expected feature names in order
        - 'feature_count': Expected number of features
        - 'symbols': List of symbols this pipeline is configured for
    """
    logger.info(f"Creating standardized feature pipeline for {len(symbols_to_use)} symbols: {symbols_to_use}")
    
    # Create the feature engineer with consistent configuration
    feature_config = feature_config or ModelConfig.FEATURE_CONFIG
    feature_engineer = FeatureEngineer(feature_config)
    
    # We need to extract the exact feature set that will be used
    # First, create tiny sample dataframes for each symbol
    sample_dfs = {}
    for symbol in symbols_to_use:
        # Create a minimal sample with the required columns
        sample_df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'close': [103.0, 102.0, 104.0, 105.0, 103.0],
            'volume': [1000, 1100, 900, 1200, 1000]
        })
        
        # Add index that looks like timestamps
        sample_df.index = pd.date_range(start='2023-01-01', periods=5, freq='1h')
        sample_df.index.name = 'timestamp'
        
        # Store
        sample_dfs[symbol] = sample_df
    
    # Process these samples exactly as we would in both training and prediction
    processed_samples = {}
    feature_counts = {}
    feature_names = {}
    
    for symbol, df in sample_dfs.items():
        try:
            # Reset index which is what we do in both training and prediction
            df_reset = df.reset_index()
            
            # Process features
            processed = feature_engineer.add_all_features(df_reset.copy())
            
            # Select only numeric features - consistent with our preprocessing
            numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
            
            # Store results
            processed_samples[symbol] = processed[numeric_cols]
            feature_counts[symbol] = len(numeric_cols)
            feature_names[symbol] = numeric_cols
            
            logger.info(f"Sample processing for {symbol} generated {len(numeric_cols)} features")
        except Exception as e:
            logger.error(f"Error in feature pipeline setup for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # Verify feature consistency across symbols
    if len(set(feature_counts.values())) > 1:
        logger.warning(f"Inconsistent feature counts across symbols: {feature_counts}")
    
    # Calculate total expected features across all symbols
    total_features = sum(feature_counts.values())
    
    # Create pipeline dictionary
    pipeline = {
        'feature_engineer': feature_engineer,
        'expected_features': feature_names,
        'feature_count': total_features,
        'feature_counts_per_symbol': feature_counts,
        'symbols': symbols_to_use,
        'version': '1.0',  # Version tracking for future changes
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Standardized feature pipeline created with {total_features} total features")
    return pipeline
    
def standardize_features_for_prediction(pipeline, dataframes):
    """
    Ensures that dataframes match the expected feature set for prediction.
    This aligns the features with what the model expects.
    
    Args:
        pipeline: The feature pipeline from create_standardized_feature_pipeline
        dataframes: Dictionary of dataframes by symbol
        
    Returns:
        Dictionary of dataframes with standardized feature sets
    """
    logger.info(f"Standardizing features for prediction on {len(dataframes)} symbols")
    
    # Process each symbol
    processed_dfs = {}
    feature_engineer = pipeline['feature_engineer']
    expected_features = pipeline['expected_features']
    
    for symbol, df in dataframes.items():
        if symbol not in pipeline['symbols']:
            logger.warning(f"Symbol {symbol} was not part of the training set. Features may be inconsistent.")
            continue
            
        try:
            # Reset index to ensure consistency
            df_reset = df.reset_index() if hasattr(df, 'reset_index') else df
            
            # Generate features
            processed = feature_engineer.add_all_features(df_reset.copy())
            
            # Handle NaN values consistently
            nan_count = processed.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values for {symbol}")
                processed = processed.ffill().bfill().fillna(0)
                
            # Handle infinity values
            if (processed.values == np.inf).any() or (processed.values == -np.inf).any():
                processed = processed.replace([np.inf, -np.inf], [1e9, -1e9])
                
            # Ensure we have the exact expected feature set
            symbol_expected_features = expected_features[symbol]
            
            # Check for missing features
            missing_features = [f for f in symbol_expected_features if f not in processed.columns]
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} expected features for {symbol}. Adding as zeros.")
                for feature in missing_features:
                    processed[feature] = 0.0
                    
            # Check for extra features
            extra_features = [f for f in processed.columns if f not in symbol_expected_features and f != 'timestamp']
            if extra_features:
                logger.warning(f"Found {len(extra_features)} extra features for {symbol}. These will be removed.")
            
            # Select only the expected features in the expected order
            final_df = processed[symbol_expected_features].copy()
            
            # Final check for NaN values
            if final_df.isna().any().any():
                logger.warning(f"Still found NaN values after standardization for {symbol}. Filling with zeros.")
                final_df = final_df.fillna(0)
                
            # Store the standardized dataframe
            processed_dfs[symbol] = final_df
            logger.info(f"Standardized features for {symbol}: {final_df.shape[0]} rows, {final_df.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error standardizing features for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            
    # Check if we have processed data for all symbols
    if len(processed_dfs) != len(dataframes):
        logger.warning(f"Only processed {len(processed_dfs)} out of {len(dataframes)} symbols")
        
    return processed_dfs

def load_feature_pipeline(model_dir, symbols=None):
    """
    Load a previously saved feature pipeline for use in inference.
    This is a helper function that should be used in production code
    to ensure consistent feature engineering between training and inference.
    
    Args:
        model_dir: Directory where the model and pipeline config are saved
        symbols: Optional list of symbols to use (defaults to trained symbols)
        
    Returns:
        A reconstructed feature pipeline that can be used with standardize_features_for_prediction
    """
    logger.info(f"Loading feature pipeline from {model_dir}")
    
    # Load the serialized pipeline configuration
    pipeline_path = os.path.join(model_dir, 'feature_pipeline_config.json')
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Feature pipeline configuration not found at {pipeline_path}")
        
    with open(pipeline_path, 'r') as f:
        pipeline_config = json.load(f)
        
    logger.info(f"Loaded pipeline config with {pipeline_config['feature_count']} expected features")
    
    # Check which symbols to use
    if not symbols:
        symbols_path = os.path.join(model_dir, 'trained_symbols.txt')
        if os.path.exists(symbols_path):
            with open(symbols_path, 'r') as f:
                symbols = f.read().strip().split('\n')
            logger.info(f"Using symbols from trained_symbols.txt: {symbols}")
        else:
            symbols = pipeline_config['symbols']
            logger.info(f"Using symbols from pipeline config: {symbols}")
            
    # Create a new feature engineer with the same configuration
    feature_config = ModelConfig.FEATURE_CONFIG  # You might need to adjust this
    feature_engineer = FeatureEngineer(feature_config)
    
    # Reconstruct the pipeline with the feature engineer
    pipeline = pipeline_config.copy()
    pipeline['feature_engineer'] = feature_engineer
    
    logger.info(f"Feature pipeline loaded for {len(pipeline['symbols'])} symbols")
    return pipeline

def standardize_live_data_for_prediction(model_dir, live_dataframes):
    """
    One-step function to standardize live data for prediction using a trained model.
    This combines loading the feature pipeline and standardizing the features.
    
    Args:
        model_dir: Directory where the model and pipeline config are saved
        live_dataframes: Dictionary of live dataframes by symbol
        
    Returns:
        Dictionary of standardized dataframes ready for prediction
    """
    logger.info(f"Standardizing {len(live_dataframes)} symbols using saved pipeline from {model_dir}")
    
    # Load the feature pipeline
    pipeline = load_feature_pipeline(model_dir, list(live_dataframes.keys()))
    
    # Standardize the features
    standardized_dfs = standardize_features_for_prediction(pipeline, live_dataframes)
    
    return standardized_dfs

def main():
    """Main function to retrain the model"""
    parser = argparse.ArgumentParser(description="Retrain the LightGBM model")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                      help="Symbols to train with")
    parser.add_argument("--min-symbols", type=int, default=1,
                      help="Minimum number of symbols required to train")
    parser.add_argument("--output-dir", default=BaseConfig.MODEL_DIR,
                      help="Directory to save the trained model")
    args = parser.parse_args()
    
    # Print banner for more visibility
    print("\n===== LGBM MODEL RETRAINING =====\n")
    logger.info("Starting model retraining")
    
    # Validate arguments
    if not args.symbols:
        logger.error("No symbols provided for training")
        print("❌ No symbols provided for training")
        return
        
    logger.info(f"Starting model retraining with symbols: {args.symbols}")
    
    try:
        # Load data for all specified symbols
        dfs = load_data(args.symbols)
        
        if len(dfs) < args.min_symbols:
            logger.error(f"Not enough symbols with valid data. Found {len(dfs)}, need at least {args.min_symbols}")
            print(f"\n❌ Not enough symbols with valid data. Found {len(dfs)}, need at least {args.min_symbols}")
            return
            
        # Print quick summary of loaded data
        print(f"\nLoaded data for {len(dfs)} symbols:")
        for symbol, df in dfs.items():
            print(f"- {symbol}: {len(df)} rows")
            
        # Split data for cross-validation
        train_dfs, val_dfs, test_dfs = split_data(dfs)
        
        # Create standardized feature pipeline - THIS IS THE KEY CHANGE
        logger.info("Creating standardized feature pipeline for consistent training and inference")
        feature_pipeline = create_standardized_feature_pipeline(list(dfs.keys()), ModelConfig.FEATURE_CONFIG)
        
        # Save the feature pipeline configuration for future reference and inference
        pipeline_info_path = os.path.join(args.output_dir, 'feature_pipeline_config.json')
        # Remove the actual feature engineer object which is not JSON serializable
        serializable_pipeline = {k: v for k, v in feature_pipeline.items() if k != 'feature_engineer'}
        with open(pipeline_info_path, 'w') as f:
            json.dump(serializable_pipeline, f, indent=4)
        logger.info(f"Saved feature pipeline configuration to {pipeline_info_path}")
        
        # Process training data using the standardized pipeline
        logger.info("Processing training data with standardized pipeline")
        # Process through feature engineering but preserve original splits
        standard_train_dfs = standardize_features_for_prediction(feature_pipeline, train_dfs)
        standard_val_dfs = standardize_features_for_prediction(feature_pipeline, val_dfs)
        
        if not standard_train_dfs or not standard_val_dfs:
            logger.error("No valid data for training after standardization!")
            print("\n❌ No valid data for training after standardization!")
            print("Check the logs for details on why data was filtered out.")
            return
        
        logger.info(f"Training with {len(standard_train_dfs)} symbols: {list(standard_train_dfs.keys())}")
        
        # Save list of trained symbols for reference
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'trained_symbols.txt'), 'w') as f:
            f.write('\n'.join(list(standard_train_dfs.keys())))
        logger.info(f"Saved trained symbols list to: {os.path.join(args.output_dir, 'trained_symbols.txt')}")
        
        # Initialize and train model
        logger.info("Initializing model...")
        model = LGBMmodel(list(standard_train_dfs.keys()), feature_config=ModelConfig.FEATURE_CONFIG, logger=logger)
        
        logger.info("Training model...")
        try:
            # Final verification before training
            for symbol in standard_train_dfs:
                train_df = standard_train_dfs[symbol]
                val_df = standard_val_dfs[symbol]
                
                # Check for NaN values one more time
                train_nans = train_df.isna().sum().sum()
                val_nans = val_df.isna().sum().sum()
                
                if train_nans > 0 or val_nans > 0:
                    logger.warning(f"Found NaNs just before training: {train_nans} in train, {val_nans} in val for {symbol}")
                    # Final NaN filling
                    standard_train_dfs[symbol] = train_df.fillna(0)
                    standard_val_dfs[symbol] = val_df.fillna(0)
            
            model.train(standard_train_dfs, standard_val_dfs)
            logger.info(f"Model trained successfully with {model.feature_count} features")
            
            # Verify feature count consistency with pipeline
            if model.feature_count != feature_pipeline['feature_count']:
                logger.warning(f"Feature count mismatch between model ({model.feature_count}) and pipeline ({feature_pipeline['feature_count']})")
                print(f"\n⚠️ Feature count mismatch: model has {model.feature_count} features but pipeline expected {feature_pipeline['feature_count']}")
            
            # Save the trained model
            model_path = os.path.join(args.output_dir, 'lgbm_model.pkl')
            model.save_model(args.output_dir)
            logger.info(f"Model saved to {os.path.join(args.output_dir, 'lgbm_model.pkl')}")
            
            # Save feature info for reference
            feature_info = {
                'total_features': model.feature_count,
                'symbols_used': list(standard_train_dfs.keys()),
                'features_per_symbol': {symbol: len(standard_train_dfs[symbol].columns) for symbol in standard_train_dfs},
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'remove_returns': DataConfig.REMOVE_RETURNS_FOR_MODEL,
                'pipeline_version': feature_pipeline['version']
            }
            
            with open(os.path.join(args.output_dir, 'feature_info.json'), 'w') as f:
                json.dump(feature_info, f, indent=4)
            logger.info(f"Feature info saved to {os.path.join(args.output_dir, 'feature_info.json')}")
            
            print("\n✅ Model retrained and saved successfully!")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            print(f"\n❌ Model training failed: {str(e)}")
            
            # Try to analyze what went wrong
            for symbol in standard_train_dfs:
                logger.info(f"Training data for {symbol}: {standard_train_dfs[symbol].shape}, NaN values: {standard_train_dfs[symbol].isna().sum().sum()}")
                logger.info(f"Validation data for {symbol}: {standard_val_dfs[symbol].shape}, NaN values: {standard_val_dfs[symbol].isna().sum().sum()}")
            
            return
        
        # Validate the model works with live data format
        logger.info("Validating model with live-format data...")
        live_dfs = {}
        for symbol in standard_train_dfs:
            live_file = f'data/{symbol}_1h.csv'
            if os.path.exists(live_file):
                live_df = pd.read_csv(live_file)
                live_dfs[symbol] = live_df
                logger.info(f"Loaded {len(live_df)} rows of live-format data for {symbol}")
        
        if live_dfs:
            try:
                # CRITICAL CHANGE: First standardize the live data through the same pipeline
                logger.info("Standardizing live data through the same feature pipeline")
                standardized_live_dfs = standardize_live_data_for_prediction(args.output_dir, live_dfs)
                
                # Now make predictions using the standardized data
                logger.info("Making predictions with standardized live data")
                predictions = model.predict(standardized_live_dfs)
                logger.info(f"Model correctly predicted {len(predictions)} symbols in live data format")
                print("\n✅ Model validation with live data format successful!")
                
                # Display prediction summary
                for symbol, pred in predictions.items():
                    if len(pred) > 0:
                        logger.info(f"Predictions for {symbol}: min={pred.min():.6f}, max={pred.max():.6f}, mean={pred.mean():.6f}")
                        print(f"Predictions for {symbol}: {len(pred)} values, avg={pred.mean():.6f}")
            except Exception as e:
                logger.error(f"Model validation with live data failed: {str(e)}")
                print(f"\n⚠️ Model validation with live data failed: {str(e)}")
                print("The model may still work in production, but you should investigate this issue.")
                
                # Provide more detailed diagnostics
                try:
                    # Check if the standardization worked but prediction failed
                    if 'standardized_live_dfs' in locals():
                        for symbol in standardized_live_dfs:
                            logger.info(f"Standardized live data for {symbol}: {standardized_live_dfs[symbol].shape} vs model expected {model.feature_count} total features")
                            
                        # Check for feature count mismatch
                        total_features = sum(df.shape[1] for df in standardized_live_dfs.values())
                        if total_features != model.feature_count:
                            logger.error(f"Feature count mismatch: model expects {model.feature_count} features but live data has {total_features}")
                            
                            # Try to identify which symbol is causing the problem
                            for symbol, df in standardized_live_dfs.items():
                                expected_symbol_features = feature_pipeline['feature_counts_per_symbol'].get(symbol, 0)
                                actual_features = df.shape[1]
                                if expected_symbol_features != actual_features:
                                    logger.error(f"Feature count mismatch for {symbol}: expected {expected_symbol_features}, got {actual_features}")
                                    # List the actual columns for debugging
                                    logger.debug(f"Actual columns for {symbol}: {df.columns.tolist()}")
                except Exception as diagnostic_error:
                    logger.error(f"Error during diagnostics: {str(diagnostic_error)}")
        else:
            logger.warning("No live-format data available for validation")
            print("\n⚠️ No live-format data available for validation")
    
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        print(f"\n❌ Retraining failed: {str(e)}")
        logger.debug(traceback.format_exc())
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
        print("\n✅ Model retraining process completed!")
        print(f"Model saved to: {BaseConfig.MODEL_DIR}")
    except Exception as e:
        logger.exception(f"Error during model retraining: {str(e)}")
        print("\n❌ Model retraining failed. See logs for details.")
        sys.exit(1) 