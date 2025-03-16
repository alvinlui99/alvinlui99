# Standard library imports
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import pandas as pd
import argparse
from datetime import datetime

# Third-party imports

# Local application/library specific imports
from config import BaseConfig, ModelConfig, DataConfig
from utils import utils
from model.LGBMmodel import LGBMmodel
from utils.feature_engineering import FeatureEngineer

def setup_custom_logging():
    """Set up custom logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'logs/train_model_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def prepare_production_like_data(symbols, source_files, logger):
    """
    Prepare data that exactly matches the production data format.
    Ensures consistent feature generation between training and inference.
    
    Args:
        symbols: List of symbols to include
        source_files: Dictionary mapping source file patterns
        logger: Logger instance
        
    Returns:
        Dictionary of prepared DataFrames by symbol
    """
    logger.info(f"Preparing production-like data for {len(symbols)} symbols")
    
    dfs = {}
    for symbol in symbols:
        # Try to find the file for this symbol
        filename = source_files.get(symbol)
        if not filename or not os.path.exists(filename):
            logger.warning(f"Data file for {symbol} not found at {filename}")
            continue
            
        # Load the data
        df = pd.read_csv(filename)
        logger.info(f"Loaded {len(df)} rows for {symbol} from {filename}")
        
        # Apply same transformations as in production
        # 1. Rename columns if needed
        if 'Open time' in df.columns:
            df = df.rename(columns={
                'Open time': 'Open_time',
                'Close time': 'Close_time',
                'Quote asset volume': 'Quote_asset_volume',
                'Number of trades': 'Number_of_trades',
                'Taker buy base asset volume': 'Taker_buy_base_asset_volume',
                'Taker buy quote asset volume': 'Taker_buy_quote_asset_volume'
            })
            
        # 2. Convert date columns
        if 'Open_time' in df.columns:
            # Check if already datetime
            if not pd.api.types.is_datetime64_dtype(df['Open_time']):
                try:
                    df['Open_time'] = pd.to_datetime(df['Open_time'])
                except:
                    # Try millisecond timestamp conversion
                    try:
                        df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
                    except Exception as e:
                        logger.error(f"Failed to convert Open_time for {symbol}: {str(e)}")
        
        # 3. Ensure numeric columns are properly typed
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # 4. Calculate returns if configured in DataConfig
        # This matches what would happen in the DataFetcher
        if DataConfig.CALCULATE_RETURNS:
            df['Return'] = df['Close'].pct_change()
            
        if DataConfig.CALCULATE_LOG_RETURNS:
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
        # 5. *Critical step* - if REMOVE_RETURNS_FOR_MODEL is True, remove returns
        # This ensures consistency with live inference under REMOVE_RETURNS_FOR_MODEL=True
        if DataConfig.REMOVE_RETURNS_FOR_MODEL:
            if 'Return' in df.columns:
                df = df.drop(['Return'], axis=1)
            if 'Log_Return' in df.columns:
                df = df.drop(['Log_Return'], axis=1)
            logger.info(f"Removed Return columns from {symbol} data for model consistency")
                
        # Store the processed dataframe
        dfs[symbol] = df
        
    # Log feature counts to verify consistency
    if dfs:
        feature_engineer = FeatureEngineer(ModelConfig.FEATURE_CONFIG)
        feature_counts = {}
        for symbol, df in dfs.items():
            try:
                processed = feature_engineer.add_all_features(df.copy())
                feature_counts[symbol] = processed.shape[1]
            except Exception as e:
                logger.error(f"Error calculating features for {symbol}: {str(e)}")
                feature_counts[symbol] = "ERROR"
                
        logger.info(f"Feature counts per symbol: {feature_counts}")
        
    return dfs

def check_model_consistency(model, logger):
    """Verify model feature count and other properties."""
    if not hasattr(model, 'feature_count'):
        logger.warning("Model does not have feature_count attribute!")
        return
        
    # This will be the expected feature count during inference
    logger.info(f"Model trained with {model.feature_count} total features")
    logger.info(f"Feature counts per symbol stored: {model.expected_feature_counts}")
    
    # Store this information for reference
    feature_info = {
        'total_features': model.feature_count,
        'per_symbol_counts': model.expected_feature_counts,
        'symbols_used': model.symbols,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save feature info to a file for future reference
    os.makedirs(BaseConfig.MODEL_DIR, exist_ok=True)
    with open(os.path.join(BaseConfig.MODEL_DIR, 'feature_info.txt'), 'w') as f:
        for key, value in feature_info.items():
            f.write(f"{key}: {value}\n")
            
    logger.info(f"Saved feature information to {os.path.join(BaseConfig.MODEL_DIR, 'feature_info.txt')}")

def main():
    """Main function to train the model and evaluate predictions."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the LGBM model with production-consistent data')
    parser.add_argument('--symbols', type=str, nargs='+', help='Specific symbols to train on (default: all from BaseConfig)')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    parser.add_argument('--save-dir', type=str, default='model/trained_models', help='Directory to save the model')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_custom_logging()
    logger.info("Starting model training process (PRODUCTION CONSISTENT VERSION)")
    
    try:
        # Determine which symbols to use
        symbols_to_use = args.symbols if args.symbols else BaseConfig.SYMBOLS
        logger.info(f"Training model for symbols: {symbols_to_use}")
        
        # Configure data sources - prioritize production-like data formats
        data_sources = {}
        for symbol in symbols_to_use:
            # Check for live data format first (this is what will be used in production)
            live_path = f'data/{symbol}_1h.csv'
            training_path = f'data/klines_{symbol}.csv'
            
            if os.path.exists(live_path):
                data_sources[symbol] = live_path
                logger.info(f"Using production format data for {symbol}: {live_path}")
            elif os.path.exists(training_path):
                data_sources[symbol] = training_path
                logger.info(f"Using training format data for {symbol}: {training_path}")
            else:
                logger.warning(f"No data found for {symbol}")
                
        # Check if we have enough symbols with data
        available_symbols = list(data_sources.keys())
        if not available_symbols:
            logger.error("No data available for any symbols! Aborting.")
            return
            
        logger.info(f"Proceeding with {len(available_symbols)} symbols: {available_symbols}")
        
        # Prepare data in a production-consistent format
        logger.info("Preparing production-consistent data")
        dfs = prepare_production_like_data(available_symbols, data_sources, logger)
        
        # Check if all symbols have data after preparation
        symbols_with_data = list(dfs.keys())
        if not symbols_with_data:
            logger.error("No data available after preparation! Aborting.")
            return
            
        logger.info(f"Final symbols for training: {symbols_with_data}")
        
        # Save list of trained symbols for reference during inference
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'trained_symbols.txt'), 'w') as f:
            f.write('\n'.join(symbols_with_data))
        logger.info(f"Saved list of trained symbols to {os.path.join(args.save_dir, 'trained_symbols.txt')}")
        
        # Split the data
        logger.info("Splitting data into train/val/test sets")
        train_dfs, val_dfs, test_dfs = utils.split_dfs(dfs)
        logger.info(f"Train data rows: {[len(df) for df in train_dfs.values()]}")
        logger.info(f"Val data rows: {[len(df) for df in val_dfs.values()]}")
        logger.info(f"Test data rows: {[len(df) for df in test_dfs.values()]}")
        
        # Create and train model
        logger.info("Initializing model")
        model = LGBMmodel(symbols_with_data, feature_config=ModelConfig.FEATURE_CONFIG, logger=logger)
        
        logger.info("Training model")
        model.train(train_dfs, val_dfs)
        
        # Verify model feature consistency
        check_model_consistency(model, logger)
        
        # Save model
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f"Saving trained model to {args.save_dir}")
        model.save_model(args.save_dir)
        
        # Generate predictions
        logger.info("Generating predictions on test data")
        predictions = model.predict(test_dfs)

        # Evaluate results
        logger.info("Evaluating model performance")
        for symbol in symbols_with_data:
            if symbol not in test_dfs or symbol not in predictions:
                logger.warning(f"Missing test data or predictions for {symbol}")
                continue
                
            actual = test_dfs[symbol]['Close'].pct_change(fill_method=None).shift(-1).values[:-1]
            predicted = predictions[symbol]
            
            # Check if predictions are available
            if len(predicted) > 0:
                # Calculate RMSE
                min_len = min(len(actual), len(predicted))
                if min_len > 0:
                    actual = actual[:min_len]
                    predicted = predicted[:min_len]
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                    logger.info(f"RMSE for {symbol}: {rmse}")
                    
                    # Plot predictions vs actual
                    plt.figure(figsize=(12, 6))
                    plt.plot(actual, label='Actual')
                    plt.plot(predicted, label='Predicted')
                    plt.title(f'Actual vs Predicted Returns for {symbol}')
                    plt.legend()
                    os.makedirs('plots', exist_ok=True)
                    plt.savefig(f'plots/{symbol}_predictions.png')
                    plt.close()
                else:
                    logger.warning(f"Not enough data points to calculate RMSE for {symbol}")
            else:
                logger.warning(f"No predictions available for {symbol}")
        
        # Try running prediction with the same model we just trained, but with live-like data
        # to verify consistency between training and inference
        logger.info("Verifying inference consistency...")
        
        # Use original data without splitting to simulate production data
        try:
            logger.info("Testing prediction with original data format")
            live_predictions = model.predict(dfs)
            logger.info("✓ Prediction with original format successful!")
        except Exception as e:
            logger.error(f"✗ Prediction with original format failed: {str(e)}")
            
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.exception(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 