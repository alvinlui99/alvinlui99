import os
import pandas as pd
import numpy as np
from model.LGBMmodel import LGBMmodel
from utils.feature_engineering import FeatureEngineer
from config import ModelConfig, BaseConfig
import logging
from utils import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_feature_dimensions():
    # Use a small set of symbols
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    # 1. Load historical training data
    logger.info("Loading training data...")
    train_data = {}
    for symbol in test_symbols:
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
            train_data[symbol] = df
    
    # Split into train/val sets - limited number of rows for quick testing
    train_sets = {}
    val_sets = {}
    for symbol, df in train_data.items():
        # Take a small slice of data for testing
        if len(df) > 1000:
            df = df.iloc[:1000]
        split_idx = int(len(df) * 0.8)
        train_sets[symbol] = df.iloc[:split_idx].copy()
        val_sets[symbol] = df.iloc[split_idx:].copy()
    
    # 2. Create and train model
    logger.info("Training model...")
    model = LGBMmodel(test_symbols, feature_config=ModelConfig.FEATURE_CONFIG, logger=logger)
    
    # Train the model to capture feature dimensions
    model.train(train_sets, val_sets)
    
    # 3. Save trained dimensions
    training_features = model.feature_count
    logger.info(f"Model trained with {training_features} features")
    
    # 4. Analyze training features per symbol
    logger.info("Analyzing training features per symbol...")
    # This simulates part of the internal process of _convert_klines
    training_analysis = {}
    feature_engineer = FeatureEngineer(ModelConfig.FEATURE_CONFIG, logger=logger)
    
    for symbol, df in train_sets.items():
        # Ensure data types are correct for TA-Lib functions
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Add features
        processed_df = feature_engineer.add_all_features(df)
        training_analysis[symbol] = {
            'features_per_symbol': processed_df.shape[1],
            'sample_features': list(processed_df.columns)[:5]
        }
        logger.info(f"Training data {symbol}: {processed_df.shape[1]} features")
        
    # Calculate expected feature dimensions for training
    expected_train_features = sum(info['features_per_symbol'] for info in training_analysis.values())
    logger.info(f"Expected training features: {len(test_symbols)} symbols × ~{training_analysis[test_symbols[0]]['features_per_symbol']} features/symbol = {expected_train_features}")
    
    # 5. Load live data examples
    logger.info("Loading live data...")
    live_data = {}
    for symbol in test_symbols:
        filename = f'data/{symbol}_1h.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            live_data[symbol] = df
    
    # 6. Analyze live features per symbol
    logger.info("Analyzing live features per symbol...")
    live_analysis = {}
    
    for symbol, df in live_data.items():
        # Ensure data types are correct for TA-Lib functions
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Add features
        processed_df = feature_engineer.add_all_features(df)
        live_analysis[symbol] = {
            'features_per_symbol': processed_df.shape[1],
            'sample_features': list(processed_df.columns)[:5]
        }
        logger.info(f"Live data {symbol}: {processed_df.shape[1]} features")
        
    # Calculate expected feature dimensions for live data
    expected_live_features = sum(info['features_per_symbol'] for info in live_analysis.values())
    logger.info(f"Expected live features: {len(test_symbols)} symbols × ~{live_analysis[test_symbols[0]]['features_per_symbol']} features/symbol = {expected_live_features}")
    
    # 7. Try prediction with live data
    logger.info("Running prediction with live data...")
    try:
        # First test with compatibility mode disabled
        os.environ["ENABLE_MODEL_COMPATIBILITY"] = "false"
        predictions = model.predict(live_data)
        logger.info("Prediction successful without compatibility mode!")
    except ValueError as e:
        logger.warning(f"Prediction failed without compatibility mode: {str(e)}")
        
        # Try with compatibility mode enabled
        logger.info("Enabling compatibility mode...")
        os.environ["ENABLE_MODEL_COMPATIBILITY"] = "true"
        try:
            predictions = model.predict(live_data)
            logger.info("Prediction successful with compatibility mode!")
        except Exception as e:
            logger.error(f"Prediction failed even with compatibility mode: {str(e)}")
    
    # 8. Direct analysis from model's convert_klines
    logger.info("Analyzing feature counts from model conversion...")
    input_data, _ = model._convert_klines(live_data)
    live_features = input_data.shape[1]
    logger.info(f"Actual live data features after model conversion: {live_features}")
    
    # 9. Print feature differences
    logger.info(f"Feature difference: {live_features - training_features}")
    
    # 10. Test prediction with same data (should work)
    logger.info("Testing prediction with training data...")
    try:
        predictions = model.predict(train_sets)
        logger.info("Prediction with training data successful!")
    except Exception as e:
        logger.error(f"Prediction with training data failed: {str(e)}")
    
    # 11. Check for raw column differences
    logger.info("Analyzing raw column differences...")
    train_raw_cols = set(list(train_sets[test_symbols[0]].columns))
    live_raw_cols = set(list(live_data[test_symbols[0]].columns))
    if train_raw_cols != live_raw_cols:
        logger.info(f"Raw columns in training but not in live: {train_raw_cols - live_raw_cols}")
        logger.info(f"Raw columns in live but not in training: {live_raw_cols - train_raw_cols}")
        
    return {
        'training_features': training_features,
        'live_features': live_features,
        'difference': live_features - training_features,
        'expected_train_features': expected_train_features,
        'expected_live_features': expected_live_features,
        'symbols_count': len(test_symbols),
        'features_per_symbol_train': training_analysis[test_symbols[0]]['features_per_symbol'],
        'features_per_symbol_live': live_analysis[test_symbols[0]]['features_per_symbol']
    }

if __name__ == "__main__":
    results = test_model_feature_dimensions()
    print("\nSummary of Results:")
    print(f"Symbols used: {results['symbols_count']}")
    print(f"Training features per symbol: {results['features_per_symbol_train']}")
    print(f"Live features per symbol: {results['features_per_symbol_live']}")
    print(f"Total training features: {results['training_features']}")
    print(f"Total live features: {results['live_features']}")
    print(f"Difference: {results['difference']} features")
    print(f"Expected training features: {results['expected_train_features']}")
    print(f"Expected live features: {results['expected_live_features']}")
    
    if results['difference'] > 0:
        print(f"\nRecommendation: Set ENABLE_MODEL_COMPATIBILITY=true or retrain model") 