import os
import pandas as pd
import numpy as np
from utils.feature_engineering import FeatureEngineer
import logging
from config import ModelConfig, BaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_features():
    """Count features in training vs live data with the same feature engineering process."""
    # Set test symbols - use all 9 symbols like in your actual training
    test_symbols = BaseConfig.SYMBOLS
    
    # Check which symbols have both training and live data
    available_symbols = []
    for symbol in test_symbols:
        train_file = f'data/klines_{symbol}.csv'
        live_file = f'data/{symbol}_1h.csv'
        if os.path.exists(train_file) and os.path.exists(live_file):
            available_symbols.append(symbol)
    
    if not available_symbols:
        logger.error("No symbols have both training and live data!")
        return None
        
    logger.info(f"Using {len(available_symbols)} symbols for testing: {available_symbols}")
    
    # Create feature engineer with model config
    feature_engineer = FeatureEngineer(ModelConfig.FEATURE_CONFIG)
    
    # Process one symbol at a time - count features after feature engineering
    symbol_features = {}
    for symbol in available_symbols:
        symbol_features[symbol] = {}
        
        # 1. Training data
        train_df = pd.read_csv(f'data/klines_{symbol}.csv')
        # Standardize column names if needed
        if 'Open time' in train_df.columns:
            train_df = train_df.rename(columns={
                'Open time': 'Open_time',
                'Close time': 'Close_time',
                'Quote asset volume': 'Quote_asset_volume',
                'Number of trades': 'Number_of_trades',
                'Taker buy base asset volume': 'Taker_buy_base_asset_volume',
                'Taker buy quote asset volume': 'Taker_buy_quote_asset_volume'
            })
        
        # Convert types for processing
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        
        # Process with feature engineering - just use a small slice
        try:
            train_processed = feature_engineer.add_all_features(train_df.iloc[:500])
            symbol_features[symbol]['train_features'] = train_processed.shape[1]
            symbol_features[symbol]['train_columns'] = list(train_processed.columns)
            logger.info(f"{symbol} training features: {train_processed.shape[1]}")
        except Exception as e:
            logger.error(f"Error processing training data for {symbol}: {str(e)}")
            continue
        
        # 2. Live data
        live_df = pd.read_csv(f'data/{symbol}_1h.csv')
        
        # Convert types for processing 
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in live_df.columns:
                live_df[col] = pd.to_numeric(live_df[col], errors='coerce')
        
        # Process with feature engineering
        try:
            live_processed = feature_engineer.add_all_features(live_df)
            symbol_features[symbol]['live_features'] = live_processed.shape[1]
            symbol_features[symbol]['live_columns'] = list(live_processed.columns)
            logger.info(f"{symbol} live features: {live_processed.shape[1]}")
        except Exception as e:
            logger.error(f"Error processing live data for {symbol}: {str(e)}")
            continue
            
        # Find differences in columns
        train_cols = set(train_processed.columns)
        live_cols = set(live_processed.columns)
        if train_cols != live_cols:
            logger.info(f"{symbol} columns in training but not in live: {train_cols - live_cols}")
            logger.info(f"{symbol} columns in live but not in training: {live_cols - train_cols}")
    
    # Total up features across all symbols (simulating model's horizontal concatenation)
    total_train_features = sum(info.get('train_features', 0) for info in symbol_features.values())
    total_live_features = sum(info.get('live_features', 0) for info in symbol_features.values())
    
    logger.info(f"Total training features across {len(available_symbols)} symbols: {total_train_features}")
    logger.info(f"Total live features across {len(available_symbols)} symbols: {total_live_features}")
    logger.info(f"Difference: {total_live_features - total_train_features} features")
    
    # Calculate per-symbol averages
    avg_train = total_train_features / len(available_symbols)
    avg_live = total_live_features / len(available_symbols)
    logger.info(f"Average features per symbol - Training: {avg_train:.1f}, Live: {avg_live:.1f}")
    
    return {
        'symbols': available_symbols,
        'total_train_features': total_train_features,
        'total_live_features': total_live_features,
        'difference': total_live_features - total_train_features,
        'avg_train_per_symbol': avg_train,
        'avg_live_per_symbol': avg_live,
        'symbol_features': symbol_features
    }

if __name__ == "__main__":
    results = count_features()
    if results:
        print("\nSummary:")
        print(f"Number of symbols: {len(results['symbols'])}")
        print(f"Average features per symbol in training: {results['avg_train_per_symbol']:.1f}")
        print(f"Average features per symbol in live data: {results['avg_live_per_symbol']:.1f}")
        print(f"Total training features: {results['total_train_features']}")
        print(f"Total live features: {results['total_live_features']}")
        print(f"Difference: {results['difference']} additional features in live data")
        
        if results['difference'] > 0:
            print("\nSolution options:")
            print("1. Use compatibility mode: ENABLE_MODEL_COMPATIBILITY=true")
            print("2. Retrain your model using the same symbols as in production")
            print("3. Modify feature engineering to ensure consistency between training and live data") 